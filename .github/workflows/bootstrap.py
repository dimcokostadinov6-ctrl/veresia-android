#!/usr/bin/env python3
from pathlib import Path

def w(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

root = Path(".")

# -------- core --------
w(root/"core/__init__.py", "")
w(root/"core/ports.py", r'''
from typing import List, Tuple, Protocol, Optional
import numpy as np

class IHandwriteAI(Protocol):
    def recognize_names(self, page_bgr: 'np.ndarray', known_names: List[str], min_conf: float = 0.6) -> List[str]: ...

class IOCR(Protocol):
    def parse_entries(self, image_path: str, user_words: Optional[List[str]] = None) -> List[Tuple[str,int]]: ...

class IRepository(Protocol):
    def init(self) -> None: ...
    def add_page(self, path: str, ts: str) -> int: ...
    def add_entry(self, name: str, amount_st: int, ts: str, page_id: int | None) -> None: ...
    def all_known_names(self) -> List[str]: ...
    def search_by_name(self, q: str) -> List[Tuple[str,int]]: ...
''')

w(root/"core/services.py", r'''
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .ports import IHandwriteAI, IOCR, IRepository
from infra.ai.recognizer import mask_crossed_out_names

@dataclass
class SavePageService:
    repo: IRepository
    ai: IHandwriteAI
    ocr: IOCR

    def save_drawn_page(self, image_path: str, ts_iso: str) -> tuple[int, int]:
        self.last_removed_count = 0
        page_id = self.repo.add_page(image_path, ts_iso)
        try: known = self.repo.all_known_names()
        except Exception: known = []
        ai_names: List[str] = []
        try:
            import cv2, os
            page_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if page_bgr is None:
                masked_path = image_path
            else:
                masked_bgr, removed = mask_crossed_out_names(page_bgr, left_ratio=0.6)
                self.last_removed_count = len(removed)
                base, ext = os.path.splitext(image_path)
                masked_path = base + "_masked" + (ext if ext else ".png")
                cv2.imwrite(masked_path, masked_bgr)
            if page_bgr is not None:
                ai_names = self.ai.recognize_names(page_bgr, known, min_conf=0.6)
        except Exception:
            masked_path = image_path
            ai_names = []
        user_words = list({*(known or []), *ai_names})
        try: entries = self.ocr.parse_entries(masked_path, user_words=user_words)
        except Exception: entries = []
        n = 0
        for name, amount_st in entries:
            if not name or amount_st is None: continue
            self.repo.add_entry(name=name.strip(), amount_st=int(amount_st), ts=ts_iso, page_id=page_id); n += 1
        return page_id, n
''')

# -------- infra --------
w(root/"infra/__init__.py", "")
w(root/"infra/database_sqlite.py", r'''
import sqlite3
from pathlib import Path
from typing import List, Tuple
from core.ports import IRepository

DB_PATH = Path(__file__).resolve().parent.parent / 'veresia.db'

class SQLiteRepo(IRepository):
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
    def _conn(self):
        return sqlite3.connect(str(self.db_path))
    def init(self) -> None:
        with self._conn() as con:
            cur = con.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS pages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                ts TEXT NOT NULL
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS entries(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                amount_st INTEGER NOT NULL,
                ts TEXT NOT NULL,
                page_id INTEGER,
                FOREIGN KEY(page_id) REFERENCES pages(id)
            );""")
            con.commit()
    def add_page(self, path: str, ts: str) -> int:
        with self._conn() as con:
            cur = con.cursor()
            cur.execute('INSERT INTO pages(path, ts) VALUES(?,?)', (path, ts))
            con.commit(); return cur.lastrowid
    def add_entry(self, name: str, amount_st: int, ts: str, page_id: int | None) -> None:
        with self._conn() as con:
            cur = con.cursor()
            cur.execute('INSERT INTO entries(name, amount_st, ts, page_id) VALUES(?,?,?,?)', (name, int(amount_st), ts, page_id))
            con.commit()
    def all_known_names(self) -> List[str]:
        with self._conn() as con:
            cur = con.cursor(); cur.execute('SELECT DISTINCT name FROM entries ORDER BY name COLLATE NOCASE')
            return [r[0] for r in cur.fetchall()]
    def search_by_name(self, q: str) -> List[Tuple[str,int]]:
        q = (q or '').strip()
        with self._conn() as con:
            cur = con.cursor()
            if q:
                cur.execute("""
                    SELECT name, SUM(amount_st) as total
                    FROM entries
                    WHERE name LIKE ?
                    GROUP BY name
                    ORDER BY total DESC
                """, (f"%{q}%",))
            else:
                cur.execute("""
                    SELECT name, SUM(amount_st) as total
                    FROM entries
                    GROUP BY name
                    ORDER BY total DESC
                """)
            return [(r[0], int(r[1] or 0)) for r in cur.fetchall()]
''')

w(root/"infra/ocr_bul.py", r'''
import re
from typing import List, Tuple, Optional, Union
from pathlib import Path
from core.ports import IOCR

class TesseractOCR(IOCR):
    def _try_pytesseract_on_image(self, image_path: Union[str, Path]) -> str:
        try:
            import cv2, pytesseract
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None: return ''
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return pytesseract.image_to_string(gray, lang='bul+eng') or ''
        except Exception:
            return ''
    def _extract_lines(self, text: str) -> list:
        return [ln.strip() for ln in (text or '').splitlines() if ln.strip()]
    def _parse_amount(self, s: str) -> Optional[int]:
        nums = re.findall(r'([0-9]+(?:[\\.,][0-9]{1,2})?)', s.replace(' ', ''))
        if not nums: return None
        val = nums[-1].replace(',', '.')
        try: return int(round(float(val)*100))
        except Exception: return None
    def _name_part(self, s: str) -> str:
        import re as _re
        parts = _re.split(r'([0-9]+(?:[\\.,][0-9]{1,2})?)', s)
        if not parts: return ''
        return (''.join(parts[:-2]).strip(' -:/\\t') if len(parts) >= 3 else s.strip())
    def parse_entries(self, image_path: str, user_words: Optional[list] = None) -> List[Tuple[str, int]]:
        text = self._try_pytesseract_on_image(image_path)
        if not text: return []
        out=[]
        for ln in self._extract_lines(text):
            amt = self._parse_amount(ln)
            if amt is None: continue
            nm = self._name_part(ln).strip()
            if nm: out.append((nm, amt))
        return out
''')

# -------- ai --------
w(root/"infra/ai/__init__.py", "")
w(root/"infra/ai/recognizer.py", r'''
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np, cv2, unicodedata

AI_DIR = Path(__file__).resolve().parent
MODEL_PATH = AI_DIR / "ai_model" / "prototypes.npz"

_LAT_TO_CYR = str.maketrans({"A":"А","a":"а","B":"В","E":"Е","e":"е","K":"К","k":"к","M":"М","m":"м","H":"Н","O":"О","o":"о","P":"Р","p":"р","C":"С","c":"с","T":"Т","t":"т","X":"Х","x":"х","Y":"У","y":"у"})
def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "").strip()).translate(_LAT_TO_CYR).casefold()
    return s

def _binarize(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3,3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    return cv2.medianBlur(bw,3)

def _resize_keep_ar(img: np.ndarray, size=32) -> np.ndarray:
    h,w = img.shape; scale = size/max(1,max(h,w)); nh,nw = int(round(h*scale)), int(round(w*scale))
    rs = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size,size), np.uint8); y=(size-nh)//2; x=(size-nw)//2; canvas[y:y+nh, x:x+nw]=rs; return canvas

def char_embed(char_img: np.ndarray) -> np.ndarray:
    if char_img.ndim==3: char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    bw = _binarize(char_img); bw = _resize_keep_ar(bw, 32)
    pix = (bw.astype(np.float32)/255.0).reshape(-1)
    gx = cv2.Sobel(bw, cv2.CV_32F, 1,0,ksize=3); gy = cv2.Sobel(bw, cv2.CV_32F, 0,1,ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    ang_bins = (ang * (8.0/(2.0*np.pi))).astype(np.int32) % 8
    H = np.zeros((4,4,8), np.float32); cell=8
    for cy in range(4):
        for cx in range(4):
            y0,y1 = cy*cell, (cy+1)*cell; x0,x1 = cx*cell, (cx+1)*cell
            m = mag[y0:y1, x0:x1]; b = ang_bins[y0:y1, x0:x1]
            hist = np.bincount(b.reshape(-1), weights=m.reshape(-1), minlength=8)
            H[cy,cx,:]=hist
    hog = H.reshape(-1); hog = hog/(np.linalg.norm(hog)+1e-6)
    return np.concatenate([pix, hog])

@dataclass
class ProtoModel:
    labels: List[str]
    prototypes: np.ndarray
    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> Optional['ProtoModel']:
        if not path.exists(): return None
        d = np.load(path, allow_pickle=True)
        return cls(list(d['labels']), d['prototypes'].astype(np.float32))
    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=np.array(self.labels, dtype=object), prototypes=self.prototypes.astype(np.float32))
    def predict_chars(self, char_imgs: List[np.ndarray]):
        if self.prototypes.size==0: return [], []
        p = self.prototypes; p = p/(np.linalg.norm(p,axis=1,keepdims=True)+1e-6)
        outs, confs = [], []
        for ci in char_imgs:
            e = char_embed(ci).astype(np.float32); e = e/(np.linalg.norm(e)+1e-6)
            sims = p @ e; k = int(np.argmax(sims)); outs.append(self.labels[k]); confs.append(float((sims[k]+1)/2))
        return outs, confs

def extract_word_boxes(page_bgr: np.ndarray, left_ratio=0.6):
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY) if page_bgr.ndim==3 else page_bgr
    bw = _binarize(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,3))
    merge = cv2.dilate(bw, kernel, 1)
    contours,_ = cv2.findContours(merge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H,W = bw.shape; max_x = int(W*left_ratio); boxes=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if x>max_x or h<16 or w<20 or w*h<200 or h>H*0.35: continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: (b[1]//40, b[0])); return boxes

def segment_chars(word_img: np.ndarray):
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY) if word_img.ndim==3 else word_img
    bw = _binarize(gray); proj = (bw>0).sum(axis=0)
    cuts, in_gap = [], True
    for x,val in enumerate(proj):
        if in_gap and val>0: start=x; in_gap=False
        elif (not in_gap) and val==0: cuts.append((start,x)); in_gap=True
    if not in_gap: cuts.append((start,len(proj)-1))
    chars=[]; import numpy as np
    for x0,x1 in cuts:
        if x1-x0+1<2: continue
        crop = bw[:, max(0,x0-1):min(bw.shape[1], x1+2)]
        rows = (crop>0).sum(axis=1); ys = np.where(rows>0)[0]
        if len(ys)==0: continue
        y0,y1 = ys[0], ys[-1]; chars.append(crop[y0:y1+1,:])
    return chars

def decode_word(model: ProtoModel, word_img: np.ndarray):
    chars = segment_chars(word_img)
    if not chars: return "", 0.0
    preds, confs = model.predict_chars(chars)
    return "".join(preds), (float(np.mean(confs)) if confs else 0.0)

def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur = [i]
        for j,cb in enumerate(b,1):
            cost = 0 if ca==cb else 1
            cur.append(min(prev[j]+1, cur[-1]+1, prev[j-1]+cost))
        prev = cur
    return prev[-1]

def best_name_candidates(raw: str, lexicon: List[str], topk=3):
    nrm = norm_name(raw)
    scored=[]
    for name in lexicon:
        d = levenshtein(nrm, norm_name(name))
        m = max(1, max(len(nrm), len(name)))
        sim = 1.0 - d/m; scored.append((name, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]

def recognize_names_page(page_bgr: np.ndarray, known_names: List[str], min_conf=0.55, topk=3, left_ratio=0.6):
    model = ProtoModel.load(MODEL_PATH)
    if model is None: return []
    boxes = extract_word_boxes(page_bgr, left_ratio=left_ratio)
    out=[]
    for (x,y,w,h) in boxes:
        crop = page_bgr[y:y+h, x:x+w]
        raw, conf = decode_word(model, crop)
        if conf < min_conf or not raw: continue
        cand = best_name_candidates(raw, known_names, topk=topk)
        if not cand: continue
        best, sim = cand[0]; final = float(min(conf, sim))
        if final < min_conf: continue
        out.append((best, final, (x,y,w,h)))
    seen=set(); uniq=[]
    for name,sc,box in out:
        key=(norm_name(name), box[1]//40)
        if key in seen: continue
        seen.add(key); uniq.append((name,sc,box))
    return uniq

# Strict horizontal back-and-forth strike-through
def is_crossed_out(word_img: np.ndarray) -> bool:
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY) if word_img.ndim==3 else word_img
    h, w = gray.shape[:2]
    if w < 30 or h < 15:
        return False
    edges = cv2.Canny(gray, 50, 150)
    min_len = max(12, int(0.65 * w))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=min_len, maxLineGap=12)
    if lines is None:
        return False
    horiz = []
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        dx, dy = x2-x1, y2-y1
        L = max(1, int(np.hypot(dx, dy)))
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) <= 15 or abs(abs(ang)-180) <= 15:
            horiz.append((x1,y1,x2,y2, dx, dy, L, ang))
    if len(horiz) < 3:
        return False
    has_pos = any(dx > 0 for *_, dx, __, ___, ____ in horiz)
    has_neg = any(dx < 0 for *_, dx, __, ___, ____ in horiz)
    if not (has_pos and has_neg):
        return False
    ys = [y1 for _,y1,__,___,____,______,______,_____ in horiz] + [y2 for __,___,____,y2,_____,______,_______,________ in horiz]
    band = max(ys) - min(ys) if ys else h
    if band > 0.45 * h:
        return False
    lengths = [L for *_, L, __ in horiz]
    if np.median(lengths) < 0.70 * w:
        return False
    return True

def mask_crossed_out_names(page_bgr: np.ndarray, left_ratio: float = 0.6):
    boxes = extract_word_boxes(page_bgr, left_ratio=left_ratio)
    out = page_bgr.copy()
    removed = []
    for (x,y,w,h) in boxes:
        crop = page_bgr[y:y+h, x:x+w]
        if is_crossed_out(crop):
            pad = 3
            cv2.rectangle(out, (max(0,x-pad), max(0,y-pad)), (min(out.shape[1]-1, x+w+pad), min(out.shape[0]-1, y+h+pad)), (255,255,255), thickness=-1)
            removed.append((x,y,w,h))
    return out, removed
''')

w(root/"infra/ai/handwrite_ai.py", r'''
from typing import List
import numpy as np
from core.ports import IHandwriteAI
from .recognizer import recognize_names_page

class ProtoHandwriteAI(IHandwriteAI):
    def recognize_names(self, page_bgr: 'np.ndarray', known_names: List[str], min_conf: float = 0.6) -> List[str]:
        hits = recognize_names_page(page_bgr, known_names=known_names, min_conf=min_conf)
        return [nm for (nm,score,box) in hits]
''')

w(root/"infra/ai/train_prototypes.py", r'''
from pathlib import Path
import numpy as np, cv2
from .recognizer import char_embed, ProtoModel, AI_DIR

def build_prototypes(data_dir: Path) -> ProtoModel:
    labels, vecs = [], []
    for label_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        embs = []
        for img_path in label_dir.glob('*.png'):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            embs.append(char_embed(img))
        if embs:
            labels.append(label_dir.name)
            vecs.append(np.mean(np.stack(embs, axis=0), axis=0))
    if not vecs:
        raise RuntimeError('Няма обучаващи примери.')
    return ProtoModel(labels=labels, prototypes=np.stack(vecs, axis=0).astype(np.float32))
''')

# -------- ui --------
w(root/"ui_kivy/__init__.py", "")
w(root/"ui_kivy/app.py", r'''
from __future__ import annotations
from datetime import datetime
import os, time
from pathlib import Path

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition
from kivy.graphics import Color, Rectangle, Line, InstructionGroup
from kivy.metrics import dp

from core.services import SavePageService
from infra.ai.recognizer import MODEL_PATH, AI_DIR
from infra.ai.train_prototypes import build_prototypes

def _cm_to_dp(cm: float) -> float:
    from kivy.metrics import dp
    return dp(cm * 160.0 / 2.54)

def add_banded_background(widget, white_cm=0.5, purple_cm=0.5, purple_rgba=(0.7,0.6,0.9,0.55)):
    group = InstructionGroup()
    with widget.canvas.before:
        Color(1,1,1,1); bg = Rectangle(pos=widget.pos, size=widget.size)
    widget._ruled_bg_rect = bg
    def redraw(*_):
        try: widget.canvas.before.remove(group)
        except Exception: pass
        group.clear()
        with widget.canvas.before:
            Color(1,1,1,1)
            widget._ruled_bg_rect.pos = widget.pos
            widget._ruled_bg_rect.size = widget.size
            y = widget.y; band_h = _cm_to_dp(white_cm + purple_cm); ph = _cm_to_dp(purple_cm); i = 0
            while y + i*band_h < widget.top:
                Color(*purple_rgba); group.add(Rectangle(pos=(widget.x, y + i*band_h), size=(widget.width, ph))); i += 1
    widget.bind(pos=redraw, size=redraw); widget.canvas.before.add(group); redraw()

class DrawArea(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'; self.padding = dp(8); self.spacing = dp(8)
        self.canvas_box = BoxLayout(size_hint=(1, 1)); self.add_widget(self.canvas_box)
        with self.canvas_box.canvas:
            Color(1,1,1,1); self.bg_rect = Rectangle(size=self.canvas_box.size, pos=self.canvas_box.pos)
        self.canvas_box.bind(size=self._update_bg, pos=self._update_bg)
        add_banded_background(self.canvas_box)
        self._lines = []; self.canvas_box.bind(on_touch_down=self._touch_down, on_touch_move=self._touch_move)
    def _update_bg(self, *_): self.bg_rect.size = self.canvas_box.size; self.bg_rect.pos = self.canvas_box.pos
    def _touch_down(self, widget, touch):
        if not self.canvas_box.collide_point(*touch.pos): return
        with self.canvas_box.canvas: Color(0,0,0,1); self._lines.append(Line(points=[*touch.pos], width=2))
    def _touch_move(self, widget, touch):
        if not self.canvas_box.collide_point(*touch.pos): return
        if self._lines: self._lines[-1].points += [*touch.pos]
    def export_to_png(self, path: str):
        img = self.canvas_box.export_as_image(); img.save(path)

class SearchView(Screen):
    def __init__(self, repo, **kwargs):
        super().__init__(**kwargs); self.repo = repo
        root = BoxLayout(orientation='vertical')
        top = GridLayout(cols=3, size_hint=(1,None), height=dp(64), padding=[dp(12),0,dp(12),0], spacing=dp(8))
        with top.canvas.before: Color(0,0,0,1); self._r = Rectangle(size=top.size, pos=top.pos)
        top.bind(size=lambda *_: self._u(top), pos=lambda *_: self._u(top))
        btn_back = Button(text='← Платно', size_hint=(None,1), width=dp(160), background_normal='', background_color=(1,1,1,1), color=(0,0,0,1), font_size=dp(18))
        btn_back.bind(on_press=lambda *_: setattr(self.manager, 'current', 'draw'))
        top.add_widget(btn_back); top.add_widget(Label(text='Търсене', color=(1,1,1,1), font_size=dp(26))); top.add_widget(Label()); root.add_widget(top)
        bar = BoxLayout(orientation='horizontal', size_hint=(1,None), height=dp(64), padding=[dp(12), dp(8)], spacing=dp(8))
        self.search_input = TextInput(hint_text='🔍 Въведи име…', multiline=False, font_size=dp(18), background_normal='', background_color=(0.95,0.95,0.95,1), foreground_color=(0,0,0,1), padding=[dp(12)])
        btn = Button(text='Търси', size_hint=(None,1), width=dp(100), background_normal='', background_color=(0.2,0.6,1,1), color=(1,1,1,1), font_size=dp(18))
        btn.bind(on_press=self.on_search); bar.add_widget(self.search_input); bar.add_widget(btn); root.add_widget(bar)
        self.results = Label(text='', halign='left', valign='top', markup=True); self.results.bind(size=lambda *_: setattr(self.results,'text_size', self.results.size))
        sv = ScrollView(size_hint=(1,1)); sv.add_widget(self.results); root.add_widget(sv); self.add_widget(root)
    def _u(self, w): self._r.size = w.size; self._r.pos = w.pos
    def on_search(self, *_):
        q = (self.search_input.text or '').strip()
        rows = self.repo.search_by_name(q)
        if not rows: self.results.text = 'Нищо не е намерено.'; return
        lines = [f'• [b]{name}[/b] — {total/100.0:.2f} лв' for name, total in rows]
        self.results.text = '\n'.join(lines)

class TrainerScreen(Screen):
    def __init__(self, on_trained_callback, **kwargs):
        super().__init__(**kwargs); self.on_trained_callback = on_trained_callback
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.uix.button import Button
        from kivy.metrics import dp
        import cv2, numpy as np, time
        from pathlib import Path
        self.RAW_DIR = AI_DIR / "ai_data" / "raw"
        self.LABELS = list('абвгдежзийклмнопрстуфхцчшщъьюя0123456789')
        self.SAMPLES = 40
        def count(l): 
            p=self.RAW_DIR/l; 
            return len(list(p.glob('*.png'))) if p.exists() else 0
        def next_pos():
            for i,lab in enumerate(self.LABELS):
                n=count(lab)
                if n<self.SAMPLES: return i,n
            return len(self.LABELS)-1, self.SAMPLES
        self._count=count; self._next=next_pos

        root = BoxLayout(orientation='vertical')
        bar = BoxLayout(orientation='horizontal', size_hint=(1,None), height=dp(56), padding=[dp(12),0], spacing=dp(8))
        self.title = Label(text='[b]Първоначално обучение — напиши показаната буква[/b]', markup=True, font_size=dp(18))
        btn_train = Button(text='Тренирай и стартирай', size_hint=(None,1), width=dp(220)); btn_train.bind(on_press=self._train_and_go)
        bar.add_widget(self.title); bar.add_widget(btn_train); root.add_widget(bar)

        self.label_idx, self.sample_idx = self._next()
        mid = BoxLayout(orientation='vertical', size_hint=(1,1), padding=[dp(8)], spacing=dp(8))
        self.lbl = Label(text=self._label_text(), markup=True, size_hint=(1,None), height=dp(32)); mid.add_widget(self.lbl)
        self.canvas_box = BoxLayout(size_hint=(1,1)); mid.add_widget(self.canvas_box)
        with self.canvas_box.canvas:
            Color(1,1,1,1); self.bg=Rectangle(pos=self.canvas_box.pos, size=self.canvas_box.size)
        self.canvas_box.bind(pos=lambda *_: self._upd_bg(), size=lambda *_: self._upd_bg())
        self.lines=[]; self.canvas_box.bind(on_touch_down=self._down, on_touch_move=self._move)
        bot = BoxLayout(orientation='horizontal', size_hint=(1,None), height=dp(56), padding=[dp(8)], spacing=dp(8))
        btn_clear = Button(text='Изчисти', size_hint=(None,1), width=dp(100)); btn_clear.bind(on_press=self._clear)
        btn_save = Button(text='Запази', size_hint=(1,1)); btn_save.bind(on_press=self._save)
        bot.add_widget(btn_clear); bot.add_widget(btn_save)
        root.add_widget(mid); root.add_widget(bot); self.add_widget(root)

    def _label_text(self):
        if self.label_idx >= len(self.LABELS): return 'Готово!'
        return f'Буква: [b]{self.LABELS[self.label_idx]}[/b]  ({self.sample_idx+1}/{self.SAMPLES})'

    def _upd_bg(self): self.bg.pos=self.canvas_box.pos; self.bg.size=self.canvas_box.size
    def _down(self, w, t):
        if not self.canvas_box.collide_point(*t.pos): return
        with self.canvas_box.canvas: Color(0,0,0,1); self.lines.append(Line(points=[*t.pos], width=2))
    def _move(self, w, t):
        if not self.canvas_box.collide_point(*t.pos): return
        if self.lines: self.lines[-1].points += [*t.pos]
    def _clear(self, *_):
        self.canvas_box.canvas.clear()
        with self.canvas_box.canvas: Color(1,1,1,1); self.bg=Rectangle(pos=self.canvas_box.pos, size=self.canvas_box.size)
        self.lines=[]

    def _save(self, *_):
        import numpy as np, cv2, time
        fbo = self.canvas_box.export_as_image()
        arr = np.frombuffer(fbo.texture.pixels, np.uint8).reshape(fbo.texture.height, fbo.texture.width, 4)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,10)
        ys, xs = np.where(bw>0)
        if len(xs)==0: return
        y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = gray[y0:y1+1, x0:x1+1]
        lab = self.LABELS[self.label_idx]
        out = (self.RAW_DIR/lab); out.mkdir(parents=True, exist_ok=True)
        import cv2 as _c; _c.imwrite(str(out/f'{int(time.time()*1000)}.png'), crop)
        self.sample_idx = self._count(lab)
        if self.sample_idx >= self.SAMPLES and self.label_idx < len(self.LABELS)-1:
            self.label_idx += 1; self.sample_idx = self._count(self.LABELS[self.label_idx])
        self._clear(); self.lbl.text = self._label_text()

    def _train_and_go(self, *_):
        try:
            data_dir = AI_DIR / "ai_data" / "raw"
            out = AI_DIR / "ai_model" / "prototypes.npz"
            m = build_prototypes(data_dir)
            out.parent.mkdir(parents=True, exist_ok=True); m.save(out)
        except Exception as e:
            print("TRAIN ERROR:", e); return
        if callable(self.on_trained_callback): self.on_trained_callback()

class DrawView(Screen):
    def __init__(self, service: SavePageService, **kwargs):
        super().__init__(**kwargs); self.service = service
        root = BoxLayout(orientation='vertical')
        top = GridLayout(cols=3, size_hint=(1,None), height=dp(64), padding=[dp(12),0,dp(12),0], spacing=dp(8))
        with top.canvas.before: Color(0,0.4,1,1); self._r = Rectangle(size=top.size, pos=top.pos)
        top.bind(size=lambda *_: self._u(top), pos=lambda *_: self._u(top))
        btn_save = Button(text='Запазване', size_hint=(None,1), width=dp(140), background_normal='', background_color=(1,0.5,0,1), color=(1,1,1,1), font_size=dp(18)); btn_save.bind(on_press=self.on_save); top.add_widget(btn_save)
        btn_retrain = Button(text='Обучи наново', size_hint=(None,1), width=dp(160), background_normal='', background_color=(0.3,0.7,0.3,1), color=(1,1,1,1), font_size=dp(18)); btn_retrain.bind(on_press=lambda *_: setattr(self.manager, 'current', 'trainer')); top.add_widget(btn_retrain)
        title = Button(text='Търсене', background_normal='', background_color=(0,0,0,0), color=(1,1,1,1), font_size=dp(26)); title.bind(on_press=lambda *_: setattr(self.manager, 'current', 'search')); top.add_widget(title)
        root.add_widget(top)
        self.draw_area = DrawArea(); root.add_widget(self.draw_area)
        self.status = Label(text='Готово', size_hint=(1,None), height=dp(28), color=(0,0,0,1), font_size=dp(14), halign='left', valign='middle'); root.add_widget(self.status)
        self.add_widget(root)
    def _u(self, w): self._r.size = w.size; self._r.pos = w.pos
    def set_status(self, msg): self.status.text = msg; print('[SAVE]', msg)
    def on_save(self, *_):
        os.makedirs('pages', exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S'); path = os.path.join('pages', f'page_{ts}.png')
        try: self.draw_area.export_to_png(path)
        except Exception as e: self.set_status(f'Грешка при запазване на PNG: {e}'); return
        page_id, n = self.service.save_drawn_page(path, datetime.now().isoformat(timespec='seconds'))
        self.draw_area.canvas_box.canvas.clear()
        with self.draw_area.canvas_box.canvas: Color(1,1,1,1); self.draw_area.bg_rect = Rectangle(size=self.draw_area.canvas_box.size, pos=self.draw_area.canvas_box.pos)
        add_banded_background(self.draw_area.canvas_box)
        skipped = getattr(self.service, 'last_removed_count', 0)
        self.set_status(f'Запазени {n} реда. Пропуснати (задраскани): {skipped}. PNG: {os.path.basename(path)}')

class VeresiyaApp(App):
    def __init__(self, repo, service: SavePageService, **kwargs):
        super().__init__(**kwargs); self.repo = repo; self.service = service
    def build(self):
        self.repo.init()
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(TrainerScreen(on_trained_callback=lambda : setattr(sm,'current','draw'), name='trainer'))
        sm.add_widget(DrawView(self.service, name='draw'))
        sm.add_widget(SearchView(self.repo, name='search'))
        sm.current = 'draw'  # директно към основния екран; trainer е наличен от бутона
        return sm
''')

# -------- app entry + buildozer --------
w(root/"main.py", r'''
from infra.database_sqlite import SQLiteRepo
from infra.ocr_bul import TesseractOCR
from infra.ai.handwrite_ai import ProtoHandwriteAI
from core.services import SavePageService
from ui_kivy.app import VeresiyaApp

repo = SQLiteRepo()
ocr = TesseractOCR()
ai = ProtoHandwriteAI()
service = SavePageService(repo=repo, ai=ai, ocr=ocr)

if __name__ == '__main__':
    VeresiyaApp(repo=repo, service=service).run()
''')

w(root/"buildozer.spec", r'''
[app]
title = Veresia
package.name = veresia
package.domain = com.yourstore
source.dir = .
source.include_exts = py,kv,png,jpg,ttf,zip,txt,md,npz,db
version = 0.1
requirements = python3,kivy,numpy,opencv,plyer,pillow,pytesseract
orientation = sensor
fullscreen = 0
android.archs = armeabi-v7a, arm64-v8a
android.api = 33
android.minapi = 21
p4a.branch = master
android.permissions =
''')

# done
print("Project generated.")
