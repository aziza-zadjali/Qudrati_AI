# -*- coding: utf-8 -*-
"""
Streamlit page 7: Arabic visual IQ question generator (no API/JSON)
- Paper folding & hole punching
- 2D rotation (Cross-board & Diagonal/X-board) + “رموز متغيّرة”
- 3D isometric cube rotation
- Shape assembly (which set forms the target?)
- Difficulty tiers: سهل / متوسط / صعب
- Thick, crisp strokes (STYLE + render scale)
- Stem panel shows (Reference  →  Arrow banner in Arabic  →  ?)

File path: pages/7_spacial_questions.py
"""

import io
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import streamlit as st

# ------------------------------ Paths ------------------------------
ROOT = Path(__file__).parents[1]
LOGO_PATH = ROOT / "MOL_logo.png"

# ------------------------------ Config ------------------------------
AR_LETTERS = ["أ", "ب", "ج", "د"]
RNG = random.Random()

STYLE = {
    "grid": 6,
    "square": 6,
    "glyph": 6,
    "circle": 5,
    "arrow": 6,
    "iso_edge": 3,
    "dash": 4,
}
CANVAS_SCALE = 1.7  # render bigger -> downscale -> crisp lines

# ------------------ Optional LLM via Ollama -----------------------
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def ollama_chat_or_fallback(system: str, user: str, model: str, enabled: bool = True, max_tokens: int = 256) -> str:
    if not enabled or requests is None:
        return user
    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=8,
        )
        if r.ok:
            out = r.json().get("message", {}).get("content", "").strip()
            if out:
                return out
    except Exception:
        pass
    return user

# ---------------------- Arabic shaping helpers --------------------
def get_ar_font(px: int) -> Optional[ImageFont.FreeTypeFont]:
    """Load a decent Arabic-capable font if available; fallback to DejaVuSans."""
    for name in ["NotoNaskhArabic-Regular.ttf", "NotoKufiArabic-Regular.ttf", "Amiri-Regular.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, int(px))
        except Exception:
            continue
    return None  # PIL will fallback

def shape_ar(text: str) -> str:
    """
    Proper Arabic shaping + bidi for PIL.
    Uses arabic_reshaper + python-bidi if available; otherwise returns the original text.
    """
    try:
        import arabic_reshaper  # type: ignore
        from bidi.algorithm import get_display  # type: ignore

        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)  # make it RTL visually
    except Exception:
        return text  # graceful fallback

# ---------------------- Basic drawing helpers ---------------------
def new_canvas(w: int, h: int, bg=(255, 255, 255)) -> Image.Image:
    W = int(w * CANVAS_SCALE)
    H = int(h * CANVAS_SCALE)
    return Image.new("RGB", (W, H), bg)

def _finalize_for_display(img: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    if target_size:
        return img.resize((int(target_size[0]), int(target_size[1])), Image.LANCZOS)
    if CANVAS_SCALE == 1.0:
        return img
    return img.resize((int(img.width / CANVAS_SCALE), int(img.height / CANVAS_SCALE)), Image.LANCZOS)

def dashed_line(draw: ImageDraw.ImageDraw, start, end, dash_len=10, gap_len=8, fill=(0, 0, 0), width: Optional[int] = None):
    if width is None:
        width = STYLE["dash"]
    x1, y1 = start
    x2, y2 = end
    total_len = math.hypot(x2 - x1, y2 - y1)
    if total_len == 0:
        return
    dx = (x2 - x1) / total_len
    dy = (y2 - y1) / total_len
    dist = 0.0
    while dist < total_len:
        s = dist
        e = min(dist + dash_len, total_len)
        xs = x1 + dx * s
        ys = y1 + dy * s
        xe = x1 + dx * e
        ye = y1 + dy * e
        draw.line((xs, ys, xe, ye), fill=fill, width=width)
        dist += dash_len + gap_len

def poly_with_width(draw: ImageDraw.ImageDraw, pts, fill=None, outline=(0, 0, 0), width=1):
    if fill is not None:
        draw.polygon(pts, fill=fill)
    if outline is not None and width > 0:
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            draw.line([a, b], fill=outline, width=width)

def draw_square(draw: ImageDraw.ImageDraw, xy, size, outline=(0, 0, 0), width: Optional[int] = None):
    if width is None:
        width = STYLE["square"]
    x, y = xy
    draw.rectangle([x, y, x + size, y + size], outline=outline, width=width)

def draw_circle(draw: ImageDraw.ImageDraw, center, r, fill=None, outline=(0, 0, 0), width: Optional[int] = None):
    if width is None:
        width = STYLE["circle"]
    cx, cy = center
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)

def draw_half_circle(draw: ImageDraw.ImageDraw, center, r, orientation: str = "up", outline=(0, 0, 0), width: Optional[int] = None):
    if width is None:
        width = STYLE["glyph"]
    cx, cy = center
    bbox = [cx - r, cy - r, cx + r, cy + r]
    start_map = {"up": 0, "right": 90, "down": 180, "left": 270}
    end_map = {"up": 180, "right": 270, "down": 360, "left": 90}
    draw.pieslice(bbox, start=start_map[orientation], end=end_map[orientation], fill=None, outline=outline, width=width)

def draw_triangle(draw: ImageDraw.ImageDraw, pts, fill=None, outline=(0, 0, 0), width: Optional[int] = None):
    if width is None:
        width = STYLE["glyph"]
    poly_with_width(draw, pts, fill=fill, outline=outline, width=width)

def draw_plus(draw: ImageDraw.ImageDraw, center, size, width: Optional[int] = None, fill=(0, 0, 0)):
    if width is None:
        width = STYLE["glyph"]
    cx, cy = center
    s = size // 2
    draw.line((cx - s, cy, cx + s, cy), fill=fill, width=width)
    draw.line((cx, cy - s, cx, cy + s), fill=fill, width=width)

def draw_diamond(draw: ImageDraw.ImageDraw, center, size, fill=None, outline=(0, 0, 0), width: Optional[int] = None):
    if width is None:
        width = STYLE["glyph"]
    cx, cy = center
    s = size // 2
    pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
    poly_with_width(draw, pts, fill=fill, outline=outline, width=width)

def draw_star(draw: ImageDraw.ImageDraw, center, r, points=5, outline=(0, 0, 0), width: Optional[int] = None):
    if width is None:
        width = STYLE["glyph"]
    cx, cy = center
    pts = []
    for i in range(points * 2):
        angle = math.pi * i / points
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(angle - math.pi / 2), cy + rr * math.sin(angle - math.pi / 2)))
    poly_with_width(draw, pts, fill=None, outline=outline, width=width)

def draw_L(draw: ImageDraw.ImageDraw, center, size, outline=(0, 0, 0), width: Optional[int] = None):
    if width is None:
        width = STYLE["glyph"]
    cx, cy = center
    s = size
    draw.line((cx - s // 2, cy - s // 2, cx - s // 2, cy + s // 2), fill=outline, width=width)
    draw.line((cx - s // 2, cy + s // 2, cx + s // 2, cy + s // 2), fill=outline, width=width)

def hstack(*imgs: Image.Image, pad: int = 16, bg=(255, 255, 255)) -> Image.Image:
    H = max(im.height for im in imgs)
    W = sum(im.width for im in imgs) + int(pad * CANVAS_SCALE) * (len(imgs) - 1)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for im in imgs:
        y = (H - im.height) // 2
        out.paste(im, (x, y))
        x += im.width + int(pad * CANVAS_SCALE)
    return out

# -------- Banner arrow like sample; Arabic is shaped correctly ----
def draw_banner_arrow(text: str, direction: str = "right") -> Image.Image:
    W, H = int(360 * CANVAS_SCALE), int(120 * CANVAS_SCALE)
    img = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)

    body_h = int(60 * CANVAS_SCALE)
    y0 = (H - body_h) // 2
    if direction == "left":
        d.rectangle([(int(40 * CANVAS_SCALE), y0), (W - int(20 * CANVAS_SCALE), y0 + body_h)], outline=(0, 0, 0), width=STYLE["arrow"])
        tri = [(int(40 * CANVAS_SCALE), H // 2),
               (int(75 * CANVAS_SCALE), y0 - int(6 * CANVAS_SCALE)),
               (int(75 * CANVAS_SCALE), y0 + body_h + int(6 * CANVAS_SCALE))]
        d.polygon(tri, fill=(255, 255, 255), outline=(0, 0, 0))
    else:
        d.rectangle([(int(20 * CANVAS_SCALE), y0), (W - int(40 * CANVAS_SCALE), y0 + body_h)], outline=(0, 0, 0), width=STYLE["arrow"])
        tri = [(W - int(40 * CANVAS_SCALE), H // 2),
               (W - int(75 * CANVAS_SCALE), y0 - int(6 * CANVAS_SCALE)),
               (W - int(75 * CANVAS_SCALE), y0 + body_h + int(6 * CANVAS_SCALE))]
        d.polygon(tri, fill=(255, 255, 255), outline=(0, 0, 0))

    label = shape_ar(text)
    f = get_ar_font(int(28 * CANVAS_SCALE))
    d.text((W // 2, H // 2), label, fill=(0, 0, 0), anchor="mm", font=f)
    return img.convert("RGB")

def faint_hint_box(side: int = 220, text: str = "؟") -> Image.Image:
    img = new_canvas(side, side)
    d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), img.width - 20, outline=(150, 150, 150), width=max(2, STYLE["square"] - 2))
    f = get_ar_font(int(72 * CANVAS_SCALE))
    d.text((img.width // 2, img.height // 2), text, fill=(170, 170, 170), anchor="mm", font=f)
    return img

# ---------------------- Utilities ---------------------------------
def bytes_from_img(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@dataclass
class Question:
    title: str
    stem_image: Image.Image   # Reference (left) + banner arrow + placeholder ?
    options: List[Image.Image]
    correct_index: int
    explanation: str

# ---------------------- Glyph pools (“رموز متغيّرة”) --------------
def glyph_pool(variable_mode: bool, difficulty: str) -> List[str]:
    base = ["plus", "diamond", "circle", "triangle"]
    extra = ["star", "half_up", "half_right", "square_small", "L"]
    if not variable_mode:
        return base
    if difficulty == "سهل":
        return base + ["star", "square_small"]
    if difficulty == "متوسط":
        return base + ["star", "half_up", "square_small", "L"]
    return base + extra

def draw_glyph(draw: ImageDraw.ImageDraw, glyph: str, center: Tuple[int, int], size: int) -> None:
    if glyph == "plus":
        draw_plus(draw, center, size)
    elif glyph == "diamond":
        draw_diamond(draw, center, size - int(4 * CANVAS_SCALE))
    elif glyph == "circle":
        draw_circle(draw, center, (size // 2) - int(2 * CANVAS_SCALE))
    elif glyph == "triangle":
        cx, cy = center
        half = (size // 2)
        pts = [(cx - half, cy + half), (cx + half, cy + half), (cx, cy - half)]
        draw_triangle(draw, pts)
    elif glyph == "star":
        draw_star(draw, center, (size // 2) - int(2 * CANVAS_SCALE))
    elif glyph == "half_up":
        draw_half_circle(draw, center, (size // 2) - int(2 * CANVAS_SCALE), orientation="up")
    elif glyph == "half_right":
        draw_half_circle(draw, center, (size // 2) - int(2 * CANVAS_SCALE), orientation="right")
    elif glyph == "square_small":
        cx, cy = center
        s = size - int(8 * CANVAS_SCALE)
        draw_square(draw, (cx - s // 2, cy - s // 2), s)
    elif glyph == "L":
        draw_L(draw, center, size)

# ---------------------- 2D boards --------------------------------
def board_cross(canvas_size=300, seed=0, variable_mode: bool = True, difficulty: str = "سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas_size, canvas_size)
    d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), img.width - 20)
    midx = img.width // 2
    midy = img.height // 2
    d.line((10, midy, img.width - 10, midy), fill=(0, 0, 0), width=STYLE["grid"])
    d.line((midx, 10, midx, img.height - 10), fill=(0, 0, 0), width=STYLE["grid"])
    pool = glyph_pool(variable_mode, difficulty); RNG.shuffle(pool); chosen = pool[:4]
    jitter = {"سهل": 6, "متوسط": 10, "صعب": 16}[difficulty]; jitter = int(jitter * CANVAS_SCALE)
    offset = int((70 if difficulty != "صعب" else 64) * CANVAS_SCALE)
    locs = [(midx - offset, midy - offset), (midx + offset, midy - offset),
            (midx - offset, midy + offset), (midx + offset, midy + offset)]
    for glyph, (cx, cy) in zip(chosen, locs):
        draw_glyph(d, glyph, (cx + RNG.randint(-jitter, jitter), cy + RNG.randint(-jitter, jitter)), size=int(48 * CANVAS_SCALE))
    return img

def board_diag(canvas_size=300, seed=0, variable_mode: bool = True, difficulty: str = "سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas_size, canvas_size)
    d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), img.width - 20)
    d.line((10, 10, img.width - 10, img.height - 10), fill=(0, 0, 0), width=STYLE["grid"])
    d.line((10, img.height - 10, img.width - 10, 10), fill=(0, 0, 0), width=STYLE["grid"])
    pool = glyph_pool(variable_mode, difficulty); RNG.shuffle(pool); chosen = pool[:3]
    midx = img.width // 2; midy = img.height // 2
    base = int((76 if difficulty != "صعب" else 70) * CANVAS_SCALE)
    jitter = int({"سهل": 6, "متوسط": 10, "صعب": 14}[difficulty] * CANVAS_SCALE)
    slots = [(midx - base, midy - base), (midx + base, midy - base), (midx - base, midy + base)]
    for glyph, (cx, cy) in zip(chosen, slots):
        draw_glyph(d, glyph, (cx + RNG.randint(-jitter, jitter), cy + RNG.randint(-jitter, jitter)), int(48 * CANVAS_SCALE))
    return img

def rotate_image(img: Image.Image, angle_deg: int, allow_mirror=False) -> Image.Image:
    out = img.rotate(angle_deg, expand=True, fillcolor=(255, 255, 255))
    if allow_mirror and RNG.random() < 0.25:
        out = ImageOps.mirror(out)
    return out

def compose_stem(reference: Image.Image, banner_text: str) -> Image.Image:
    """Reference (big, left) + Arabic banner arrow → + faint '?' box."""
    ref = _finalize_for_display(reference)
    arrow = draw_banner_arrow(banner_text, direction="right")
    qbox = faint_hint_box(text="؟")
    return _finalize_for_display(hstack(ref, arrow, qbox))

# ---------------------- Question builders -------------------------
def quadrant_rotation_question(seed: int = 0, difficulty: str = "سهل", variable_mode: bool = True, use_llm: bool = True) -> Question:
    base = board_cross(canvas_size=300, seed=seed, variable_mode=variable_mode, difficulty=difficulty)
    angle = RNG.choice([90, 180, 270])
    stem = compose_stem(base, banner_text=f"عند تدويرها {angle}° يصبح")

    correct = rotate_image(base, -angle, allow_mirror=False)
    if difficulty == "سهل":
        distract = [rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])) for _ in range(2)] + [ImageOps.mirror(base)]
    elif difficulty == "متوسط":
        distract = [ImageOps.flip(rotate_image(base, -angle))] + [rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])) for _ in range(2)]
    else:
        distract = [ImageOps.mirror(rotate_image(base, -angle)), rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])), ImageOps.flip(base)]

    options = [correct] + distract
    RNG.shuffle(options)
    correct_index = options.index(correct)

    stem = _finalize_for_display(stem)
    options = [_finalize_for_display(o) for o in options]

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(sys, "انظر إلى اللوحة على اليسار والسهم. أيُّ بديل يطابق نتيجة التدوير؟", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    expl = ollama_chat_or_fallback(sys, "الصحيح يحافظ على ترتيب الرموز بعد تدويرها حول المركز دون انعكاس.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    return Question(title=title, stem_image=stem, options=options, correct_index=correct_index, explanation=expl)

def diagonal_rotation_question(seed: int = 0, variable_mode: bool = True, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    base = board_diag(canvas_size=300, seed=seed, variable_mode=variable_mode, difficulty=difficulty)
    angle = RNG.choice([90, 180, 270])
    stem = compose_stem(base, banner_text=f"عند تدويرها {angle}° يصبح")

    correct = rotate_image(base, -angle)
    if difficulty == "سهل":
        distract = [rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])) for _ in range(2)] + [ImageOps.mirror(base)]
    elif difficulty == "متوسط":
        distract = [ImageOps.flip(rotate_image(base, -angle))] + [rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])) for _ in range(2)]
    else:
        distract = [ImageOps.mirror(rotate_image(base, -angle)), rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])), ImageOps.flip(base)]

    options = [correct] + distract
    RNG.shuffle(options)
    correct_index = options.index(correct)

    stem = _finalize_for_display(stem)
    options = [_finalize_for_display(o) for o in options]

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(sys, "أيُّ بديل يطابق نتيجة التدوير الموضّحة؟", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    expl = ollama_chat_or_fallback(sys, "راجع مواضع الرموز على القطرين بعد الدوران؛ لا يوجد انعكاس.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    return Question(title=title, stem_image=stem, options=options, correct_index=correct_index, explanation=expl)

# ----------- 3D isometric cube rotation ---------------------------
def orientation_matrices() -> List[np.ndarray]:
    mats = []
    axes = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    ]
    Rz = lambda k: np.array([[math.cos(k), -math.sin(k), 0], [math.sin(k), math.cos(k), 0], [0, 0, 1]])
    for base in axes:
        for k in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            mats.append(Rz(k) @ base)
    uniq, seen = [], set()
    for m in mats:
        key = tuple(np.round(m, 3).flatten().tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq

ORIENTS = orientation_matrices()

def iso_project(pt: Tuple[int, int, int], scale=28) -> Tuple[float, float]:
    x, y, z = pt
    u = (x - y) * scale
    v = (x + y) * scale * 0.5 - z * scale
    return u, v

def draw_iso_cubes(coords: List[Tuple[int, int, int]], size=28, img_size=(300, 260)) -> Image.Image:
    img = new_canvas(*img_size)
    d = ImageDraw.Draw(img)
    us = [iso_project(p, size * CANVAS_SCALE) for p in coords]
    minx = min(u for u, _ in us); maxx = max(u for u, _ in us)
    miny = min(v for _, v in us); maxy = max(v for _, v in us)
    cx = (img.width - (maxx - minx)) / 2 - minx
    cy = (img.height - (maxy - miny)) / 2 - miny
    order = sorted(range(len(coords)), key=lambda i: sum(coords[i]))
    for i in order:
        x, y, z = coords[i]
        u, v = iso_project((x, y, z), size * CANVAS_SCALE); u += cx; v += cy
        S = size * CANVAS_SCALE
        top   = [(u, v - S), (u + S*0.5, v - S*0.5), (u, v), (u - S*0.5, v - S*0.5)]
        right = [(u, v), (u + S*0.5, v - S*0.5), (u + S*0.5, v + S*0.5), (u, v + S)]
        left  = [(u, v), (u - S*0.5, v - S*0.5), (u - S*0.5, v + S*0.5), (u, v + S)]
        poly_with_width(d, top,   fill=(220, 220, 220), outline=(0, 0, 0), width=STYLE["iso_edge"])
        poly_with_width(d, right, fill=(190, 190, 190), outline=(0, 0, 0), width=STYLE["iso_edge"])
        poly_with_width(d, left,  fill=(160, 160, 160), outline=(0, 0, 0), width=STYLE["iso_edge"])
    return img

def random_polycube(n=4, seed=0) -> List[Tuple[int, int, int]]:
    RNG.seed(seed)
    pts = [(0, 0, 0)]
    while len(pts) < n:
        x, y, z = RNG.choice(pts)
        nx, ny, nz = RNG.choice([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
        cand = (x+nx, y+ny, z+nz)
        if cand not in pts:
            pts.append(cand)
        minx = min(p[0] for p in pts); miny = min(p[1] for p in pts); minz = min(p[2] for p in pts)
        pts = [(p[0]-minx, p[1]-miny, p[2]-minz) for p in pts]
    return pts

def apply_rot(coords: List[Tuple[int, int, int]], R: np.ndarray) -> List[Tuple[int, int, int]]:
    arr = np.array(coords).T
    out = np.rint(R @ arr).astype(int).T
    minx = out[:,0].min(); miny = out[:,1].min(); minz = out[:,2].min()
    out = out - np.array([minx, miny, minz])
    return [tuple(map(int, p)) for p in out.tolist()]

def cubes_rotation_question(seed: int = 0, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    RNG.seed(seed)
    n = {"سهل": 4, "متوسط": 5, "صعب": 6}[difficulty]
    shape = random_polycube(n=n, seed=seed)

    ref = draw_iso_cubes(shape, img_size=(300, 260))
    stem = compose_stem(ref, banner_text="عند تدويره يصبح")

    R_true = RNG.choice(ORIENTS)
    correct_coords = apply_rot(shape, R_true)
    correct_img = draw_iso_cubes(correct_coords, img_size=(260, 240))

    used = {tuple(map(int, np.rint(R_true).flatten()))}
    options: List[Image.Image] = [correct_img]
    while len(options) < 4:
        R_alt = RNG.choice(ORIENTS)
        key = tuple(map(int, np.rint(R_alt).flatten()))
        if key in used:
            continue
        used.add(key)
        options.append(draw_iso_cubes(apply_rot(shape, R_alt), img_size=(260, 240)))

    RNG.shuffle(options)
    correct_index = options.index(correct_img)

    stem = _finalize_for_display(stem)
    options = [_finalize_for_display(o) for o in options]

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(sys, "انظر إلى المجسّم على اليسار والسهم. أيُّ بديل يطابقه بعد التدوير؟", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    expl = ollama_chat_or_fallback(sys, "الصحيح يمثّل نفس البوليكويب من زاوية دوران مختلفة فقط.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    return Question(title=title, stem_image=stem, options=options, correct_index=correct_index, explanation=expl)

# ----------- Paper folding ----------------------------------------
def paper_fold_question(seed: int = 0, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    RNG.seed(seed)
    base_size = 280

    folded = new_canvas(base_size, base_size)
    d = ImageDraw.Draw(folded)
    draw_square(d, (10, 10), folded.width - 20)

    fold_dir = RNG.choice(["h", "v"])
    if fold_dir == "h":
        dashed_line(d, (10, folded.height // 2), (folded.width - 10, folded.height // 2))
    else:
        dashed_line(d, (folded.width // 2, 10), (folded.width // 2, folded.height - 10))

    ranges = {"سهل": (1, 2), "متوسط": (2, 3), "صعب": (3, 4)}
    lo, hi = ranges.get(difficulty, (1, 2))
    holes = RNG.randint(lo, hi)
    pts: List[Tuple[int,int]] = []
    for _ in range(holes):
        x = RNG.randint(int(50*CANVAS_SCALE), folded.width - int(50*CANVAS_SCALE))
        y = RNG.randint(int(50*CANVAS_SCALE), folded.height - int(50*CANVAS_SCALE))
        if fold_dir == "h" and y > folded.height // 2:
            y = folded.height - y
        if fold_dir == "v" and x < folded.width // 2:
            x = folded.width - x
        pts.append((x, y))
    for (x, y) in pts:
        draw_circle(d, (x, y), int(12 * CANVAS_SCALE))

    stem = compose_stem(folded, banner_text="افتح وفق خط الطيّ")

    def mirror(p: Tuple[int,int]) -> Tuple[int,int]:
        x, y = p
        if fold_dir == "h":
            return (x, folded.height - y)
        return (folded.width - x, y)

    def render_unfolded(correct=True, jitter=0) -> Image.Image:
        img = new_canvas(base_size, base_size)
        dd = ImageDraw.Draw(img)
        draw_square(dd, (10,10), img.width-20)
        for (x,y) in pts:
            draw_circle(dd, (x,y), int(12*CANVAS_SCALE))
        for (x,y) in pts:
            mx, my = mirror((x,y))
            if not correct and jitter:
                mx += RNG.choice([-jitter, jitter]); my += RNG.choice([-jitter, jitter])
            draw_circle(dd, (mx,my), int(12*CANVAS_SCALE))
        return img

    correct_img = render_unfolded(True)
    wrong1 = render_unfolded(False, jitter=int(12*CANVAS_SCALE))
    wrong2 = ImageOps.mirror(correct_img) if fold_dir == "h" else ImageOps.flip(correct_img)
    wrong3 = new_canvas(base_size, base_size); dd = ImageDraw.Draw(wrong3); draw_square(dd, (10,10), wrong3.width-20)
    for (x,y) in pts: draw_circle(dd, (x,y), int(12*CANVAS_SCALE))

    options = [correct_img, wrong1, wrong2, wrong3]; RNG.shuffle(options)
    correct_index = options.index(correct_img)

    stem = _finalize_for_display(stem); options = [_finalize_for_display(o) for o in options]

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(sys, "بعد فتح الورقة كما في السهم، أيُّ بديل يطابق نمط الثقوب؟", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    expl = ollama_chat_or_fallback(sys, "الصحيح يُظهر انعكاسًا تامًا حول خط الطيّ مع تكرار الثقوب.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    return Question(title=title, stem_image=stem, options=options, correct_index=correct_index, explanation=expl)

# ----------- Shape assembly ---------------------------------------
def shape_assembly_question(seed: int = 0, use_llm: bool = True) -> Question:
    RNG.seed(seed)
    target = RNG.choice(["square", "pentagon"])

    if target == "square":
        ref = new_canvas(280, 280); draw_square(ImageDraw.Draw(ref), (10, 10), ref.width - 20)
    else:
        ref = new_canvas(280, 280); d = ImageDraw.Draw(ref)
        cx, cy, r = ref.width // 2, ref.height // 2, int(110 * CANVAS_SCALE)
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(-math.pi/2, 1.5*math.pi, 6)[:-1]]
        poly_with_width(d, pts, fill=None, outline=(0,0,0), width=STYLE["square"])

    stem = compose_stem(ref, banner_text="كوّن الشكل المطلوب")

    def tri_img(angle=0, flip=False) -> Image.Image:
        S = int(120 * CANVAS_SCALE); pad = int(8 * CANVAS_SCALE)
        im = new_canvas(S + pad * 2, S + pad * 2); dr = ImageDraw.Draw(im)
        pts = [(pad+10, pad+10), (pad+S-10, pad+10), (pad+10, pad+S-10)]
        if flip: pts = [(pad+10, pad+10), (pad+S-10, pad+10), (pad+S-10, pad+S-10)]
        draw_triangle(dr, pts); return im.rotate(angle, expand=True, fillcolor=(255,255,255))

    def panel(imgs: List[Image.Image]) -> Image.Image:
        w = sum(i.width for i in imgs) + int(20*CANVAS_SCALE)*(len(imgs)-1) + int(40*CANVAS_SCALE)
        h = max(i.height for i in imgs) + int(40*CANVAS_SCALE)
        out = new_canvas(w, h); x = int(20*CANVAS_SCALE)
        for im in imgs: out.paste(im, (x, (h-im.height)//2)); x += im.width + int(20*CANVAS_SCALE)
        return out

    if target == "square":
        optA = panel([tri_img(), tri_img(angle=180, flip=True)])  # correct
        rect = new_canvas(100, 140)
        ImageDraw.Draw(rect).rectangle(
            [int(10*CANVAS_SCALE), int(10*CANVAS_SCALE), rect.width-int(10*CANVAS_SCALE), rect.height-int(10*CANVAS_SCALE)],
            outline=(0,0,0), width=STYLE["square"])
        rhombus = new_canvas(120,120); draw_diamond(ImageDraw.Draw(rhombus), (rhombus.width//2, rhombus.height//2), int(80*CANVAS_SCALE))
        optB = panel([rect, rhombus]); optC = panel([tri_img(angle=90), rhombus.rotate(45, expand=True, fillcolor=(255,255,255))]); optD = panel([tri_img(flip=True), rect])
        options = [optA, optB, optC, optD]; correct_index = 0
    else:
        quad = new_canvas(160,140); ImageDraw.Draw(quad).polygon([(int(20*CANVAS_SCALE),int(120*CANVAS_SCALE)),(int(80*CANVAS_SCALE),int(20*CANVAS_SCALE)),(int(140*CANVAS_SCALE),int(60*CANVAS_SCALE)),(int(100*CANVAS_SCALE),int(120*CANVAS_SCALE))], outline=(0,0,0), fill=None)
        quad2 = new_canvas(160,140); ImageDraw.Draw(quad2).polygon([(int(20*CANVAS_SCALE),int(60*CANVAS_SCALE)),(int(60*CANVAS_SCALE),int(20*CANVAS_SCALE)),(int(140*CANVAS_SCALE),int(80*CANVAS_SCALE)),(int(100*CANVAS_SCALE),int(120*CANVAS_SCALE))], outline=(0,0,0), fill=None)
        circ = new_canvas(160,140); ImageDraw.Draw(circ).ellipse([int(30*CANVAS_SCALE),int(30*CANVAS_SCALE),int(130*CANVAS_SCALE),int(110*CANVAS_SCALE)], outline=(0,0,0), width=STYLE["square"])
        tri = new_canvas(160,140); draw_triangle(ImageDraw.Draw(tri), [(int(20*CANVAS_SCALE),int(120*CANVAS_SCALE)),(int(80*CANVAS_SCALE),int(20*CANVAS_SCALE)),(int(140*CANVAS_SCALE),int(120*CANVAS_SCALE))])
        options = [panel([quad, quad2]), panel([quad, circ]), panel([quad2, tri]), panel([circ, tri])]; correct_index = 0

    stem = _finalize_for_display(stem); options = [_finalize_for_display(o) for o in options]

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(sys, "أي مجموعة من القطع يمكن أن تُكوِّن الشكل المطلوب؟", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=st.session_state.get("use_llm", True))
    expl = ollama_chat_or_fallback(sys, "قارن الحواف والزوايا والمساحات؛ المجموعة الصحيحة تُكوِّن الحدود دون فجوات.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=st.session_state.get("use_llm", True))
    return Question(title=title, stem_image=stem, options=options, correct_index=correct_index, explanation=expl)

# ---------------------- Streamlit Page UI -------------------------
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH))
st.sidebar.markdown("### الإعدادات")

left, right = st.sidebar.columns([1, 1])
with left:
    n_q = st.number_input("عدد الأسئلة", 1, 24, 8)
with right:
    difficulty = st.selectbox("مستوى الصعوبة", ["سهل", "متوسط", "صعب"])

seed_base = st.sidebar.number_input("البذرة (Seed)", 0, 10_000_000, 12345, help="لإعادة إنتاج نفس الأسئلة.")
if st.sidebar.button("🎲 بذرة عشوائية"):
    seed_base = RNG.randint(0, 10_000_000)

st.session_state["llm_model"] = st.sidebar.text_input("نموذج Ollama (اختياري)", value="qwen2.5:3b")
st.session_state["use_llm"] = st.sidebar.checkbox("استخدم LLM لكتابة التعليمات/الشرح", value=True)
variable_symbols = st.sidebar.checkbox("وضع **رموز متغيّرة** (تنويع الرموز)", value=True)

puzzle_types = st.sidebar.multiselect(
    "أنواع الأسئلة",
    ["طيّ الورق", "تدوير رباعي", "تدوير قطري", "تدوير مكعّبات ثلاثي", "تركيب شكل"],
    default=["طيّ الورق", "تدوير رباعي", "تدوير قطري", "تدوير مكعّبات ثلاثي", "تركيب شكل"],
)

col1, col2 = st.columns([1, 1])
with col1:
    gen = st.button("🚀 إنشاء الأسئلة", use_container_width=True)
with col2:
    st.caption("اللوحة العلوية تعرض **المرجع** وسهمًا عربيًا كبيرًا «عند تدويره يصبح». ثم اختر من البدائل (أ/ب/ج/د).")

st.title("مولّد أسئلة ذكاء مرئية (Spatial IQ)")

def build_by_type(kind: str, seed: int) -> Question:
    if kind == "طيّ الورق":
        return paper_fold_question(seed=seed, difficulty=difficulty, use_llm=st.session_state.get("use_llm", True))
    if kind == "تدوير رباعي":
        return quadrant_rotation_question(seed=seed, difficulty=difficulty, variable_mode=variable_symbols, use_llm=st.session_state.get("use_llm", True))
    if kind == "تدوير قطري":
        return diagonal_rotation_question(seed=seed, difficulty=difficulty, variable_mode=variable_symbols, use_llm=st.session_state.get("use_llm", True))
    if kind == "تدوير مكعّبات ثلاثي":
        return cubes_rotation_question(seed=seed, difficulty=difficulty, use_llm=st.session_state.get("use_llm", True))
    if kind == "تركيب شكل":
        return shape_assembly_question(seed=seed, use_llm=st.session_state.get("use_llm", True))
    raise ValueError("نوع غير معروف")

if gen:
    RNG.seed(seed_base)
    order: List[str] = []
    while len(order) < n_q:
        order.extend(puzzle_types)
    RNG.shuffle(order)
    order = order[:n_q]

    answers_csv = io.StringIO()
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, kind in enumerate(order, 1):
            qseed = seed_base ^ (RNG.randint(1, 1_000_000_007) + idx * 9973)
            q = build_by_type(kind, seed=qseed)

            st.markdown(f"#### سؤال {idx}: {q.title}")
            st.image(q.stem_image, use_container_width=True)
            cols = st.columns(4, gap="small")
            for i, (c, col) in enumerate(zip(q.options, cols)):
                col.image(c, use_container_width=True)
                col.markdown(
                    f"<div style='text-align:center;font-weight:bold;font-size:20px'>{AR_LETTERS[i]}</div>",
                    unsafe_allow_html=True,
                )

            # ZIP export
            zf.writestr(f"q{idx}_stem.png", bytes_from_img(q.stem_image))
            for i, c in enumerate(q.options, start=1):
                zf.writestr(f"q{idx}_opt_{i}.png", bytes_from_img(c))
            answers_csv.write(f"{idx},{kind},{AR_LETTERS[q.correct_index]},{qseed}\n")

            with st.expander("إظهار الحل/الشرح"):
                st.markdown(f"**الإجابة الصحيحة:** {AR_LETTERS[q.correct_index]}")
                st.write(q.explanation if st.session_state.get("use_llm", True) else "الصحيح الوحيد يوافق قاعدة السؤال (انعكاس/تدوير/تجميع).")
            st.divider()

        zf.writestr("answers.csv", answers_csv.getvalue().encode("utf-8"))

    st.download_button(
        "⬇️ تنزيل كل الأسئلة (ZIP)",
        data=zip_buf.getvalue(),
        file_name="arabic_visual_iq_questions.zip",
        mime="application/zip",
        use_container_width=True,
    )
else:
    st.info("اضغط **إنشاء الأسئلة** لبدء التوليد.")
