# Write a fully fixed Streamlit page (no stray file-writes; no bad literals)
fixed = r'''# -*- coding: utf-8 -*-
"""
Streamlit page 7: Arabic visual IQ question generator
- Paper folding & hole punching
- 2D rotation (cross + diagonals) with variable symbols
- 3D isometric cube rotation
- Shape assembly (which set forms the target?)
- Difficulty tiers: سهل / متوسط / صعب
- Crisp strokes via hi-res render then downscale
File path suggestion: pages/7_spacial_questions.py
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

# ------------------------------ Basics ------------------------------
AR_LETTERS = ["أ", "ب", "ج", "د"]
RNG = random.Random(1234)

STYLE = {"grid": 6, "square": 6, "glyph": 6, "circle": 5, "arrow": 6, "iso_edge": 3, "dash": 4}
CANVAS_SCALE = 1.7  # render big then downscale for crisp lines

def new_canvas(w: int, h: int, bg=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (int(w * CANVAS_SCALE), int(h * CANVAS_SCALE)), bg)

def finalize(img: Image.Image, target: Optional[Tuple[int, int]] = None) -> Image.Image:
    if target:
        return img.resize((int(target[0]), int(target[1])), Image.LANCZOS)
    return img.resize((int(img.width / CANVAS_SCALE), int(img.height / CANVAS_SCALE)), Image.LANCZOS)

def get_ar_font(px: int) -> Optional[ImageFont.FreeTypeFont]:
    for name in ["NotoNaskhArabic-Regular.ttf", "NotoKufiArabic-Regular.ttf", "Amiri-Regular.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, int(px))
        except Exception:
            continue
    return None

def shape_ar(text: str) -> str:
    try:
        import arabic_reshaper  # type: ignore
        from bidi.algorithm import get_display  # type: ignore
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text

def draw_square(d: ImageDraw.ImageDraw, xy, size, outline=(0, 0, 0), width=None):
    width = width or STYLE["square"]
    x, y = xy
    d.rectangle([x, y, x + size, y + size], outline=outline, width=width)

def draw_circle(d: ImageDraw.ImageDraw, center, r, fill=None, outline=(0, 0, 0), width=None):
    width = width or STYLE["circle"]
    cx, cy = center
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)

def poly_with_width(d: ImageDraw.ImageDraw, pts, fill=None, outline=(0, 0, 0), width=1):
    if fill is not None:
        d.polygon(pts, fill=fill)
    if outline is not None and width > 0:
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            d.line([a, b], fill=outline, width=width)

def draw_triangle(d: ImageDraw.ImageDraw, pts, fill=None, outline=(0, 0, 0), width=None):
    width = width or STYLE["glyph"]
    poly_with_width(d, pts, fill=fill, outline=outline, width=width)

def draw_plus(d: ImageDraw.ImageDraw, center, size, width=None, fill=(0, 0, 0)):
    width = width or STYLE["glyph"]
    cx, cy = center; s = size // 2
    d.line((cx - s, cy, cx + s, cy), fill=fill, width=width)
    d.line((cx, cy - s, cx, cy + s), fill=fill, width=width)

def draw_diamond(d: ImageDraw.ImageDraw, center, size, fill=None, outline=(0, 0, 0), width=None):
    width = width or STYLE["glyph"]
    cx, cy = center; s = size // 2
    pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
    poly_with_width(d, pts, fill=fill, outline=outline, width=width)

def draw_half_circle(d: ImageDraw.ImageDraw, center, r, orientation="up", outline=(0, 0, 0), width=None):
    width = width or STYLE["glyph"]
    cx, cy = center; bbox = [cx - r, cy - r, cx + r, cy + r]
    start = {"up": 0, "right": 90, "down": 180, "left": 270}[orientation]
    end = {"up": 180, "right": 270, "down": 360, "left": 90}[orientation]
    d.pieslice(bbox, start=start, end=end, outline=outline, width=width)

def dashed_line(d: ImageDraw.ImageDraw, start, end, dash=10, gap=8, fill=(0, 0, 0), width=None):
    width = width or STYLE["dash"]
    x1, y1 = start; x2, y2 = end
    L = math.hypot(x2 - x1, y2 - y1)
    if L == 0: return
    dx, dy = (x2 - x1) / L, (y2 - y1) / L
    t = 0.0
    while t < L:
        s, e = t, min(t + dash, L)
        xs, ys = x1 + dx * s, y1 + dy * s
        xe, ye = x1 + dx * e, y1 + dy * e
        d.line((xs, ys, xe, ye), fill=fill, width=width)
        t += dash + gap

def banner_arrow(text: str) -> Image.Image:
    W, H = int(360 * CANVAS_SCALE), int(120 * CANVAS_SCALE)
    img = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    body_h = int(60 * CANVAS_SCALE); y0 = (H - body_h) // 2
    d.rectangle([(int(20 * CANVAS_SCALE), y0), (W - int(40 * CANVAS_SCALE), y0 + body_h)], outline=(0, 0, 0), width=STYLE["arrow"])
    tri = [(W - int(40 * CANVAS_SCALE), H // 2),
           (W - int(75 * CANVAS_SCALE), y0 - int(6 * CANVAS_SCALE)),
           (W - int(75 * CANVAS_SCALE), y0 + body_h + int(6 * CANVAS_SCALE))]
    d.polygon(tri, fill=(255, 255, 255), outline=(0, 0, 0))
    f = get_ar_font(int(28 * CANVAS_SCALE))
    d.text((W // 2, H // 2), shape_ar(text), fill=(0, 0, 0), anchor="mm", font=f)
    return img.convert("RGB")

def faint_qbox(side=220, mark="؟") -> Image.Image:
    img = new_canvas(side, side); d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), img.width - 20, outline=(150, 150, 150), width=max(2, STYLE["square"] - 2))
    f = get_ar_font(int(72 * CANVAS_SCALE))
    d.text((img.width // 2, img.height // 2), mark, fill=(170, 170, 170), anchor="mm", font=f)
    return img

def hstack(*imgs: Image.Image, pad=16) -> Image.Image:
    H = max(i.height for i in imgs)
    W = sum(i.width for i in imgs) + int(pad * CANVAS_SCALE) * (len(imgs) - 1)
    out = Image.new("RGB", (W, H), (255, 255, 255))
    x = 0
    for im in imgs:
        out.paste(im, (x, (H - im.height) // 2))
        x += im.width + int(pad * CANVAS_SCALE)
    return out

# ---------------------- Glyphs & boards ----------------------------
def glyph_pool(variable: bool, difficulty: str) -> List[str]:
    base = ["plus", "diamond", "circle", "triangle"]
    extra = ["star", "half_up", "half_right", "square_small", "L"]
    if not variable: return base
    if difficulty == "سهل": return base + ["star", "square_small"]
    if difficulty == "متوسط": return base + ["star", "half_up", "square_small", "L"]
    return base + extra

def draw_glyph(d: ImageDraw.ImageDraw, name: str, center: Tuple[int, int], size: int) -> None:
    if name == "plus":
        draw_plus(d, center, size)
    elif name == "diamond":
        draw_diamond(d, center, size - int(4 * CANVAS_SCALE))
    elif name == "circle":
        draw_circle(d, center, (size // 2) - int(2 * CANVAS_SCALE))
    elif name == "triangle":
        cx, cy = center; half = size // 2
        pts = [(cx - half, cy + half), (cx + half, cy + half), (cx, cy - half)]
        draw_triangle(d, pts)
    elif name == "star":
        # simple 5-point star
        cx, cy = center; r = (size // 2) - int(2 * CANVAS_SCALE); pts = []
        for i in range(10):
            ang = math.pi * i / 5.0 - math.pi / 2
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        poly_with_width(d, pts, fill=None, outline=(0, 0, 0), width=STYLE["glyph"])
    elif name == "half_up":
        draw_half_circle(d, center, (size // 2) - int(2 * CANVAS_SCALE), orientation="up")
    elif name == "half_right":
        draw_half_circle(d, center, (size // 2) - int(2 * CANVAS_SCALE), orientation="right")
    elif name == "square_small":
        cx, cy = center; s = size - int(8 * CANVAS_SCALE)
        draw_square(d, (cx - s // 2, cy - s // 2), s)
    elif name == "L":
        cx, cy = center; s = size; w = STYLE["glyph"]
        d.line((cx - s // 2, cy - s // 2, cx - s // 2, cy + s // 2), fill=(0, 0, 0), width=w)
        d.line((cx - s // 2, cy + s // 2, cx + s // 2, cy + s // 2), fill=(0, 0, 0), width=w)

def board_cross(canvas=300, seed=0, variable=True, difficulty="سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas, canvas); d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), img.width - 20)
    midx, midy = img.width // 2, img.height // 2
    d.line((10, midy, img.width - 10, midy), fill=(0, 0, 0), width=STYLE["grid"])
    d.line((midx, 10, midx, img.height - 10), fill=(0, 0, 0), width=STYLE["grid"])
    pool = glyph_pool(variable, difficulty); RNG.shuffle(pool); chosen = pool[:4]
    jitter = int({"سهل": 6, "متوسط": 10, "صعب": 16}[difficulty] * CANVAS_SCALE)
    offset = int((70 if difficulty != "صعب" else 64) * CANVAS_SCALE)
    locs = [(midx - offset, midy - offset), (midx + offset, midy - offset), (midx - offset, midy + offset), (midx + offset, midy + offset)]
    for g, (cx, cy) in zip(chosen, locs):
        draw_glyph(d, g, (cx + RNG.randint(-jitter, jitter), cy + RNG.randint(-jitter, jitter)), int(48 * CANVAS_SCALE))
    return img

def board_diag(canvas=300, seed=0, variable=True, difficulty="سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas, canvas); d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), img.width - 20)
    d.line((10, 10, img.width - 10, img.height - 10), fill=(0, 0, 0), width=STYLE["grid"])
    d.line((10, img.height - 10, img.width - 10, 10), fill=(0, 0, 0), width=STYLE["grid"])
    pool = glyph_pool(variable, difficulty); RNG.shuffle(pool); chosen = pool[:3]
    midx, midy = img.width // 2, img.height // 2
    base = int((76 if difficulty != "صعب" else 70) * CANVAS_SCALE)
    jitter = int({"سهل": 6, "متوسط": 10, "صعب": 14}[difficulty] * CANVAS_SCALE)
    locs = [(midx - base, midy - base), (midx + base, midy - base), (midx - base, midy + base)]
    for g, (cx, cy) in zip(chosen, locs):
        draw_glyph(d, g, (cx + RNG.randint(-jitter, jitter), cy + RNG.randint(-jitter, jitter)), int(48 * CANVAS_SCALE))
    return img

def rotate_image(img: Image.Image, angle_deg: int, allow_mirror=False) -> Image.Image:
    out = img.rotate(angle_deg, expand=True, fillcolor=(255, 255, 255))
    if allow_mirror and RNG.random() < 0.25:
        out = ImageOps.mirror(out)
    return out

def compose_stem(reference: Image.Image, banner_text: str) -> Image.Image:
    return finalize(hstack(finalize(reference), banner_arrow(banner_text), faint_qbox()))

# ---------------------- 3D isometric cubes ------------------------
def orientation_matrices() -> List[np.ndarray]:
    mats = []
    bases = [
        np.eye(3),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    ]
    def Rz(k):
        return np.array([[math.cos(k), -math.sin(k), 0], [math.sin(k), math.cos(k), 0], [0, 0, 1]])
    for base in bases:
        for k in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            mats.append(Rz(k) @ base)
    uniq, seen = [], set()
    for m in mats:
        key = tuple(np.round(m, 3).flatten())
        if key not in seen:
            seen.add(key); uniq.append(m)
    return uniq

ORIENTS = orientation_matrices()

def iso_project(pt: Tuple[int, int, int], scale=28) -> Tuple[float, float]:
    x, y, z = pt
    u = (x - y) * scale
    v = (x + y) * scale * 0.5 - z * scale
    return u, v

def draw_iso_cubes(coords: List[Tuple[int, int, int]], size=28, img_size=(300, 260)) -> Image.Image:
    img = new_canvas(*img_size); d = ImageDraw.Draw(img)
    us = [iso_project(p, size * CANVAS_SCALE) for p in coords]
    minx, maxx = min(u for u, _ in us), max(u for u, _ in us)
    miny, maxy = min(v for _, v in us), max(v for _, v in us)
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
        dx, dy, dz = RNG.choice([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
        cand = (x+dx, y+dy, z+dz)
        if cand not in pts:
            pts.append(cand)
        minx = min(p[0] for p in pts); miny = min(p[1] for p in pts); minz = min(p[2] for p in pts)
        pts = [(p[0]-minx, p[1]-miny, p[2]-minz) for p in pts]
    return pts

def apply_rot(coords: List[Tuple[int, int, int]], R: np.ndarray) -> List[Tuple[int, int, int]]:
    arr = np.array(coords).T
    out = np.rint(R @ arr).astype(int).T
    minx, miny, minz = out[:,0].min(), out[:,1].min(), out[:,2].min()
    out = out - np.array([minx, miny, minz])
    return [tuple(map(int, p)) for p in out.tolist()]

# ---------------------- Question defs -----------------------------
@dataclass
class Question:
    title: str
    stem_image: Image.Image
    options: List[Image.Image]
    correct_index: int
    explanation: str

def quadrant_rotation_question(seed=0, difficulty="سهل", variable=True) -> Question:
    base = board_cross(canvas=300, seed=seed, variable=variable, difficulty=difficulty)
    angle = RNG.choice([90, 180, 270])
    stem = compose_stem(base, f"عند تدويرها {angle}° يصبح")
    correct = rotate_image(base, -angle, allow_mirror=False)
    distract = [
        rotate_image(base, -a) for a in [a for a in [90, 180, 270] if a != angle]
    ] + [ImageOps.mirror(base)]
    options = [correct] + distract; RNG.shuffle(options)
    idx = options.index(correct)
    return Question("اختر الشكل المطابق بعد التدوير", finalize(stem), [finalize(o) for o in options], idx, "التدوير حول المركز دون انعكاس.")

def diagonal_rotation_question(seed=0, difficulty="سهل", variable=True) -> Question:
    base = board_diag(canvas=300, seed=seed, variable=variable, difficulty=difficulty)
    angle = RNG.choice([90, 180, 270])
    stem = compose_stem(base, f"عند تدويرها {angle}° يصبح")
    correct = rotate_image(base, -angle)
    distract = [
        rotate_image(base, -a) for a in [a for a in [90, 180, 270] if a != angle]
    ] + [ImageOps.mirror(base)]
    options = [correct] + distract; RNG.shuffle(options)
    idx = options.index(correct)
    return Question("أي بديل يطابق النتيجة؟", finalize(stem), [finalize(o) for o in options], idx, "قارن مواضع الرموز بعد الدوران.")

def cubes_rotation_question(seed=0, difficulty="سهل") -> Question:
    n = {"سهل": 4, "متوسط": 5, "صعب": 6}[difficulty]
    shape = random_polycube(n=n, seed=seed)
    ref = draw_iso_cubes(shape, img_size=(300, 260))
    stem = compose_stem(ref, "عند تدويره يصبح")
    R_true = RNG.choice(ORIENTS)
    correct = draw_iso_cubes(apply_rot(shape, R_true), img_size=(260, 240))
    opts = [correct]
    used = {tuple(np.rint(R_true).flatten())}
    while len(opts) < 4:
        R_alt = RNG.choice(ORIENTS)
        k = tuple(np.rint(R_alt).flatten())
        if k in used: continue
        used.add(k)
        opts.append(draw_iso_cubes(apply_rot(shape, R_alt), img_size=(260, 240)))
    RNG.shuffle(opts); idx = opts.index(correct)
    return Question("أختر الزاوية المطابقة للمجسّم", finalize(stem), [finalize(o) for o in opts], idx, "المجرّد نفسه بزوايا مختلفة.")

def paper_fold_question(seed=0, difficulty="سهل") -> Question:
    RNG.seed(seed)
    S = 280
    fold_dir = RNG.choice(["h", "v"])
    folded = new_canvas(S, S); d = ImageDraw.Draw(folded)
    draw_square(d, (10, 10), folded.width - 20)
    if fold_dir == "h":
        dashed_line(d, (10, folded.height // 2), (folded.width - 10, folded.height // 2))
    else:
        dashed_line(d, (folded.width // 2, 10), (folded.width // 2, folded.height - 10))
    lo, hi = {"سهل": (1, 2), "متوسط": (2, 3), "صعب": (3, 4)}[difficulty]
    holes = RNG.randint(lo, hi)
    pts: List[Tuple[int, int]] = []
    for _ in range(holes):
        x = RNG.randint(int(50 * CANVAS_SCALE), folded.width - int(50 * CANVAS_SCALE))
        y = RNG.randint(int(50 * CANVAS_SCALE), folded.height - int(50 * CANVAS_SCALE))
        if fold_dir == "h" and y > folded.height // 2: y = folded.height - y
        if fold_dir == "v" and x < folded.width // 2: x = folded.width - x
        pts.append((x, y))
    for x, y in pts:
        draw_circle(d, (x, y), int(12 * CANVAS_SCALE))

    stem = compose_stem(folded, "افتح وفق خط الطيّ")

    def mirror(p: Tuple[int, int]) -> Tuple[int, int]:
        x, y = p
        return (x, folded.height - y) if fold_dir == "h" else (folded.width - x, y)

    def render_unfolded(correct=True, jitter=0) -> Image.Image:
        img = new_canvas(S, S); dd = ImageDraw.Draw(img)
        draw_square(dd, (10, 10), img.width - 20)
        for (x, y) in pts: draw_circle(dd, (x, y), int(12 * CANVAS_SCALE))
        for (x, y) in pts:
            mx, my = mirror((x, y))
            if not correct and jitter:
                mx += RNG.choice([-jitter, jitter]); my += RNG.choice([-jitter, jitter])
            draw_circle(dd, (mx, my), int(12 * CANVAS_SCALE))
        return img

    correct = render_unfolded(True)
    wrong1 = render_unfolded(False, jitter=int(12 * CANVAS_SCALE))
    wrong2 = ImageOps.mirror(correct) if fold_dir == "h" else ImageOps.flip(correct)
    wrong3 = new_canvas(S, S); dd = ImageDraw.Draw(wrong3); draw_square(dd, (10, 10), wrong3.width - 20)
    for (x, y) in pts: draw_circle(dd, (x, y), int(12 * CANVAS_SCALE))

    options = [correct, wrong1, wrong2, wrong3]; RNG.shuffle(options); idx = options.index(correct)
    return Question("بعد فتح الورقة، أيّ بديل صحيح؟", finalize(stem), [finalize(o) for o in options], idx, "الأنماط تنعكس حول خط الطيّ وتُضاعَف.")

def shape_assembly_question(seed=0) -> Question:
    RNG.seed(seed)
    target = RNG.choice(["square", "pentagon"])

    if target == "square":
        ref = new_canvas(280, 280); draw_square(ImageDraw.Draw(ref), (10, 10), ref.width - 20)
    else:
        ref = new_canvas(280, 280); d = ImageDraw.Draw(ref)
        cx, cy, r = ref.width // 2, ref.height // 2, int(110 * CANVAS_SCALE)
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(-math.pi/2, 1.5*math.pi, 6)[:-1]]
        poly_with_width(d, pts, fill=None, outline=(0,0,0), width=STYLE["square"])

    stem = compose_stem(ref, "كوّن الشكل المطلوب")

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
        ImageDraw.Draw(rect).rectangle([int(10*CANVAS_SCALE), int(10*CANVAS_SCALE), rect.width-int(10*CANVAS_SCALE), rect.height-int(10*CANVAS_SCALE)], outline=(0,0,0), width=STYLE["square"])
        rh = new_canvas(120,120); draw_diamond(ImageDraw.Draw(rh), (rh.width//2, rh.height//2), int(80*CANVAS_SCALE))
        optB = panel([rect, rh]); optC = panel([tri_img(angle=90), rh.rotate(45, expand=True, fillcolor=(255,255,255))]); optD = panel([tri_img(flip=True), rect])
        options = [optA, optB, optC, optD]; idx = 0
    else:
        q1 = new_canvas(160,140); ImageDraw.Draw(q1).polygon([(int(20*CANVAS_SCALE),int(120*CANVAS_SCALE)),(int(80*CANVAS_SCALE),int(20*CANVAS_SCALE)),(int(140*CANVAS_SCALE),int(60*CANVAS_SCALE)),(int(100*CANVAS_SCALE),int(120*CANVAS_SCALE))], outline=(0,0,0), fill=None)
        q2 = new_canvas(160,140); ImageDraw.Draw(q2).polygon([(int(20*CANVAS_SCALE),int(60*CANVAS_SCALE)),(int(60*CANVAS_SCALE),int(20*CANVAS_SCALE)),(int(140*CANVAS_SCALE),int(80*CANVAS_SCALE)),(int(100*CANVAS_SCALE),int(120*CANVAS_SCALE))], outline=(0,0,0), fill=None)
        circ = new_canvas(160,140); ImageDraw.Draw(circ).ellipse([int(30*CANVAS_SCALE),int(30*CANVAS_SCALE),int(130*CANVAS_SCALE),int(110*CANVAS_SCALE)], outline=(0,0,0), width=STYLE["square"])
        tri = new_canvas(160,140); draw_triangle(ImageDraw.Draw(tri), [(int(20*CANVAS_SCALE),int(120*CANVAS_SCALE)),(int(80*CANVAS_SCALE),int(20*CANVAS_SCALE)),(int(140*CANVAS_SCALE),int(120*CANVAS_SCALE))])
        options = [panel([q1, q2]), panel([q1, circ]), panel([q2, tri]), panel([circ, tri])]; idx = 0

    return Question("أي مجموعة تكوّن الشكل؟", finalize(stem), [finalize(o) for o in options], idx, "قارن الحواف والزوايا والمساحات.")

# ---------------------- UI ----------------------------------------
st.title("مولّد أسئلة الذكاء البصري (Spatial IQ)")

st.sidebar.markdown("### الإعدادات")
n_q = st.sidebar.number_input("عدد الأسئلة", 1, 24, 8)
difficulty = st.sidebar.selectbox("مستوى الصعوبة", ["سهل", "متوسط", "صعب"])
seed_base = st.sidebar.number_input("البذرة (Seed)", 0, 10_000_000, 12345)
if st.sidebar.button("🎲 بذرة عشوائية"):
    seed_base = RNG.randint(0, 10_000_000)
variable_symbols = st.sidebar.checkbox("رموز متغيّرة", value=True)

puzzle_types = st.sidebar.multiselect(
    "أنواع الأسئلة",
    ["طيّ الورق", "تدوير رباعي", "تدوير قطري", "تدوير مكعّبات ثلاثي", "تركيب شكل"],
    default=["طيّ الورق", "تدوير رباعي", "تدوير قطري", "تدوير مكعّبات ثلاثي", "تركيب شكل"],
)

col1, col2 = st.columns([1, 1])
with col1:
    gen = st.button("🚀 إنشاء الأسئلة", use_container_width=True)
with col2:
    st.caption("سترى المرجع ← سهم عربي «عند تدويرها/فتحها يصبح…» ثم البدائل (أ/ب/ج/د).")

def build(kind: str, seed: int) -> Question:
    if kind == "طيّ الورق": return paper_fold_question(seed, difficulty)
    if kind == "تدوير رباعي": return quadrant_rotation_question(seed, difficulty, variable_symbols)
    if kind == "تدوير قطري": return diagonal_rotation_question(seed, difficulty, variable_symbols)
    if kind == "تدوير مكعّبات ثلاثي": return cubes_rotation_question(seed, difficulty)
    if kind == "تركيب شكل": return shape_assembly_question(seed)
    raise ValueError("نوع غير معروف")

if gen:
    RNG.seed(seed_base)
    order: List[str] = []
    while len(order) < n_q: order.extend(puzzle_types)
    RNG.shuffle(order); order = order[:n_q]

    answers_csv = io.StringIO()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, kind in enumerate(order, 1):
            qseed = seed_base ^ (RNG.randint(1, 1000000007) + idx * 9973)
            q = build(kind, qseed)

            st.markdown(f"#### سؤال {idx}: {q.title}")
            st.image(q.stem_image, use_container_width=True)
            cols = st.columns(4, gap="small")
            for i, (c, col) in enumerate(zip(q.options, cols)):
                col.image(c, use_container_width=True)
                col.markdown(f"<div style='text-align:center;font-weight:bold;font-size:20px'>{AR_LETTERS[i]}</div>", unsafe_allow_html=True)

            # ZIP export
            def img_bytes(im: Image.Image) -> bytes:
                b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()
            zf.writestr(f"q{idx}_stem.png", img_bytes(q.stem_image))
            for i, c in enumerate(q.options, start=1):
                zf.writestr(f"q{idx}_opt_{i}.png", img_bytes(c))
            answers_csv.write(f"{idx},{kind},{AR_LETTERS[q.correct_index]},{qseed}\n")

            with st.expander("إظهار الحل/الشرح"):
                st.markdown(f"**الإجابة الصحيحة:** {AR_LETTERS[q.correct_index]}")
                st.write(q.explanation if q.explanation else "الصحيح الوحيد يطابق القاعدة (انعكاس/تدوير/تجميع).")
            st.divider()

        zf.writestr("answers.csv", answers_csv.getvalue().encode("utf-8"))

    st.download_button("⬇️ تنزيل كل الأسئلة (ZIP)", data=zip_buf.getvalue(), file_name="arabic_visual_iq_questions.zip", mime="application/zip", use_container_width=True)
else:
    st.info("اضغط **إنشاء الأسئلة** لبدء التوليد.")
'''
with open('/mnt/data/7_spacial_questions_fixed.py', 'w', encoding='utf-8') as f:
    f.write(fixed)
