# -*- coding: utf-8 -*-
"""
Streamlit page 7: Arabic visual IQ question generator (no API/JSON)
- Paper folding & hole punching
- 2D rotation (Cross-board & Diagonal/X-board) + “رموز متغيّرة”
- 3D isometric cube rotation
- Shape assembly (which set forms the target?)
- Difficulty tiers: سهل / متوسط / صعب

Key UX fix: every question now shows a clear **reference stem** image at the top:
- Left: المصدر/المجسّم أو اللوحة الأصلية
- Center: سهم تدوير مع زاوية (أو سهم طيّ)
- Right: صندوق تلميح شفاف (لا يكشف الحل) — فقط لبيان المطلوب
Options تُعرض أسفلها دائمًا.

File path: pages/7_spacial_questions.py
"""

import io
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import streamlit as st

# ------------------------------ Paths ------------------------------
ROOT = Path(__file__).parents[1]
LOGO_PATH = ROOT / "MOL_logo.png"

# ------------------------------ Config ------------------------------
AR_LETTERS = ["أ", "ب", "ج", "د"]
RNG = random.Random()

# ------------------ LLM (optional via Ollama) ----------------------
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def ollama_chat_or_fallback(system: str, user: str, model: str, enabled: bool = True, max_tokens: int = 256) -> str:
    """Use local Ollama if available/enabled; otherwise return the provided user text."""
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


# ---------------------- Helpers: drawing ---------------------------
def new_canvas(w: int, h: int, bg=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (w, h), bg)


def dashed_line(draw: ImageDraw.ImageDraw, start, end, dash_len=6, gap_len=6, fill=(0, 0, 0), width=1):
    x1, y1 = start
    x2, y2 = end
    total_len = math.hypot(x2 - x1, y2 - y1)
    if total_len == 0:
        return
    dx = (x2 - x1) / total_len
    dy = (y2 - y1) / total_len
    dist = 0
    while dist < total_len:
        s = dist
        e = min(dist + dash_len, total_len)
        xs = x1 + dx * s
        ys = y1 + dy * s
        xe = x1 + dx * e
        ye = y1 + dy * e
        draw.line((xs, ys, xe, ye), fill=fill, width=width)
        dist += dash_len + gap_len


def draw_square(draw: ImageDraw.ImageDraw, xy, size, outline=(0, 0, 0), width=3):
    x, y = xy
    draw.rectangle([x, y, x + size, y + size], outline=outline, width=width)


def draw_circle(draw: ImageDraw.ImageDraw, center, r, fill=None, outline=(0, 0, 0), width=3):
    cx, cy = center
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)


def draw_half_circle(draw: ImageDraw.ImageDraw, center, r, orientation: str = "up", outline=(0, 0, 0), width=3):
    cx, cy = center
    bbox = [cx - r, cy - r, cx + r, cy + r]
    start_map = {"up": 0, "right": 90, "down": 180, "left": 270}
    end_map = {"up": 180, "right": 270, "down": 360, "left": 90}
    draw.pieslice(bbox, start=start_map[orientation], end=end_map[orientation], fill=None, outline=outline, width=width)


def draw_triangle(draw: ImageDraw.ImageDraw, pts, fill=None, outline=(0, 0, 0), width=3):
    draw.polygon(pts, fill=fill, outline=outline)


def draw_plus(draw: ImageDraw.ImageDraw, center, size, width=6, fill=(0, 0, 0)):
    cx, cy = center
    s = size // 2
    draw.line((cx - s, cy, cx + s, cy), fill=fill, width=width)
    draw.line((cx, cy - s, cx, cy + s), fill=fill, width=width)


def draw_diamond(draw: ImageDraw.ImageDraw, center, size, fill=None, outline=(0, 0, 0), width=3):
    cx, cy = center
    s = size // 2
    pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
    draw.polygon(pts, fill=fill, outline=outline)


def draw_star(draw: ImageDraw.ImageDraw, center, r, points=5, outline=(0, 0, 0), width=3):
    cx, cy = center
    pts = []
    for i in range(points * 2):
        angle = math.pi * i / points
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(angle - math.pi / 2), cy + rr * math.sin(angle - math.pi / 2)))
    draw.line(pts + [pts[0]], fill=outline, width=width, joint="curve")


def draw_L(draw: ImageDraw.ImageDraw, center, size, outline=(0, 0, 0), width=6):
    cx, cy = center
    s = size
    draw.line((cx - s // 2, cy - s // 2, cx - s // 2, cy + s // 2), fill=outline, width=width)
    draw.line((cx - s // 2, cy + s // 2, cx + s // 2, cy + s // 2), fill=outline, width=width)


def draw_rot_arrow(img: Image.Image, caption: str = "", bold: bool = False) -> Image.Image:
    """Curved rotation arrow on transparent background with optional caption."""
    w, h = 220, 220
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    d.arc([30, 30, w - 30, h - 30], start=210, end=-30, fill=(0, 0, 0, 255), width=6 if bold else 4)
    d.polygon([(w - 45, h // 2 - 10), (w - 85, h // 2 - 6), (w - 70, h // 2 - 28)], fill=(0, 0, 0, 255))
    if caption:
        try:
            f = ImageFont.truetype("DejaVuSans.ttf", 20)
        except Exception:
            f = None
        d.text((w // 2, h - 26), caption, fill=(0, 0, 0, 255), anchor="mm", font=f)
    return canvas.convert("RGB")


def faint_hint_box(side=220, text="؟") -> Image.Image:
    img = new_canvas(side, side)
    d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), side - 20, outline=(150, 150, 150), width=2)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 72)
    except Exception:
        f = None
    d.text((side // 2, side // 2), text, fill=(170, 170, 170), anchor="mm", font=f)
    return img


# ---------------------- Utilities ---------------------------------
def bytes_from_img(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def images_unique(imgs: List[Image.Image]) -> bool:
    sigs = [img.tobytes() for img in imgs]
    return len(sigs) == len(set(sigs))


def hstack(*imgs: Image.Image, pad: int = 16, bg=(255, 255, 255)) -> Image.Image:
    """Horizontal stack with padding, vertically centered."""
    H = max(im.height for im in imgs)
    W = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for im in imgs:
        y = (H - im.height) // 2
        out.paste(im, (x, y))
        x += im.width + pad
    return out


@dataclass
class Question:
    title: str
    image: Image.Image
    options: List[Image.Image]
    correct_index: int
    explanation: str


# ---------------------- Puzzle 1: Paper folding -------------------
def paper_fold_question(seed: int = 0, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    RNG.seed(seed)

    # Build stem parts
    base_size = 220
    folded = new_canvas(base_size, base_size)
    d = ImageDraw.Draw(folded)
    draw_square(d, (10, 10), base_size - 20)
    fold_dir = RNG.choice(["h", "v"])
    # Fold guide
    if fold_dir == "h":
        dashed_line(d, (10, base_size // 2), (base_size - 10, base_size // 2), width=3)
    else:
        dashed_line(d, (base_size // 2, 10), (base_size // 2, base_size - 10), width=3)

    # Random holes
    n_by_level = {"سهل": (1, 2), "متوسط": (2, 3), "صعب": (3, 4)}
    lo, hi = n_by_level.get(difficulty, (1, 2))
    holes = RNG.randint(lo, hi)
    pts = []
    for _ in range(holes):
        x = RNG.randint(40, base_size - 40)
        y = RNG.randint(40, base_size - 40)
        if fold_dir == "h" and y > base_size // 2:
            y = base_size - y
        if fold_dir == "v" and x < base_size // 2:
            x = base_size - x
        pts.append((x, y))
    for (x, y) in pts:
        draw_circle(d, (x, y), 12, outline=(0, 0, 0), width=4)

    # Stem: folded sheet + arrow + hint
    stem = hstack(folded, draw_rot_arrow(caption="افتح"), faint_hint_box())

    # Unfold logic
    def mirror(p):
        x, y = p
        if fold_dir == "h":
            return (x, base_size - y)
        return (base_size - x, y)

    def render_unfolded(correct=True, jitter=0):
        S = base_size
        img = new_canvas(S, S)
        dd = ImageDraw.Draw(img)
        draw_square(dd, (10, 10), S - 20)
        # original holes
        for (x, y) in pts:
            draw_circle(dd, (x, y), 12, outline=(0, 0, 0), width=4)
        # mirrored holes (or wrong variants)
        for (x, y) in pts:
            mx, my = mirror((x, y))
            if not correct and jitter:
                # displace to fake reflection
                mx += RNG.choice([-jitter, jitter])
                my += RNG.choice([-jitter, jitter])
            draw_circle(dd, (mx, my), 12, outline=(0, 0, 0), width=4)
        return img

    correct_img = render_unfolded(correct=True)
    wrong1 = render_unfolded(correct=False, jitter=16 if difficulty != "سهل" else 10)
    wrong2 = ImageOps.mirror(correct_img) if fold_dir == "h" else ImageOps.flip(correct_img)
    wrong3 = new_canvas(correct_img.width, correct_img.height)  # missing mirrors
    d3 = ImageDraw.Draw(wrong3)
    draw_square(d3, (10, 10), base_size - 20)
    for (x, y) in pts:
        draw_circle(d3, (x, y), 12, outline=(0, 0, 0), width=4)

    options = [correct_img, wrong1, wrong2, wrong3]
    RNG.shuffle(options)
    correct_index = options.index(correct_img)

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(
        sys, "بعد فتح الورقة وفق خط الطيّ، أيُّ بديل يطابق نمط الثقوب الصحيح؟", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=st.session_state.get("use_llm", True)
    )
    expl = ollama_chat_or_fallback(
        sys, "الصحيح يُظهر انعكاسًا تامًا حول خط الطيّ مع بقاء مواقع الثقوب الأصلية.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=st.session_state.get("use_llm", True)
    )
    return Question(title=title, image=stem, options=options, correct_index=correct_index, explanation=expl)


# ---------------------- Glyph pools (“رموز متغيّرة”) --------------
def glyph_pool(variable_mode: bool, difficulty: str):
    base = ["plus", "diamond", "circle", "triangle"]
    extra = ["star", "half_up", "half_right", "square_small", "L"]
    if not variable_mode:
        return base
    if difficulty == "سهل":
        return base + ["star", "square_small"]
    if difficulty == "متوسط":
        return base + ["star", "half_up", "square_small", "L"]
    return base + extra  # صعب


def draw_glyph(draw: ImageDraw.ImageDraw, glyph: str, center: Tuple[int, int], size: int):
    if glyph == "plus":
        draw_plus(draw, center, size, width=6)
    elif glyph == "diamond":
        draw_diamond(draw, center, size - 6, outline=(0, 0, 0))
    elif glyph == "circle":
        draw_circle(draw, center, (size // 2) - 4, outline=(0, 0, 0), width=4)
    elif glyph == "triangle":
        cx, cy = center
        half = (size // 2)
        pts = [(cx - half, cy + half), (cx + half, cy + half), (cx, cy - half)]
        draw_triangle(draw, pts, outline=(0, 0, 0))
    elif glyph == "star":
        draw_star(draw, center, (size // 2) - 2, outline=(0, 0, 0))
    elif glyph == "half_up":
        draw_half_circle(draw, center, (size // 2) - 2, orientation="up", outline=(0, 0, 0))
    elif glyph == "half_right":
        draw_half_circle(draw, center, (size // 2) - 2, orientation="right", outline=(0, 0, 0))
    elif glyph == "square_small":
        cx, cy = center
        s = size - 10
        draw_square(draw, (cx - s // 2, cy - s // 2), s, outline=(0, 0, 0), width=3)
    elif glyph == "L":
        draw_L(draw, center, size, outline=(0, 0, 0), width=6)


# ---------------------- Boards & stems (2D) ------------------------
def board_cross(canvas_size=260, seed=0, variable_mode: bool = True, difficulty: str = "سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas_size, canvas_size)
    d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), canvas_size - 20)
    mid = canvas_size // 2
    d.line((10, mid, canvas_size - 10, mid), fill=(0, 0, 0), width=3)
    d.line((mid, 10, mid, canvas_size - 10), fill=(0, 0, 0), width=3)

    pool = glyph_pool(variable_mode, difficulty)
    RNG.shuffle(pool)
    chosen = pool[:4]

    jitter = {"سهل": 6, "متوسط": 10, "صعب": 16}[difficulty]
    offset = 60 if difficulty != "صعب" else 56
    locs = [(mid - offset, mid - offset), (mid + offset, mid - offset), (mid - offset, mid + offset), (mid + offset, mid + offset)]

    for glyph, (cx, cy) in zip(chosen, locs):
        draw_glyph(ImageDraw.Draw(img), glyph, (cx + RNG.randint(-jitter, jitter), cy + RNG.randint(-jitter, jitter)), size=40)
    return img


def board_diag(canvas_size=260, seed=0, variable_mode: bool = True, difficulty: str = "سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas_size, canvas_size)
    d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), canvas_size - 20)
    d.line((10, 10, canvas_size - 10, canvas_size - 10), fill=(0, 0, 0), width=3)
    d.line((10, canvas_size - 10, canvas_size - 10, 10), fill=(0, 0, 0), width=3)
    pool = glyph_pool(variable_mode, difficulty)
    RNG.shuffle(pool)
    chosen = pool[:3]
    mid = canvas_size // 2
    base = 65 if difficulty != "صعب" else 60
    jitter = {"سهل": 6, "متوسط": 10, "صعب": 14}[difficulty]
    slots = [(mid - base, mid - base), (mid + base, mid - base), (mid - base, mid + base)]
    for glyph, (cx, cy) in zip(chosen, slots):
        draw_glyph(d, glyph, (cx + RNG.randint(-jitter, jitter), cy + RNG.randint(-jitter, jitter)), 40)
    return img


def stem_with_rotation(source_img: Image.Image, angle: int) -> Image.Image:
    return hstack(source_img, draw_rot_arrow(caption=f"{angle}°"), faint_hint_box())


def rotate_image(img: Image.Image, angle_deg: int, allow_mirror=False) -> Image.Image:
    out = img.rotate(angle_deg, expand=True, fillcolor=(255, 255, 255))
    if allow_mirror and RNG.random() < 0.25:
        out = ImageOps.mirror(out)
    return out


def quadrant_rotation_question(seed: int = 0, difficulty: str = "سهل", variable_mode: bool = True, use_llm: bool = True) -> Question:
    base = board_cross(canvas_size=260, seed=seed, variable_mode=variable_mode, difficulty=difficulty)
    angle = RNG.choice([90, 180, 270])
    stem = stem_with_rotation(base, angle)

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

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(
        sys, f"انظر إلى اللوحة على اليسار. أيُّ بديل يطابقها بعد تدويرها {angle}°؟", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm
    )
    expl = ollama_chat_or_fallback(
        sys, "الصحيح يحافظ على ترتيب الرموز بالنسبة للمحاور بعد الدوران دون أي انعكاس.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm
    )
    return Question(title=title, image=stem, options=options, correct_index=correct_index, explanation=expl)


def diagonal_rotation_question(seed: int = 0, variable_mode: bool = True, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    base = board_diag(canvas_size=260, seed=seed, variable_mode=variable_mode, difficulty=difficulty)
    angle = RNG.choice([90, 180, 270])
    stem = stem_with_rotation(base, angle)

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

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(
        sys, f"لوحة مقسّمة بالأقطار. أيُّ بديل يطابقها بعد تدويرها {angle}°؟", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm
    )
    expl = ollama_chat_or_fallback(
        sys, "تحقق من مواقع الرموز على القطرين؛ الصحيح فقط يوافق دوران الزاوية المطلوبة.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm
    )
    return Question(title=title, image=stem, options=options, correct_index=correct_index, explanation=expl)


# ---------------------- 3D isometric cube rotation ----------------
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


def draw_iso_cubes(coords: List[Tuple[int, int, int]], size=28, img_size=(260, 240)) -> Image.Image:
    img = new_canvas(*img_size)
    d = ImageDraw.Draw(img)
    us = [iso_project(p, size) for p in coords]
    minx = min(u for u, _ in us)
    maxx = max(u for u, _ in us)
    miny = min(v for _, v in us)
    maxy = max(v for _, v in us)
    cx = (img_size[0] - (maxx - minx)) / 2 - minx
    cy = (img_size[1] - (maxy - miny)) / 2 - miny
    order = sorted(range(len(coords)), key=lambda i: sum(coords[i]))
    for i in order:
        x, y, z = coords[i]
        u, v = iso_project((x, y, z), size)
        u += cx
        v += cy
        top = [(u, v - size), (u + size * 0.5, v - size * 0.5), (u, v), (u - size * 0.5, v - size * 0.5)]
        right = [(u, v), (u + size * 0.5, v - size * 0.5), (u + size * 0.5, v + size * 0.5), (u, v + size)]
        left = [(u, v), (u - size * 0.5, v - size * 0.5), (u - size * 0.5, v + size * 0.5), (u, v + size)]
        d.polygon(top, fill=(220, 220, 220), outline=(0, 0, 0))
        d.polygon(right, fill=(190, 190, 190), outline=(0, 0, 0))
        d.polygon(left, fill=(160, 160, 160), outline=(0, 0, 0))
    return img


def random_polycube(n=4, seed=0) -> List[Tuple[int, int, int]]:
    RNG.seed(seed)
    pts = [(0, 0, 0)]
    while len(pts) < n:
        x, y, z = RNG.choice(pts)
        nx, ny, nz = RNG.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        cand = (x + nx, y + ny, z + nz)
        if cand not in pts:
            pts.append(cand)
        minx = min(p[0] for p in pts)
        miny = min(p[1] for p in pts)
        minz = min(p[2] for p in pts)
        pts = [(p[0] - minx, p[1] - miny, p[2] - minz) for p in pts]
    return pts


def apply_rot(coords: List[Tuple[int, int, int]], R: np.ndarray) -> List[Tuple[int, int, int]]:
    arr = np.array(coords).T
    out = np.rint(R @ arr).astype(int).T
    minx = out[:, 0].min()
    miny = out[:, 1].min()
    minz = out[:, 2].min()
    out = out - np.array([minx, miny, minz])
    return [tuple(map(int, p)) for p in out.tolist()]


def cubes_rotation_question(seed: int = 0, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    RNG.seed(seed)
    n = {"سهل": 4, "متوسط": 5, "صعب": 6}[difficulty]
    shape = random_polycube(n=n, seed=seed)

    # Stem: show the original 3D model ONLY (no giveaway) + rotation arrow + hint box
    src_img = draw_iso_cubes(shape, img_size=(260, 240))
    angle_txt = RNG.choice(["90°", "180°", "270°"])  # just a cue; actual rotations are in options via matrices
    stem = hstack(src_img, draw_rot_arrow(caption=f"دوِّر {angle_txt}", bold=True), faint_hint_box())

    # Correct orientation (random matrix)
    R_true = RNG.choice(ORIENTS)
    correct_coords = apply_rot(shape, R_true)
    correct_img = draw_iso_cubes(correct_coords, img_size=(240, 220))

    # Distractors: different valid orientations (not equal to correct)
    used = {tuple(map(int, np.rint(R_true).flatten()))}
    options: List[Image.Image] = [correct_img]
    while len(options) < 4:
        R_alt = RNG.choice(ORIENTS)
        key = tuple(map(int, np.rint(R_alt).flatten()))
        if key in used:
            continue
        used.add(key)
        options.append(draw_iso_cubes(apply_rot(shape, R_alt), img_size=(240, 220)))

    RNG.shuffle(options)
    correct_index = options.index(correct_img)

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(
        sys, "انظر إلى المجسّم على اليسار. أيُّ بديل يطابقه بعد تدويره كما في السهم؟", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm
    )
    expl = ollama_chat_or_fallback(
        sys, "طابِق هيئة المكعّبات بعد التدوير فقط (لا انعكاس). البديل الصحيح يُحافظ على نفس الشكل من زاوية أخرى.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm
    )
    return Question(title=title, image=stem, options=options, correct_index=correct_index, explanation=expl)


# ---------------------- Shape assembly -----------------------------
def shape_assembly_question(seed: int = 0, use_llm: bool = True) -> Question:
    RNG.seed(seed)
    target = RNG.choice(["square", "pentagon"])

    # Stem target (right) + hint
    if target == "square":
        tgt = new_canvas(240, 240)
        draw_square(ImageDraw.Draw(tgt), (20, 20), 200)
    else:
        tgt = new_canvas(240, 240)
        d = ImageDraw.Draw(tgt)
        cx, cy, r = 120, 120, 95
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(-math.pi / 2, 1.5 * math.pi, 6)[:-1]]
        d.polygon(pts, outline=(0, 0, 0), fill=None)
    stem = hstack(faint_hint_box(text="ركّب"), draw_rot_arrow(caption=""), tgt)

    # Options (groups of pieces)
    def tri_img(angle=0, flip=False):
        S = 120
        pad = 8
        im = new_canvas(S + pad * 2, S + pad * 2)
        dr = ImageDraw.Draw(im)
        pts = [(pad + 10, pad + 10), (pad + S - 10, pad + 10), (pad + 10, pad + S - 10)]
        if flip:
            pts = [(pad + 10, pad + 10), (pad + S - 10, pad + 10), (pad + S - 10, pad + S - 10)]
        draw_triangle(dr, pts, outline=(0, 0, 0))
        return im.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    def panel(imgs: List[Image.Image]) -> Image.Image:
        w = sum(i.width for i in imgs) + 20 * (len(imgs) - 1) + 40
        h = max(i.height for i in imgs) + 40
        out = new_canvas(w, h)
        x = 20
        for im in imgs:
            out.paste(im, (x, (h - im.height) // 2))
            x += im.width + 20
        return out

    if target == "square":
        optA = panel([tri_img(), tri_img(angle=180, flip=True)])  # correct
        rect = new_canvas(100, 140)
        ImageDraw.Draw(rect).rectangle([10, 10, 90, 130], outline=(0, 0, 0), width=3)
        rhombus = new_canvas(120, 120)
        draw_diamond(ImageDraw.Draw(rhombus), (60, 60), 80, outline=(0, 0, 0))
        optB = panel([rect, rhombus])
        optC = panel([tri_img(angle=90), rhombus.rotate(45, expand=True, fillcolor=(255, 255, 255))])
        optD = panel([tri_img(flip=True), rect])
        options = [optA, optB, optC, optD]
        correct_index = 0
    else:
        quad = new_canvas(160, 140)
        ImageDraw.Draw(quad).polygon([(20, 120), (80, 20), (140, 60), (100, 120)], outline=(0, 0, 0), fill=None)
        quad2 = new_canvas(160, 140)
        ImageDraw.Draw(quad2).polygon([(20, 60), (60, 20), (140, 80), (100, 120)], outline=(0, 0, 0), fill=None)
        circ = new_canvas(160, 140)
        ImageDraw.Draw(circ).ellipse([30, 30, 130, 110], outline=(0, 0, 0), width=3)
        tri = new_canvas(160, 140)
        draw_triangle(ImageDraw.Draw(tri), [(20, 120), (80, 20), (140, 120)], outline=(0, 0, 0))
        options = [panel([quad, quad2]), panel([quad, circ]), panel([quad2, tri]), panel([circ, tri])]
        correct_index = 0

    sys = "أنت مساعد تعليمي بالعربية."
    title_hint = "أي مجموعة من القطع يمكن أن تُكوِّن الشكل المطلوب؟"
    title = ollama_chat_or_fallback(sys, title_hint, model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=st.session_state.get("use_llm", True))
    expl = ollama_chat_or_fallback(
        sys, "نقارن الحواف والزوايا والمساحات؛ المجموعة الصحيحة يمكن ترتيبها لتكمل حدود الشكل دون فجوات.", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=st.session_state.get("use_llm", True)
    )
    return Question(title=title, image=stem, options=options, correct_index=correct_index, explanation=expl)


# ---------------------- Streamlit Page 7 (UI/UX) -------------------
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH))
st.sidebar.markdown("### الإعدادات")

left, right = st.sidebar.columns([1, 1])
with left:
    n_q = st.number_input("عدد الأسئلة", 1, 24, 8, help="عدد العناصر التي سيتم توليدها في الصفحة.")
with right:
    difficulty = st.selectbox("مستوى الصعوبة", ["سهل", "متوسط", "صعب"])

seed_base = st.sidebar.number_input("البذرة (Seed)", 0, 10_000_000, 12345, help="لإعادة إنتاج نفس العناصر.")
rand_seed = st.sidebar.button("🎲 بذرة عشوائية")
if rand_seed:
    seed_base = RNG.randint(0, 10_000_000)
    st.session_state["_seed_base"] = seed_base
else:
    seed_base = st.session_state.get("_seed_base", seed_base)

st.session_state["llm_model"] = st.sidebar.text_input("نموذج Ollama (اختياري)", value="qwen2.5:3b")
st.session_state["use_llm"] = st.sidebar.checkbox("استخدم LLM لكتابة التعليمات/الشرح", value=True)
variable_symbols = st.sidebar.toggle("وضع **رموز متغيّرة** (تنويع الرموز)", value=True)

puzzle_types = st.sidebar.multiselect(
    "أنواع الأسئلة",
    ["طيّ الورق", "تدوير رباعي", "تدوير قطري", "تدوير مكعّبات ثلاثي", "تركيب شكل"],
    default=["طيّ الورق", "تدوير رباعي", "تدوير قطري", "تدوير مكعّبات ثلاثي", "تركيب شكل"],
)

col1, col2 = st.columns([1, 1])
with col1:
    gen = st.button("🚀 إنشاء الأسئلة", use_container_width=True)
with col2:
    st.caption("كل سؤال يُظهر **المجسّم/اللوحة المرجعية** في الأعلى بوضوح، ثم البدائل بالأسفل.")

st.title("مولّد أسئلة ذكاء مرئية")
st.write("مطابق لنمط الأمثلة المرفقة. كل تشغيل يعطي **أسئلة مختلفة** بفضل العشوائية المضبوطة بالبذرة.")

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
    generated: List[Question] = []

    order: List[str] = []
    while len(order) < n_q:
        order.extend(puzzle_types)
    RNG.shuffle(order)
    order = order[:n_q]

    answers_csv = io.StringIO()
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, kind in enumerate(order, 1):
            # strong variation
            qseed = seed_base ^ (RNG.randint(1, 1_000_000_007) + idx * 9973)
            q = build_by_type(kind, seed=qseed)
            generated.append(q)

            with st.container(border=True):
                st.markdown(f"#### سؤال {idx}: {q.title}")
                st.image(q.image, use_container_width=True)
                cols = st.columns(4, gap="small")
                opts = (q.options + [q.options[-1]])[:4] if q.options else []
                for i, (c, col) in enumerate(zip(opts, cols)):
                    col.image(c, use_container_width=True)
                    col.markdown(
                        f"<div style='text-align:center;font-weight:bold;font-size:20px'>{AR_LETTERS[i]}</div>",
                        unsafe_allow_html=True,
                    )

            zf.writestr(f"q{idx}_statement.png", bytes_from_img(q.image))
            for i, c in enumerate(opts, start=1):
                zf.writestr(f"q{idx}_opt_{i}.png", bytes_from_img(c))
            answers_csv.write(f"{idx},{kind},{AR_LETTERS[q.correct_index]},{qseed}\n")

            with st.expander("إظهار الحل/الشرح"):
                st.markdown(f"**الإجابة الصحيحة:** {AR_LETTERS[q.correct_index]}")
                st.write(q.explanation if st.session_state.get("use_llm", True) else "الصحيح الوحيد يوافق قاعدة السؤال (انعكاس/تدوير/تجميع).")

        zf.writestr("answers.csv", answers_csv.getvalue().encode("utf-8"))

    st.download_button(
        "⬇️ تنزيل كل الأسئلة (ZIP)",
        data=zip_buf.getvalue(),
        file_name="arabic_visual_iq_questions.zip",
        mime="application/zip",
        use_container_width=True,
    )
else:
    st.info("اضغط **إنشاء الأسئلة** لبدء التوليد. كل سؤال يعرض المرجع (المجسّم/اللوحة) بوضوح في أعلى البطاقة.")
