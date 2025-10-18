# -*- coding: utf-8 -*-
"""
Streamlit page 7: Arabic visual IQ question generator (no API/JSON)
- Paper folding & hole punching
- 2D rotation (Cross-board & Diagonal/X-board)
- 3D isometric cube rotation
- Shape assembly (which set forms the target?)
- NEW: “رموز متغيّرة” mode (extra glyphs: نجمة، نصف دائرة، مربّع صغير، شُكل L)
- Difficulty tiers: سهل / متوسط / صعب (affects variety, jitter, distractors)

File path: pages/7_spacial_questions.py
Do NOT call st.set_page_config() here; Home.py manages it.
"""

import io
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
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
    """Why: extra glyph for 'رموز متغيّرة' mode (visually distinct orientation)."""
    cx, cy = center
    bbox = [cx - r, cy - r, cx + r, cy + r]
    start, end = {"up": 0, "right": 90, "down": 180, "left": 270}[orientation], {"up": 180, "right": 270, "down": 360, "left": 90}[orientation]
    draw.pieslice(bbox, start=start, end=end, fill=None, outline=outline, width=width)


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
    """Classic star polygon."""
    cx, cy = center
    pts = []
    for i in range(points * 2):
        angle = math.pi * i / points
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(angle - math.pi / 2), cy + rr * math.sin(angle - math.pi / 2)))
    draw.line(pts + [pts[0]], fill=outline, width=width, joint="curve")


def draw_L(draw: ImageDraw.ImageDraw, center, size, outline=(0, 0, 0), width=6):
    """Simple 'L' bar (directional)."""
    cx, cy = center
    s = size
    # vertical
    draw.line((cx - s // 2, cy - s // 2, cx - s // 2, cy + s // 2), fill=outline, width=width)
    # horizontal
    draw.line((cx - s // 2, cy + s // 2, cx + s // 2, cy + s // 2), fill=outline, width=width)


# ---------------------- Utilities ---------------------------------
def bytes_from_img(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def images_unique(imgs: List[Image.Image]) -> bool:
    sigs = [img.tobytes() for img in imgs]
    return len(sigs) == len(set(sigs))


@dataclass
class Question:
    title: str
    image: Image.Image
    options: List[Image.Image]
    correct_index: int
    explanation: str


# ---------------------- Puzzle 1: Paper folding -------------------
def paper_fold_question(seed: int = 0, difficulty: str = "سهل", use_shapes: bool = True, use_llm: bool = True) -> Question:
    RNG.seed(seed)

    W, H = 900, 560
    img = new_canvas(W, H)
    draw = ImageDraw.Draw(img)

    fold_dir = RNG.choice(["h", "v"])
    base_size = 180
    preview_x, preview_y = 60, 60

    # Preview sheet + fold guide
    if fold_dir == "h":
        draw_square(draw, (preview_x, preview_y), base_size)
        dashed_line(draw, (preview_x, preview_y + base_size), (preview_x + base_size, preview_y + base_size), width=2)
        dashed_line(draw, (preview_x, preview_y + base_size), (preview_x, preview_y + base_size * 2), width=2)
        dashed_line(draw, (preview_x + base_size, preview_y + base_size), (preview_x + base_size, preview_y + base_size * 2), width=2)
        dashed_line(draw, (preview_x, preview_y + base_size * 2), (preview_x + base_size, preview_y + base_size * 2), width=2)
    else:
        draw_square(draw, (preview_x + base_size // 2, preview_y), base_size)
        dashed_line(draw, (preview_x, preview_y), (preview_x + base_size // 2, preview_y), width=2)
        dashed_line(draw, (preview_x, preview_y), (preview_x, preview_y + base_size), width=2)
        dashed_line(draw, (preview_x, preview_y + base_size), (preview_x + base_size // 2, preview_y + base_size), width=2)
        dashed_line(draw, (preview_x + base_size // 2, preview_y), (preview_x + base_size // 2, preview_y + base_size), width=2)

    # Random 1–3 circular holes
    n_by_level = {"سهل": (1, 2), "متوسط": (2, 3), "صعب": (3, 3)}
    lo, hi = n_by_level.get(difficulty, (1, 2))
    n_marks = RNG.randint(lo, hi)
    marks: List[Tuple[str, Tuple[int, int]]] = []
    for _ in range(n_marks):
        rx = RNG.randint(22, base_size - 22)
        ry = RNG.randint(22, base_size - 22)
        marks.append(("circle", (preview_x + rx + (0 if fold_dir == "h" else base_size // 2), preview_y + ry)))

    for _, (cx, cy) in marks:
        draw_circle(draw, (cx, cy), 12, outline=(0, 0, 0), width=4)

    # Fold arrow
    ax = preview_x + base_size + 120
    ay = preview_y + base_size // 2
    draw.arc([ax - 60, ay - 60, ax + 60, ay + 60], start=180, end=0, fill=(0, 0, 0), width=4)
    draw.polygon([(ax + 60, ay), (ax + 25, ay - 15), (ax + 25, ay + 15)], fill=(0, 0, 0))

    def mirror(point):
        x, y = point
        if fold_dir == "h":
            dy = y - preview_y
            return (x, preview_y + base_size - dy)
        else:
            dx = x - (preview_x + base_size // 2)
            return (preview_x + base_size - dx, y)

    def render_unfolded(mark_positions: List[Tuple[str, Tuple[int, int]]], wrong_variant: Optional[int] = None) -> Image.Image:
        S = 220
        pad = 10
        img_opt = new_canvas(S + pad * 2, S + pad * 2)
        d = ImageDraw.Draw(img_opt)
        draw_square(d, (pad, pad), S)

        def map_point(px, py):
            if fold_dir == "h":
                offx = px - preview_x
                offy = py - preview_y
            else:
                offx = px - (preview_x + base_size // 2)
                offy = py - preview_y
            nx = pad + int(offx * (S / base_size))
            ny = pad + int(offy * (S / base_size))
            return nx, ny

        pts = [("circle", map_point(cx, cy)) for _, (cx, cy) in mark_positions]
        if wrong_variant is None:
            for _, (cx, cy) in mark_positions:
                mx, my = mirror((cx, cy))
                pts.append(("circle", map_point(mx, my)))
        elif wrong_variant == 1:
            spread = 12 if difficulty != "صعب" else 16
            for _, (cx, cy) in mark_positions:
                if fold_dir == "h":
                    mx, my = (cx + (spread if cx < preview_x + base_size / 2 else -spread), cy)
                else:
                    mx, my = (cx, cy + (spread if cy < preview_y + base_size / 2 else -spread))
                pts.append(("circle", map_point(mx, my)))
        elif wrong_variant == 2:
            for _, (cx, cy) in mark_positions:
                pts.append(("circle", map_point(cx, cy)))

        for _, (x, y) in pts:
            draw_circle(d, (x, y), 12, outline=(0, 0, 0), width=4)
        return img_opt

    correct = render_unfolded(marks, wrong_variant=None)
    wrongs = [render_unfolded(marks, wrong_variant=i) for i in (0, 1, 2)]

    options = [correct] + wrongs
    tries = 0
    while not images_unique(options) and tries < 6:
        wrongs = [render_unfolded(marks, wrong_variant=i) for i in (0, 1, 2)]
        options = [correct] + wrongs
        tries += 1

    RNG.shuffle(options)
    correct_index = options.index(correct)

    sys = "أنت مساعد تعليمي يكتب تعليمات قصيرة وواضحة بالعربية."
    title = ollama_chat_or_fallback(
        sys,
        "بعد فتح الورقة، أي بديل يطابق نمط الثقوب الصحيح؟ اختر الإجابة.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
        enabled=st.session_state.get("use_llm", True),
    )
    expl = ollama_chat_or_fallback(
        sys,
        "الصحيح الوحيد يُظهر انعكاس الثقوب حول خط الطيّ (تماثل مرآتي).",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
        enabled=st.session_state.get("use_llm", True),
    )
    return Question(title=title, image=img, options=options, correct_index=correct_index, explanation=expl)


# ---------------------- Glyph pools (NEW “رموز متغيّرة”) ----------
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


# ---------------------- Puzzle 2a: Quadrant rotation ---------------
def random_quadrant_board(canvas_size=280, seed=0, variable_mode: bool = True, difficulty: str = "سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas_size, canvas_size)
    d = ImageDraw.Draw(img)

    # Board + cross
    draw_square(d, (10, 10), canvas_size - 20)
    mid = canvas_size // 2
    d.line((10, mid, canvas_size - 10, mid), fill=(0, 0, 0), width=3)
    d.line((mid, 10, mid, canvas_size - 10), fill=(0, 0, 0), width=3)

    # Pool and placements
    pool = glyph_pool(variable_mode, difficulty)
    RNG.shuffle(pool)
    chosen = pool[:4]

    jitter = {"سهل": 6, "متوسط": 10, "صعب": 16}[difficulty]
    base_offs = 64 if difficulty != "صعب" else 58
    offs = [(mid - base_offs, mid - base_offs), (mid + base_offs, mid - base_offs),
            (mid - base_offs, mid + base_offs), (mid + base_offs, mid + base_offs)]

    for glyph, (cx, cy) in zip(chosen, offs):
        jx = RNG.randint(-jitter, jitter)
        jy = RNG.randint(-jitter, jitter)
        draw_glyph(d, glyph, (cx + jx, cy + jy), size=42)

    return img


def rotate_image(img: Image.Image, angle_deg: int, allow_mirror=False) -> Image.Image:
    out = img.rotate(angle_deg, expand=True, fillcolor=(255, 255, 255))
    if allow_mirror and RNG.random() < 0.3:
        out = ImageOps.mirror(out)
    return out


def quadrant_rotation_question(seed: int = 0, difficulty: str = "سهل", variable_mode: bool = True, use_llm: bool = True) -> Question:
    base = random_quadrant_board(canvas_size=280, seed=seed, variable_mode=variable_mode, difficulty=difficulty)
    angle = RNG.choice([90, 180, 270])

    correct = rotate_image(base, -angle, allow_mirror=False)
    # Distractors tuned by difficulty
    if difficulty == "سهل":
        distract = [
            rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])),
            rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])),
            ImageOps.mirror(base),
        ]
    elif difficulty == "متوسط":
        distract = [
            rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])),
            ImageOps.flip(rotate_image(base, -angle)),  # mirrored after correct rotation
            ImageOps.mirror(base),
        ]
    else:  # صعب
        distract = [
            ImageOps.mirror(rotate_image(base, -angle)),  # mirror+rotate (very deceptive)
            rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])),
            ImageOps.flip(base),
        ]

    options = [correct] + distract
    tries = 0
    while not images_unique(options) and tries < 4:
        RNG.shuffle(distract)
        options = [correct] + distract
        tries += 1

    RNG.shuffle(options)
    correct_index = options.index(correct)

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(
        sys,
        f"بعد تدوير اللوحة بمقدار {angle}°، أي بديل يطابق الصورة؟",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
        enabled=use_llm,
    )
    expl = ollama_chat_or_fallback(
        sys,
        "الصحيح يحافظ على ترتيب الرموز بالنسبة للمحاور بعد الدوران دون أي انعكاس.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
        enabled=use_llm,
    )
    return Question(title=title, image=base, options=options, correct_index=correct_index, explanation=expl)


# ---------------------- Puzzle 2b: Diagonal (X-board) rotation ----
def diagonal_board(canvas_size=280, seed=0, variable_mode: bool = True, difficulty: str = "سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas_size, canvas_size)
    d = ImageDraw.Draw(img)

    draw_square(d, (10, 10), canvas_size - 20)
    # diagonals
    d.line((10, 10, canvas_size - 10, canvas_size - 10), fill=(0, 0, 0), width=3)
    d.line((10, canvas_size - 10, canvas_size - 10, 10), fill=(0, 0, 0), width=3)

    pool = glyph_pool(variable_mode, difficulty)
    RNG.shuffle(pool)
    chosen = pool[:3]

    mid = canvas_size // 2
    base = 70 if difficulty != "صعب" else 65
    jitter = {"سهل": 6, "متوسط": 10, "صعب": 14}[difficulty]
    slots = [(mid - base, mid - base), (mid + base, mid - base), (mid - base, mid + base)]
    for glyph, (cx, cy) in zip(chosen, slots):
        draw_glyph(d, glyph, (cx + RNG.randint(-jitter, jitter), cy + RNG.randint(-jitter, jitter)), 42)

    return img


def diagonal_rotation_question(seed: int = 0, variable_mode: bool = True, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    base = diagonal_board(280, seed, variable_mode, difficulty)
    angle = RNG.choice([90, 180, 270])

    correct = rotate_image(base, -angle)
    if difficulty == "سهل":
        distract = [
            rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])),
            rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])),
            ImageOps.mirror(base),
        ]
    elif difficulty == "متوسط":
        distract = [
            ImageOps.flip(rotate_image(base, -angle)),
            rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])),
            ImageOps.mirror(base),
        ]
    else:
        distract = [
            ImageOps.mirror(rotate_image(base, -angle)),
            rotate_image(base, -RNG.choice([a for a in [90, 180, 270] if a != angle])),
            ImageOps.flip(base),
        ]

    options = [correct] + distract
    RNG.shuffle(options)
    correct_index = options.index(correct)

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(
        sys,
        f"لوحة مقسّمة بالأقطار. بعد تدويرها {angle}°، أي بديل صحيح؟",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
        enabled=use_llm,
    )
    expl = ollama_chat_or_fallback(
        sys,
        "تحقق من مواقع الرموز على القطرين؛ الإجابة الصحيحة فقط تماثل الدوران المطلوب.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
        enabled=use_llm,
    )
    return Question(title=title, image=base, options=options, correct_index=correct_index, explanation=expl)


# ---------------------- Puzzle 3: 3D isometric cube rotation ------
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


def draw_iso_cubes(coords: List[Tuple[int, int, int]], size=28, img_size=(320, 280)) -> Image.Image:
    img = new_canvas(*img_size)
    d = ImageDraw.Draw(img)
    us = [iso_project(p, size) for p in coords]
    minx = min(u for u, _ in us)
    maxx = max(u for u, _ in us)
    miny = min(v for _, v in us)
    maxy = max(v for _, v in us)
    cx = (img_size[0] - (maxx - minx)) / 2 - minx
    cy = (img_size[1] - (maxy - miny)) / 2 - miny
    order = sorted(range(len(coords)), key=lambda i: sum(coords[i]))  # back-to-front
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
    R_true = RNG.choice(ORIENTS)
    correct_coords = apply_rot(shape, R_true)
    top_img = draw_iso_cubes(shape, img_size=(320, 280))
    ref_img = draw_iso_cubes(correct_coords, img_size=(320, 280))

    W, H = 900, 360
    stmt = new_canvas(W, H)
    d = ImageDraw.Draw(stmt)
    stmt.paste(top_img, (60, 40))
    stmt.paste(ref_img, (W - 60 - ref_img.width, 40))
    ax, ay = W // 2, H // 2
    d.arc([ax - 80, ay - 60, ax + 80, ay + 60], start=210, end=-30, fill=(0, 0, 0), width=5)
    d.polygon([(ax + 75, ay - 20), (ax + 30, ay - 10), (ax + 40, ay - 40)], fill=(0, 0, 0))

    opts: List[Image.Image] = []
    correct_img = draw_iso_cubes(correct_coords, img_size=(300, 260))
    opts.append(correct_img)
    used = {tuple(map(int, np.rint(R_true).flatten()))}
    while len(opts) < 4:
        R_alt = RNG.choice(ORIENTS)
        key = tuple(map(int, np.rint(R_alt).flatten()))
        if key in used:
            continue
        used.add(key)
        coords = apply_rot(shape, R_alt)
        opts.append(draw_iso_cubes(coords, img_size=(300, 260)))

    RNG.shuffle(opts)
    correct_index = opts.index(correct_img)

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(
        sys,
        "انظر إلى المجسّم على اليسار. أيُّ بديل في الأسفل يطابقه بعد تدويره كما في السهم؟",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
        enabled=use_llm,
    )
    expl = ollama_chat_or_fallback(
        sys,
        "نطابق مواقع المكعّبات بعد الدوران فقط (بلا انعكاس/تبديل).",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
        enabled=use_llm,
    )
    return Question(title=title, image=stmt, options=opts, correct_index=correct_index, explanation=expl)


# ---------------------- Puzzle 4: Shape assembly -------------------
def shape_assembly_question(seed: int = 0, use_llm: bool = True) -> Question:
    RNG.seed(seed)
    target = RNG.choice(["square", "pentagon"])

    W, H = 900, 380
    img = new_canvas(W, H)
    d = ImageDraw.Draw(img)

    if target == "square":
        draw_square(d, (W - 300, 80), 180)
        title_hint = "أي مجموعة من القطع يمكن أن تُكوِّن مربعًا كهذا؟"

        def tri_img(angle=0, flip=False):
            S = 140
            pad = 10
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
        cx, cy, r = W - 210, 170, 110
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(-math.pi / 2, 1.5 * math.pi, 6)[:-1]]
        d.polygon(pts, outline=(0, 0, 0), fill=None)
        title_hint = "أي مجموعة من القطع يمكن أن تُكوِّن خماسي الأضلاع؟"

        def quad_panel(lean=1) -> Image.Image:
            im = new_canvas(160, 140)
            dr = ImageDraw.Draw(im)
            p = [(20, 120), (80, 20), (140, 60), (100, 120)]
            if lean < 0:
                p = [(20, 60), (60, 20), (140, 80), (100, 120)]
            dr.polygon(p, outline=(0, 0, 0), fill=None)
            return im

        def panel(imgs: List[Image.Image]) -> Image.Image:
            w = sum(i.width for i in imgs) + 20 * (len(imgs) - 1) + 40
            h = max(i.height for i in imgs) + 40
            out = new_canvas(w, h)
            x = 20
            for im in imgs:
                out.paste(im, (x, (h - im.height) // 2))
                x += im.width + 20
            return out

        optA = quad_panel(1)
        optB = quad_panel(-1)
        optC = new_canvas(160, 140)
        ImageDraw.Draw(optC).ellipse([30, 30, 130, 110], outline=(0, 0, 0), width=3)
        optD = new_canvas(160, 140)
        draw_triangle(ImageDraw.Draw(optD), [(20, 120), (80, 20), (140, 120)], outline=(0, 0, 0))
        options = [panel([optA, optB]), panel([optA, optC]), panel([optB, optD]), panel([optC, optD])]
        correct_index = 0

    d.text((60, 40), "السؤال:", fill=(0, 0, 0))

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(sys, title_hint, model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=st.session_state.get("use_llm", True))
    expl = ollama_chat_or_fallback(
        sys,
        "نقارن مساحات القطع واتجاه الحواف؛ المجموعة الصحيحة يمكن ترتيبها لتكمل حدود الشكل دون فجوات.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
        enabled=st.session_state.get("use_llm", True),
    )
    return Question(title=title, image=img, options=options, correct_index=correct_index, explanation=expl)


# ---------------------- Streamlit Page 7 (UI/UX) -------------------
# Sidebar
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

# Action bar
col1, col2 = st.columns([1, 1])
with col1:
    gen = st.button("🚀 إنشاء الأسئلة", use_container_width=True)
with col2:
    st.caption("كل سؤال يُنشأ عشوائيًا بمشتتات مدروسة. فعّل **رموز متغيّرة** لزيادة التنوع.")

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
            # vary seeds strongly to ensure different results every time
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

            # ZIP & answers
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
    st.info("اضغط **إنشاء الأسئلة** لبدء التوليد. غيّر البذرة أو فعّل **رموز متغيّرة** لمزيد من التنوع.")
