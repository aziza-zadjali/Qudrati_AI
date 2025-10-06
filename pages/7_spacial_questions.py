# pages/spacial_app.py
# -*- coding: utf-8 -*-
"""
Streamlit page: Arabic visual IQ question generator
- Paper folding & hole punching
- 2D quadrant rotation
- 3D isometric cube rotation
- Shape assembly (which set forms the target?)
Optional: Arabic wording/explanations via a local open-source LLM using Ollama.

NOTE:
- This file is designed to live under the `pages/` directory of a Streamlit multipage app.
- DO NOT call st.set_page_config() here; the main app (Home.py) manages it.
"""

import io
import math
import os
import random
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import requests
import streamlit as st

# ------------------------------ Config ------------------------------
AR_LETTERS = ["أ", "ب", "ج", "د"]
RNG = random.Random()

# ------------------ LLM (optional via Ollama) ----------------------
def ollama_chat_or_fallback(system: str, user: str, model: str, max_tokens: int = 256) -> str:
    """
    Uses a local open-source LLM via Ollama if available; otherwise fallback Arabic text.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    try:
        r = requests.post(url, json=payload, timeout=8)
        if r.ok:
            out = r.json().get("message", {}).get("content", "").strip()
            if out:
                return out
    except Exception:
        pass
    # Fallback (no Ollama)
    return user

# ---------------------- Helpers: drawing ---------------------------
def new_canvas(w: int, h: int, bg=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (w, h), bg)

def dashed_line(draw: ImageDraw.ImageDraw, start, end, dash_len=6, gap_len=6, fill=(0, 0, 0), width=1):
    # PIL lacks native dashed lines
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

# ---------------------- Utilities ---------------------------------
def bytes_from_img(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@dataclass
class Question:
    title: str
    image: Image.Image
    options: List[Image.Image]
    correct_index: int
    explanation: str

# ---------------------- Puzzle 1: Paper folding -------------------
def paper_fold_question(seed: int = 0, difficulty: str = "سهل", use_shapes: bool = True) -> Question:
    RNG.seed(seed)
    W, H = 900, 560
    img = new_canvas(W, H)
    draw = ImageDraw.Draw(img)

    # Fold settings
    fold_dir = RNG.choice(["h", "v"])
    base_size = 180

    # Top-left: folded preview
    preview_x, preview_y = 60, 60

    # Draw folded sheet outline (upper visible half)
    if fold_dir == "h":
        draw_square(draw, (preview_x, preview_y), base_size)
        # dashed for hidden lower half
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

    # Punch marks
    marks = []
    if use_shapes:
        for _ in range(RNG.choice([1, 2])):
            rx = RNG.randint(20, base_size - 20)
            ry = RNG.randint(20, base_size - 20)
            marks.append(("circle", (preview_x + rx + (0 if fold_dir == "h" else base_size // 2), preview_y + ry)))
    else:
        rx = RNG.randint(30, base_size - 30)
        ry = RNG.randint(30, base_size - 30)
        marks.append(("circle", (preview_x + rx + (0 if fold_dir == "h" else base_size // 2), preview_y + ry)))

    # Draw marks on folded
    for kind, (cx, cy) in marks:
        if kind == "circle":
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
            y2 = preview_y + base_size - dy
            return (x, y2)
        else:
            dx = x - (preview_x + base_size // 2)
            x2 = preview_x + base_size - dx
            return (x2, y)

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
        elif wrong_variant == 0:
            pass
        elif wrong_variant == 1:
            for _, (cx, cy) in mark_positions:
                if fold_dir == "h":
                    mx, my = (cx + (10 if cx < preview_x + base_size / 2 else -10), cy)
                else:
                    mx, my = (cx, cy + (10 if cy < preview_y + base_size / 2 else -10))
                pts.append(("circle", map_point(mx, my)))
        elif wrong_variant == 2:
            for _, (cx, cy) in mark_positions:
                pts.append(("circle", map_point(cx, cy)))

        for _, (x, y) in pts:
            draw_circle(d, (x, y), 12, outline=(0, 0, 0), width=4)

        return img_opt

    correct = render_unfolded(marks, wrong_variant=None)
    wrongs = [
        render_unfolded(marks, wrong_variant=0),
        render_unfolded(marks, wrong_variant=1),
        render_unfolded(marks, wrong_variant=2),
    ]
    options = [correct] + wrongs
    RNG.shuffle(options)
    correct_index = options.index(correct)

    sys = "أنت مساعد تعليمي يكتب تعليمات قصيرة وواضحة بالعربية."
    prompt = "اكتب تعليمة قصيرة لهذا السؤال: ورقة مطوية مع ثقوب كما في الأعلى. بعد فتح الورقة، أي بديل صحيح؟ اكتب سطرًا واحدًا فقط."
    title = ollama_chat_or_fallback(sys, prompt, model=st.session_state.get("llm_model", "qwen2.5:3b"))
    expl = ollama_chat_or_fallback(
        sys,
        "اشرح باختصار لماذا البديل الصحيح هو الوحيد الذي يُظهر انعكاس العلامات حول خط الطيّ.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
    )
    return Question(title=title, image=img, options=options, correct_index=correct_index, explanation=expl)

# ---------------------- Puzzle 2: Quadrant rotation ---------------
def draw_quadrant_symbols(canvas_size=280, seed=0) -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas_size, canvas_size)
    d = ImageDraw.Draw(img)
    draw_square(d, (10, 10), canvas_size - 20)
    mid = canvas_size // 2
    d.line((10, mid, canvas_size - 10, mid), fill=(0, 0, 0), width=3)
    d.line((mid, 10, mid, canvas_size - 10), fill=(0, 0, 0), width=3)
    draw_plus(d, (mid - 60, mid - 60), 40, width=6)
    draw_diamond(d, (mid + 60, mid - 60), 36, outline=(0, 0, 0))
    draw_triangle(d, [(mid - 95, mid + 40), (mid - 25, mid + 40), (mid - 60, mid + 80)], outline=(0, 0, 0))
    draw_circle(d, (mid + 60, mid + 60), 18, outline=(0, 0, 0), width=4)
    return img

def rotate_image_simple(img: Image.Image, angle_deg: int) -> Image.Image:
    return img.rotate(angle_deg, expand=True, fillcolor=(255, 255, 255))

def quadrant_rotation_question(seed: int = 0, difficulty: str = "سهل") -> Question:
    base = draw_quadrant_symbols(canvas_size=280, seed=seed)
    angle = RNG.choice([90, 180, 270])
    correct = rotate_image_simple(base, -angle)
    distract = [
        rotate_image_simple(base, -random.choice([a for a in [90, 180, 270] if a != angle])),
        rotate_image_simple(base, -random.choice([a for a in [90, 180, 270] if a != angle])),
        base.transpose(Image.FLIP_LEFT_RIGHT),
    ]
    options = [correct] + distract
    RNG.shuffle(options)
    correct_index = options.index(correct)

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(
        sys,
        f"بعد تدوير الشكل رُباعياً بمقدار {angle}°، أي البدائل يطابق الصورة؟ اكتب سطرًا واحدًا.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
    )
    expl = ollama_chat_or_fallback(
        sys,
        "الإجابة الصحيحة تحتفظ بترتيب الرموز مع تدويرها حول مركز المربع دون انعكاس.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
    )
    return Question(title=title, image=base, options=options, correct_index=correct_index, explanation=expl)

# ---------------------- Puzzle 3: 3D isometric cube rotation ------
def orientation_matrices() -> List[np.ndarray]:
    mats = []
    axes = [
        np.array([[1,0,0],[0,1,0],[0,0,1]]),
        np.array([[1,0,0],[0,0,1],[0,-1,0]]),
        np.array([[1,0,0],[0,-1,0],[0,0,-1]]),
        np.array([[1,0,0],[0,0,-1],[0,1,0]]),
    ]
    Rz = lambda k: np.array([[math.cos(k), -math.sin(k), 0],
                             [math.sin(k),  math.cos(k), 0],
                             [0, 0, 1]])
    for base in axes:
        for k in [0, math.pi/2, math.pi, 3*math.pi/2]:
            mats.append(Rz(k) @ base)
    uniq, seen = [], set()
    for m in mats:
        key = tuple(np.round(m, 3).flatten().tolist())
        if key not in seen:
            seen.add(key); uniq.append(m)
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
    minx = min(u for u, _ in us); maxx = max(u for u, _ in us)
    miny = min(v for _, v in us); maxy = max(v for _, v in us)
    cx = (img_size[0] - (maxx - minx)) / 2 - minx
    cy = (img_size[1] - (maxy - miny)) / 2 - miny
    order = sorted(range(len(coords)), key=lambda i: sum(coords[i]))
    for i in order:
        x, y, z = coords[i]
        u, v = iso_project((x, y, z), size); u += cx; v += cy
        top = [(u, v - size), (u + size * 0.5, v - size * 0.5), (u, v), (u - size * 0.5, v - size * 0.5)]
        right = [(u, v), (u + size * 0.5, v - size * 0.5), (u + size * 0.5, v + size * 0.5), (u, v + size)]
        left  = [(u, v), (u - size * 0.5, v - size * 0.5), (u - size * 0.5, v + size * 0.5), (u, v + size)]
        d.polygon(top,   fill=(220, 220, 220), outline=(0, 0, 0))
        d.polygon(right, fill=(190, 190, 190), outline=(0, 0, 0))
        d.polygon(left,  fill=(160, 160, 160), outline=(0, 0, 0))
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

def cubes_rotation_question(seed: int = 0, difficulty: str = "سهل") -> Question:
    RNG.seed(seed)
    n = 4 if difficulty == "سهل" else 5
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

    opts = []
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
        "انظر إلى المجسّم على اليسار. أيُّ بديل في الأسفل يطابقه بعد تدويره كما في السهم؟ سطر واحد.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
    )
    expl = ollama_chat_or_fallback(
        sys,
        "المطابقة تعتمد على تطابق مواقع المكعّبات بعد تدوير الجسم دون انعكاس أو تبديل للاتصال.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
    )
    return Question(title=title, image=stmt, options=opts, correct_index=correct_index, explanation=expl)

# ---------------------- Puzzle 4: Shape assembly -------------------
def shape_assembly_question(seed: int = 0) -> Question:
    RNG.seed(seed)
    target = RNG.choice(["square", "pentagon"])
    W, H = 900, 380
    img = new_canvas(W, H)
    d = ImageDraw.Draw(img)

    if target == "square":
        draw_square(d, (W - 300, 80), 180)
        title_hint = "أي مجموعة من القطع يمكن أن تُكوِّن مربعًا كهذا؟"

        def tri_img(angle=0, flip=False):
            S = 140; pad = 10
            im = new_canvas(S + pad * 2, S + pad * 2)
            dr = ImageDraw.Draw(im)
            pts = [(pad+10, pad+10), (pad+S-10, pad+10), (pad+10, pad+S-10)]
            if flip:
                pts = [(pad+10, pad+10), (pad+S-10, pad+10), (pad+S-10, pad+S-10)]
            draw_triangle(dr, pts, outline=(0, 0, 0))
            return im.rotate(angle, expand=True, fillcolor=(255,255,255))

        def panel(imgs: List[Image.Image]) -> Image.Image:
            w = sum(i.width for i in imgs) + 20*(len(imgs)-1) + 40
            h = max(i.height for i in imgs) + 40
            out = new_canvas(w, h)
            x = 20
            for im in imgs:
                out.paste(im, (x, (h - im.height)//2)); x += im.width + 20
            return out

        optA = panel([tri_img(), tri_img(angle=180, flip=True)])
        rect = new_canvas(100, 140); ImageDraw.Draw(rect).rectangle([10,10,90,130], outline=(0,0,0), width=3)
        rhombus = new_canvas(120, 120); draw_diamond(ImageDraw.Draw(rhombus), (60,60), 80, outline=(0,0,0))
        optB = panel([new_canvas(10,10), rhombus])
        small_tri = tri_img().resize((120,120))
        optC = panel([small_tri, rhombus.rotate(45, expand=True, fillcolor=(255,255,255))])
        optD = panel([tri_img(), rect])
        options = [optA, optB, optC, optD]
        correct_index = 0

    else:
        cx, cy, r = W - 210, 170, 110
        pts = [(cx + r*math.cos(a), cy + r*math.sin(a)) for a in np.linspace(-math.pi/2, 1.5*math.pi, 6)[:-1]]
        d.polygon(pts, outline=(0,0,0), fill=None)
        title_hint = "أي مجموعة من القطع يمكن أن تُكوِّن خماسي الأضلاع؟"

        def quad_panel(lean=1) -> Image.Image:
            im = new_canvas(160, 140); dr = ImageDraw.Draw(im)
            p = [(20,120),(80,20),(140,60),(100,120)]
            if lean < 0:
                p = [(20,60),(60,20),(140,80),(100,120)]
            dr.polygon(p, outline=(0,0,0), fill=None)
            return im

        def panel(imgs: List[Image.Image]) -> Image.Image:
            w = sum(i.width for i in imgs) + 20*(len(imgs)-1) + 40
            h = max(i.height for i in imgs) + 40
            out = new_canvas(w, h)
            x = 20
            for im in imgs:
                out.paste(im, (x, (h - im.height)//2)); x += im.width + 20
            return out

        optA = quad_panel(1)
        optB = quad_panel(-1)
        optC = new_canvas(160,140); ImageDraw.Draw(optC).ellipse([30,30,130,110], outline=(0,0,0), width=3)
        optD = new_canvas(160,140); draw_triangle(ImageDraw.Draw(optD), [(20,120),(80,20),(140,120)], outline=(0,0,0))
        options = [panel([optA, optB]), panel([optA, optC]), panel([optB, optD]), panel([optC, optD])]
        correct_index = 0

    d.text((60, 40), "السؤال:", fill=(0,0,0))
    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat_or_fallback(sys, title_hint, model=st.session_state.get("llm_model", "qwen2.5:3b"))
    expl = ollama_chat_or_fallback(
        sys,
        "نقارن مساحات القطع واتجاه الحواف؛ المجموعة الصحيحة يمكن ترتيبها لتكمل حدود الشكل دون فجوات.",
        model=st.session_state.get("llm_model", "qwen2.5:3b"),
    )
    return Question(title=title, image=img, options=options, correct_index=correct_index, explanation=expl)

# ---------------------- Streamlit UI (Page) -----------------------
def run_spatial_app():
    # Sidebar (logo and settings)
    with st.sidebar:
        # If the logo is at repo root, this reference still works from pages/
        try:
            st.image("MOL_logo.png")
        except Exception:
            pass
        st.markdown("### الإعدادات")
        n_q = st.number_input("عدد الأسئلة", 1, 20, 4)
        seed_base = st.number_input("البذرة (Seed)", 0, 10_000, 42)
        difficulty = st.selectbox("مستوى الصعوبة", ["سهل", "متوسط"])
        st.session_state["llm_model"] = st.text_input("نموذج Ollama (اختياري)", value="qwen2.5:3b")
        use_llm = st.checkbox("استخدم LLM لكتابة التعليمات/الشرح", value=True)
        st.caption("إن لم يتوفر Ollama محليًا سيستخدم نصًا افتراضيًا.")
        puzzle_types = st.multiselect(
            "أنواع الأسئلة",
            ["طيّ الورق", "تدوير رباعي", "تدوير مكعّبات ثلاثي", "تركيب شكل"],
            default=["طيّ الورق", "تدوير رباعي", "تدوير مكعّبات ثلاثي", "تركيب شكل"],
        )
        gen = st.button("إنشاء الأسئلة", use_container_width=True)

    st.title("مولّد أسئلة ذكاء مرئية (مطابقة لنمط الأمثلة المرفقة)")
    st.write(
        "اختر النوع وعدد الأسئلة من الشريط الجانبي. جميع الرسومات تُنشأ برمجياً. "
        "يمكنك تفعيل LLM عبر **Ollama** لإنتاج تعليمات وشرح بالعربية تلقائيًا."
    )

    def build_by_type(kind: str, seed: int) -> Question:
        if kind == "طيّ الورق":
            return paper_fold_question(seed=seed, difficulty=difficulty)
        if kind == "تدوير رباعي":
            return quadrant_rotation_question(seed=seed, difficulty=difficulty)
        if kind == "تدوير مكعّبات ثلاثي":
            return cubes_rotation_question(seed=seed, difficulty=difficulty)
        if kind == "تركيب شكل":
            return shape_assembly_question(seed=seed)
        raise ValueError("نوع غير معروف")

    if gen:
        RNG.seed(seed_base)
        generated: List[Question] = []
        order: List[str] = []
        while len(order) < n_q:
            order.extend(puzzle_types)
        order = order[:n_q]

        for i, kind in enumerate(order):
            q = build_by_type(kind, seed=seed_base + i * 7 + RNG.randint(0, 999))
            generated.append(q)

        answers_csv = io.StringIO()
        zip_buf = io.BytesIO()

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, q in enumerate(generated, 1):
                st.markdown(f"### سؤال {idx}: {q.title}")
                st.image(q.image, use_container_width=True)

                cols = st.columns(4)
                for i, (c, col) in enumerate(zip(q.options, cols)):
                    col.image(c, use_container_width=True)
                    col.markdown(
                        f"<div style='text-align:center;font-weight:bold;font-size:20px'>{AR_LETTERS[i]}</div>",
                        unsafe_allow_html=True,
                    )
                    zf.writestr(f"q{idx}_opt_{i+1}.png", bytes_from_img(c))

                zf.writestr(f"q{idx}_statement.png", bytes_from_img(q.image))
                answers_csv.write(f"{idx},{AR_LETTERS[q.correct_index]}\n")

                with st.expander("إظهار الحل/الشرح"):
                    st.markdown(f"**الإجابة الصحيحة:** {AR_LETTERS[q.correct_index]}")
                    st.write(q.explanation if use_llm else "الصحيح الوحيد يوافق قاعدة السؤال (انعكاس/تدوير/تجميع).")

                st.divider()

            zf.writestr("answers.csv", answers_csv.getvalue().encode("utf-8"))

        st.download_button(
            "تنزيل كل الأسئلة (ZIP)",
            data=zip_buf.getvalue(),
            file_name="arabic_visual_iq_questions.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.info("اضغط **إنشاء الأسئلة** لبدء التوليد.")

# --------- Render the page on import (no set_page_config here) ----
run_spatial_app()
