# -*- coding: utf-8 -*-
# File: pages/7_spacial_questions.py
# Title: Spatial IQ Generator (Folding Challenge + Mimic Sample)
# Notes:
#   - ASCII-only, Pillow 10+ safe, Streamlit use_container_width
#   - Modes: Folding Challenge (with diagonal folds), Mimic Sample (Simple)
#   - Brand Palette + Style Presets + Batch ZIP

import io
import os
import json
import math
import time
import random
import zipfile
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from urllib import request as urlrequest
from urllib.error import URLError

# ----------------------------
# Utilities
# ----------------------------

def make_rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        seed = int(time.time() * 1000) % 2147483647
    return random.Random(seed)

def _load_font(font_size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        try:
            return ImageFont.truetype("Arial.ttf", font_size)
        except Exception:
            return ImageFont.load_default()

def text_image(text: str, size: Tuple[int, int] = (380, 380), font_size: int = 42,
               color=(30, 30, 30), bg=(255, 255, 255)) -> Image.Image:
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    font = _load_font(font_size)
    try:
        bbox = d.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (size[0] - w) / 2 - bbox[0]
        y = (size[1] - h) / 2 - bbox[1]
    except Exception:
        try:
            w = int(d.textlength(text, font=font))
        except Exception:
            w = int(len(text) * font_size * 0.6)
        h = int(font_size * 1.2)
        x = (size[0] - w) / 2
        y = (size[1] - h) / 2
    d.text((x, y), text, fill=color, font=font)
    return img

def _draw_polygon(draw, pts, fill, outline, width):
    draw.polygon(pts, fill=fill)
    pts_closed = list(pts) + [pts[0]]
    try:
        draw.line(pts_closed, fill=outline, width=width, joint="curve")
    except TypeError:
        draw.line(pts_closed, fill=outline, width=width)

def _rotate_points(points, angle_deg, origin):
    angle = math.radians(angle_deg)
    ox, oy = origin
    out = []
    for x, y in points:
        qx = ox + math.cos(angle) * (x - ox) - math.sin(angle) * (y - oy)
        qy = oy + math.sin(angle) * (x - ox) + math.cos(angle) * (y - oy)
        out.append((qx, qy))
    return out

def draw_shape(draw, shape: str, center: Tuple[int, int], size: int,
               rotation_deg: float = 0, fill=(20, 20, 20), outline=(20, 20, 20), width: int = 4):
    cx, cy = center
    r = size // 2
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)
    elif shape == "square":
        pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        pts = _rotate_points(pts, rotation_deg, (cx, cy))
        _draw_polygon(draw, pts, fill=fill, outline=outline, width=width)
    elif shape == "triangle":
        pts = [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        pts = _rotate_points(pts, rotation_deg, (cx, cy))
        _draw_polygon(draw, pts, fill=fill, outline=outline, width=width)
    elif shape == "pentagon":
        pts = []
        for k in range(5):
            ang = 90 + k * 72
            pts.append((cx + r * math.cos(math.radians(ang)), cy + r * math.sin(math.radians(ang))))
        pts = _rotate_points(pts, rotation_deg, (cx, cy))
        _draw_polygon(draw, pts, fill=fill, outline=outline, width=width)
    else:
        draw.line([cx - r, cy, cx + r, cy], fill=outline, width=width)
        draw.line([cx, cy - r, cx, cy + r], fill=outline, width=width)

def compose_grid(images: List[Image.Image], grid_size: Tuple[int, int], pad: int = 16, bg=(255, 255, 255)) -> Image.Image:
    rows, cols = grid_size
    assert len(images) == rows * cols, "compose_grid: image count must equal rows*cols"
    w, h = images[0].size
    for im in images:
        if im.size != (w, h):
            raise ValueError(f"compose_grid: tile size mismatch {im.size} vs {(w, h)}")
    W = cols * w + (cols + 1) * pad
    H = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    for r in range(rows):
        for c in range(cols):
            x = pad + c * (w + pad)
            y = pad + r * (h + pad)
            canvas.paste(images[r * cols + c], (x, y))
    return canvas

# ----------------------------
# Presets (JSON + session)
# ----------------------------

PRESETS_FILE = "style_presets.json"

def _load_presets() -> Dict[str, Dict]:
    if "style_presets" in st.session_state:
        return st.session_state.style_presets
    data = {}
    try:
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception:
        data = {}
    st.session_state.style_presets = data
    return data

def _save_presets(data: Dict[str, Dict]):
    st.session_state.style_presets = data
    try:
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# ----------------------------
# Style extraction (Mimic Sample)
# ----------------------------

def extract_style_from_image(img: Image.Image, n_colors: int = 6) -> Dict:
    small = img.convert("RGB")
    if max(small.size) > 512:
        scale = 512 / max(small.size)
        small = small.resize((int(small.width * scale), int(small.height * scale)), Image.BILINEAR)
    pal = small.quantize(colors=n_colors, method=Image.MEDIANCUT)
    palette = pal.getpalette()[:n_colors * 3]
    counts = pal.getcolors()
    colors_freq = []
    if counts:
        for count, idx in counts:
            rgb = tuple(palette[idx * 3: idx * 3 + 3])
            colors_freq.append((count, rgb))
        colors_freq.sort(key=lambda x: x[0], reverse=True)
    bg = colors_freq[0][1] if colors_freq else (255, 255, 255)
    def lum(c): return 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    unique = [rgb for _, rgb in colors_freq] or [(30, 30, 30), (240, 240, 240)]
    outline = min(unique, key=lum)
    fills = [c for c in unique if c != bg] or [(30, 30, 30), (0, 88, 155), (200, 0, 0)]
    stroke_width = 5 if img.width < 800 else (6 if img.width < 1400 else 8)
    text_color = (20, 20, 20) if lum(bg) > 186 else (240, 240, 240)
    paper_fill = tuple(min(255, int(bg[i] * 1.02)) for i in range(3))
    return {
        "bg": bg,
        "outline": outline,
        "fills": fills,
        "stroke_width": stroke_width,
        "text_color": text_color,
        "paper_fill": paper_fill
    }

# ----------------------------
# Data model
# ----------------------------

@dataclass
class ChoiceItem:
    label: str
    is_correct: bool
    image_filename: str

@dataclass
class QuestionPackage:
    id: str
    type: str
    difficulty: str
    seed: int
    prompt: str
    rule_description: str
    llm_explanation: Optional[str]
    correct_label: str
    choices: List[ChoiceItem]
    meta: Dict

# ----------------------------
# Matrix generator (Mimic Sample)
# ----------------------------

DEFAULT_COLORS = [
    (30, 30, 30),
    (0, 88, 155),
    (200, 0, 0),
    (0, 140, 70),
    (220, 120, 0),
    (120, 0, 160),
]
DEFAULT_SHAPES = ["circle", "square", "triangle", "pentagon"]

def generate_matrix_reasoning(rng: random.Random, img_size=(380, 380), cell_shape_size=160,
                              difficulty="Medium", style: Optional[Dict] = None):
    fills = (style.get("fills") if style else None) or DEFAULT_COLORS
    outline = (style.get("outline") if style else None) or (10, 10, 10)
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    stroke = (style.get("stroke_width") if style else None) or 4
    txt_col = (style.get("text_color") if style else None) or (30, 30, 30)

    if difficulty == "Easy":
        rules_to_use = 1
        rotation_step_choices = [0, 45, 90]
    elif difficulty == "Hard":
        rules_to_use = 3
        rotation_step_choices = [30, 45, 60, 90]
    else:
        rules_to_use = 2
        rotation_step_choices = [30, 45, 60, 90]

    use_rotation = use_count = use_color = use_size = False
    for s in rng.sample(["rotation", "count", "color", "size"], rules_to_use):
        if s == "rotation": use_rotation = True
        elif s == "count":  use_count = True
        elif s == "color":  use_color = True
        elif s == "size":   use_size = True

    rotation_step = rng.choice(rotation_step_choices) if use_rotation else 0
    base_count = rng.randint(1, 2) if use_count else 1
    base_size = cell_shape_size
    size_step = rng.choice([-20, 20]) if use_size else 0
    base_color_idx = rng.randrange(len(fills)) if use_color else 0
    color_by_row = (rng.random() < 0.5) if use_color else True
    shape = rng.choice(DEFAULT_SHAPES)

    grid_imgs = []
    rule_desc_parts = []
    for r in range(3):
        for c in range(3):
            if r == 2 and c == 2:
                grid_imgs.append(text_image("?", size=img_size, font_size=int(img_size[0] * 0.32),
                                            color=txt_col, bg=bg))
                continue
            img = Image.new("RGB", img_size, bg)
            d = ImageDraw.Draw(img)
            rot = (c * rotation_step) % 360 if use_rotation else 0
            count = base_count + r if use_count else 1
            size_px = max(50, base_size + r * size_step) if use_size else base_size
            if use_color:
                shift = r if color_by_row else c
                color_idx = (base_color_idx + shift) % len(fills)
            else:
                color_idx = base_color_idx
            cols_cnt = int(math.ceil(math.sqrt(count)))
            rows_cnt = int(math.ceil(count / cols_cnt))
            grid_w, grid_h = img_size
            margin = 30
            cell_w = (grid_w - 2 * margin) // cols_cnt
            cell_h = (grid_h - 2 * margin) // rows_cnt
            mini_size = min(size_px, int(0.8 * min(cell_w, cell_h)))
            k = 0
            for rr in range(rows_cnt):
                for cc in range(cols_cnt):
                    if k >= count: break
                    cx = margin + cc * cell_w + cell_w // 2
                    cy = margin + rr * cell_h + cell_h // 2
                    draw_shape(d, shape, (cx, cy), mini_size, rotation_deg=rot,
                               fill=fills[color_idx], outline=outline, width=stroke)
                    k += 1
            grid_imgs.append(img)

    correct_rot = (2 * rotation_step) % 360 if use_rotation else 0
    correct_count = base_count + 2 if use_count else 1
    correct_size = max(50, base_size + 2 * size_step) if use_size else base_size
    if use_color:
        shift = 2 if color_by_row else 2
        color_idx_correct = (base_color_idx + shift) % len(fills)
    else:
        color_idx_correct = base_color_idx

    def render_multi(count, rot, sz, color_idx):
        img = Image.new("RGB", img_size, bg)
        d = ImageDraw.Draw(img)
        cols_cnt = int(math.ceil(math.sqrt(count)))
        rows_cnt = int(math.ceil(count / cols_cnt))
        grid_w, grid_h = img_size
        margin = 30
        cell_w = (grid_w - 2 * margin) // cols_cnt
        cell_h = (grid_h - 2 * margin) // rows_cnt
        mini_size2 = min(sz, int(0.8 * min(cell_w, cell_h)))
        k2 = 0
        for rr in range(rows_cnt):
            for cc in range(cols_cnt):
                if k2 >= count: break
                cx = margin + cc * cell_w + cell_w // 2
                cy = margin + rr * cell_h + cell_h // 2
                draw_shape(d, shape, (cx, cy), mini_size2, rotation_deg=rot,
                           fill=fills[color_idx], outline=outline, width=stroke)
                k2 += 1
        return img

    correct_img = render_multi(correct_count, correct_rot, correct_size, color_idx_correct)

    def make_distractor(var: str):
        cnt, rot, sz, col = correct_count, correct_rot, correct_size, color_idx_correct
        if var == "rotation":
            step = rotation_step if rotation_step != 0 else rng.choice([30, 45, 60, 90])
            rot = (rot + rng.choice([-step, step])) % 360
        elif var == "count":
            cnt = max(1, cnt + rng.choice([-1, 1]))
        elif var == "size":
            sz = max(50, sz + rng.choice([-20, 20]))
        elif var == "color":
            col = (col + rng.choice([1, -1])) % len(fills)
        return render_multi(cnt, rot, sz, col)

    distractor_kinds = []
    if use_rotation: distractor_kinds.append("rotation")
    if use_count:    distractor_kinds.append("count")
    if use_size:     distractor_kinds.append("size")
    if use_color:    distractor_kinds.append("color")
    while len(distractor_kinds) < 3:
        distractor_kinds.append(rng.choice(["rotation", "count", "size", "color"]))

    choices_imgs = [correct_img] + [make_distractor(k) for k in distractor_kinds]
    rng.shuffle(choices_imgs)

    rule_desc_parts = []
    if use_rotation: rule_desc_parts.append(f"Rotation increases by {rotation_step} deg across columns.")
    if use_count:    rule_desc_parts.append("Number of shapes increases by 1 down the rows.")
    if use_color:    rule_desc_parts.append("Color alternates consistently.")
    if use_size:     rule_desc_parts.append(f"Shape size changes by {size_step:+d} px per row.")
    rule_desc = " ".join(rule_desc_parts) if rule_desc_parts else "Follow the visual pattern."
    prompt_text = "Which option completes the 3x3 matrix?"

    return {
        "grid_imgs": grid_imgs,
        "grid_size": (3, 3),
        "choices_imgs": choices_imgs,
        "correct_index": choices_imgs.index(correct_img),
        "rule_desc": rule_desc,
        "prompt": prompt_text,
        "meta": {"type": "matrix", "style_used": bool(style)}
    }

# ----------------------------
# Folding Challenge (with diagonal folds)
# ----------------------------

def reflect_point(p, axis):
    x, y = p
    if axis == "V":
        return (-x, y)
    if axis == "H":
        return (x, -y)
    if axis == "D1":
        return (y, x)
    if axis == "D2":
        return (-y, -x)
    return (x, y)

def unfold_points(base_point: Tuple[float, float], folds_axes: List[str]) -> List[Tuple[float, float]]:
    pts = [base_point]
    for axis in reversed(folds_axes):
        new_pts = []
        for p in pts:
            q = reflect_point(p, axis)
            new_pts.append(q)
        pts = pts + new_pts
    uniq = []
    seen = set()
    for x, y in pts:
        key = (round(x, 4), round(y, 4))
        if key not in seen:
            uniq.append((x, y))
            seen.add(key)
    return uniq

def draw_paper_with_holes(size=(420, 420), holes: List[Tuple[float, float]],
                          paper_margin=40, hole_radius=12,
                          bg=(255, 255, 255), paper_fill=(250, 250, 250),
                          outline=(20, 20, 20), stroke=5) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    left = paper_margin
    top = paper_margin
    right = W - paper_margin
    bottom = H - paper_margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    for (x, y) in holes:
        px = (x + 1) / 2.0 * (right - left) + left
        py = (1 - (y + 1) / 2.0) * (bottom - top) + top
        d.ellipse([px - hole_radius, py - hole_radius, px + hole_radius, py + hole_radius],
                  fill=outline, outline=outline, width=1)
    return img

def _arrow(d: ImageDraw.ImageDraw, start, end, color, width):
    d.line([start, end], fill=color, width=width)
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    head_len = max(14, width * 3)
    head_w = max(10, width * 2)
    tip = (end[0], end[1])
    base = (end[0] - ux * head_len, end[1] - uy * head_len)
    p1 = (base[0] + px * head_w, base[1] + py * head_w)
    p2 = (base[0] - px * head_w, base[1] - py * head_w)
    d.polygon([tip, p1, p2], fill=color)

def draw_fold_icon(direction: str, size=(180, 180),
                   bg=(255, 255, 255), paper_fill=(250, 250, 250),
                   outline=(20, 20, 20), stroke=4) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    margin = 18
    left = margin
    top = margin
    right = W - margin
    bottom = H - margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    cx, cy = W // 2, H // 2
    Llen = min(W, H) // 3
    if direction == "L":
        start, end = (cx + Llen // 2, cy), (cx - Llen, cy)
    elif direction == "R":
        start, end = (cx - Llen // 2, cy), (cx + Llen, cy)
    elif direction == "U":
        start, end = (cx, cy + Llen // 2), (cx, cy - Llen)
    elif direction == "D":
        start, end = (cx, cy - Llen // 2), (cx, cy + Llen)
    elif direction == "UL":
        start, end = (cx + Llen // 2, cy + Llen // 2), (cx - Llen, cy - Llen)
    elif direction == "UR":
        start, end = (cx - Llen // 2, cy + Llen // 2), (cx + Llen, cy - Llen)
    elif direction == "DL":
        start, end = (cx + Llen // 2, cy - Llen // 2), (cx - Llen, cy + Llen)
    else:
        start, end = (cx - Llen // 2, cy - Llen // 2), (cx + Llen, cy + Llen)
    _arrow(d, start, end, outline, stroke)
    return img

def draw_folded_with_punch(point_folded: Tuple[float, float], size=(220, 220),
                           bg=(255, 255, 255), paper_fill=(250, 250, 250),
                           outline=(20, 20, 20), stroke=4, text_color=(20, 20, 20)) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    margin = 16
    left = margin
    top = margin
    right = W - margin
    bottom = H - margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    x, y = point_folded
    px = (x + 1) / 2.0 * (right - left) + left
    py = (1 - (y + 1) / 2.0) * (bottom - top) + top
    r = 10
    d.ellipse([px - r, py - r, px + r, py + r], fill=outline, outline=outline)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    d.text((left, top - 22), "Punch", fill=text_color, font=font)
    return img

def generate_folding_challenge(rng: random.Random, difficulty="Medium",
                               allow_diagonal=True,
                               style: Optional[Dict] = None):
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    stroke = (style.get("stroke_width") if style else None) or 5
    text_color = (style.get("text_color") if style else None) or (20, 20, 20)

    if difficulty == "Easy":
        n_folds = rng.choice([1, 2])
    elif difficulty == "Hard":
        n_folds = 3
    else:
        n_folds = rng.choice([2, 3])

    dirs_card = ["L", "R", "U", "D"]
    dirs_diag = ["UL", "UR", "DL", "DR"]
    dirs_all = dirs_card + (dirs_diag if allow_diagonal else [])

    folds = []
    for i in range(n_folds):
        cand = rng.choice(dirs_all) if i == 0 else rng.choice([d for d in dirs_all if d != folds[-1]])
        folds.append(cand)

    def dir_to_axis(d):
        if d in ("L", "R"): return "V"
        if d in ("U", "D"): return "H"
        if d in ("UL", "DR"): return "D1"
        return "D2"

    axes = [dir_to_axis(d) for d in folds]

    px = rng.uniform(-0.65, 0.65)
    py = rng.uniform(-0.65, 0.65)
    point_folded = (px, py)

    holes = unfold_points(point_folded, axes)

    fold_icons = [draw_fold_icon(d, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke) for d in folds]
    row1 = compose_grid(fold_icons, (1, len(fold_icons))) if len(fold_icons) > 0 else text_image("No folds", bg=bg, color=outline)
    punch_panel = draw_folded_with_punch(point_folded, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke, text_color=text_color)

    W = max(row1.size[0], punch_panel.size[0]) + 32
    H = row1.size[1] + punch_panel.size[1] + 48
    problem_img = Image.new("RGB", (W, H), bg)
    x1 = (W - row1.size[0]) // 2
    problem_img.paste(row1, (x1, 16))
    x2 = (W - punch_panel.size[0]) // 2
    problem_img.paste(punch_panel, (x2, row1.size[1] + 32))

    correct_img = draw_paper_with_holes(holes=holes, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    if len(axes) > 0:
        holes_d1 = unfold_points(point_folded, axes[:-1])
    else:
        holes_d1 = [(px, py)]
    d1 = draw_paper_with_holes(holes=holes_d1, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    if len(axes) > 0:
        wrong_axes = axes.copy()
        idx = rng.randrange(len(wrong_axes))
        wrong_axes[idx] = "H" if wrong_axes[idx] == "V" else ("V" if wrong_axes[idx] == "H" else ("D2" if wrong_axes[idx] == "D1" else "D1"))
        holes_d2 = unfold_points(point_folded, wrong_axes)
    else:
        holes_d2 = [(-px, py)]
    d2 = draw_paper_with_holes(holes=holes_d2, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    d3 = correct_img.rotate(90, expand=False)

    choices = [correct_img, d1, d2, d3]
    rng.shuffle(choices)
    correct_index = choices.index(correct_img)

    rule_desc = "Unfold symmetrically across each fold line. Each fold mirrors the hole positions; the final pattern is the union of all mirrored holes."

    return {
        "problem_img": problem_img,
        "choices_imgs": choices,
        "correct_index": correct_index,
        "prompt": "Paper is folded in the shown order. A hole is punched as marked. Which option shows the fully unfolded paper?",
        "rule_desc": rule_desc,
        "meta": {
            "type": "folding",
            "folds": folds,
            "axes": axes,
            "punch": (round(px,3), round(py,3))
        }
    }

# ----------------------------
# Optional LLM (Ollama)
# ----------------------------

def ollama_chat(endpoint: str, model: str, prompt: str, temperature: float = 0.2, timeout: int = 30) -> Optional[str]:
    url = endpoint.rstrip("/") + "/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }).encode("utf-8")
    req = urlrequest.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            obj = json.loads(data.decode("utf-8"))
            return obj.get("response")
    except URLError:
        return None
    except Exception:
        return None

def build_llm_prompt(question_type: str, rule_desc: str, prompt_text: str,
                     choices_n: int, correct_label: str) -> str:
    labels = [chr(ord('A') + i) for i in range(choices_n)]
    return (
        "You are preparing concise explanations for spatial IQ questions.\n"
        "Provide ONLY a short 2-4 sentence explanation and the final answer label.\n"
        "Do NOT reveal chain-of-thought; just state the rule and why the answer fits.\n"
        f"Question type: {question_type}\n"
        f"Prompt: {prompt_text}\n"
        f"Observed rule: {rule_desc}\n"
        f"Choices: {', '.join(labels)}\n"
        f"Correct answer: {correct_label}\n"
        "Format:\n"
        "Answer: <LABEL>\n"
        "Explanation: <brief rationale>\n"
    )

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Spatial IQ Generator", layout="wide")
st.title("Spatial IQ Question Generator")

with st.sidebar:
    st.header("Mode")
    q_type = st.selectbox("Question type", ["Folding Challenge", "Mimic Sample (Simple)"], index=0)

    st.header("Difficulty and Seed")
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
    seed_input = st.text_input("Seed (optional)", value="", help="Leave empty for a new random seed.")

    st.markdown("---")
    st.header("Brand Palette")
    presets = _load_presets()
    preset_names = ["(none)"] + sorted(presets.keys())
    chosen_preset = st.selectbox("Load preset", preset_names, index=0)

    def _hex_default(key, default):
        return st.session_state.get(key, default)

    bg_hex = st.text_input("Background hex", value=_hex_default("bg_hex", "#FFFFFF"))
    paper_hex = st.text_input("Paper fill hex", value=_hex_default("paper_hex", "#FAFAFA"))
    outline_hex = st.text_input("Outline hex", value=_hex_default("outline_hex", "#141414"))
    text_hex = st.text_input("Text hex", value=_hex_default("text_hex", "#141414"))
    stroke_w = st.number_input("Stroke width", min_value=2, max_value=12, value=int(_hex_default("stroke_w", 5)), step=1)
    allow_diag = st.checkbox("Allow diagonal folds (folding mode)", value=True)

    if chosen_preset != "(none)":
        p = presets.get(chosen_preset, {})
        bg_hex = p.get("bg_hex", bg_hex)
        paper_hex = p.get("paper_hex", paper_hex)
        outline_hex = p.get("outline_hex", outline_hex)
        text_hex = p.get("text_hex", text_hex)
        stroke_w = int(p.get("stroke_w", stroke_w))

    st.session_state.bg_hex = bg_hex
    st.session_state.paper_hex = paper_hex
    st.session_state.outline_hex = outline_hex
    st.session_state.text_hex = text_hex
    st.session_state.stroke_w = stroke_w

    def _hex_to_rgb(h):
        try:
            h = h.strip().lstrip("#")
            if len(h) == 3:
                h = "".join([c*2 for c in h])
            if len(h) != 6:
                return (255, 255, 255)
            return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
        except Exception:
            return (255, 255, 255)

    st.caption("Tip: Save your palette as a preset.")
    new_preset_name = st.text_input("Preset name")
    colpr1, colpr2 = st.columns(2)
    with colpr1:
        if st.button("Save preset", use_container_width=True):
            if new_preset_name.strip():
                presets[new_preset_name.strip()] = {
                    "bg_hex": bg_hex, "paper_hex": paper_hex,
                    "outline_hex": outline_hex, "text_hex": text_hex,
                    "stroke_w": int(stroke_w)
                }
                _save_presets(presets)
                st.success(f"Saved preset: {new_preset_name.strip()}")
            else:
                st.warning("Enter a preset name.")
    with colpr2:
        if chosen_preset != "(none)" and st.button("Delete preset", use_container_width=True):
            if chosen_preset in presets:
                del presets[chosen_preset]
                _save_presets(presets)
                st.success(f"Deleted preset: {chosen_preset}")

    sample_bytes = None
    if q_type.startswith("Mimic"):
        st.markdown("---")
        st.header("Sample for style (optional)")
        f = st.file_uploader("Upload a sample image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        if f:
            sample_bytes = f.getvalue()

    st.markdown("---")
    st.header("LLM Explanation (optional)")
    include_llm = st.checkbox("Use Ollama", value=False)
    ollama_host = st.text_input("Ollama endpoint", value=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ollama_model = st.text_input("Model", value=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown("---")
    st.header("Generate")
    gen_btn = st.button("Generate New Question", type="primary", use_container_width=True)

    st.subheader("Batch")
    batch_n = st.number_input("How many to generate", min_value=2, max_value=100, value=10, step=1)
    batch_btn = st.button("Generate Batch ZIP", use_container_width=True)

# Build style from palette
def _hex_to_rgb_inline(h):
    try:
        h = h.strip().lstrip("#")
        if len(h) == 3:
            h = "".join([c*2 for c in h])
        if len(h) != 6:
            return (255, 255, 255)
        return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
    except Exception:
        return (255, 255, 255)

style_palette = {
    "bg": _hex_to_rgb_inline(st.session_state.bg_hex),
    "paper_fill": _hex_to_rgb_inline(st.session_state.paper_hex),
    "outline": _hex_to_rgb_inline(st.session_state.outline_hex),
    "text_color": _hex_to_rgb_inline(st.session_state.text_hex),
    "stroke_width": int(st.session_state.stroke_w),
}

# Seed
seed = None
if seed_input.strip():
    try:
        seed = int(seed_input.strip())
    except Exception:
        seed = abs(hash(seed_input)) % (2**31)
rng = make_rng(seed)

# Helper: build one item
def _make_one(rng_local: random.Random):
    if q_type.startswith("Folding"):
        pack = generate_folding_challenge(rng_local, difficulty=difficulty, allow_diagonal=allow_diag, style=style_palette)
        q_id = f"folding-{int(time.time()*1000)}-{rng_local.randrange(1_000_000)}"
        problem_img = pack["problem_img"]
        choices_imgs = pack["choices_imgs"]
        correct_index = pack["correct_index"]
        prompt_text = pack["prompt"]
        rule_desc = pack["rule_desc"]
        meta = pack["meta"]
        qtype_meta = "folding"
    else:
        style = style_palette.copy()
        if sample_bytes:
            try:
                im = Image.open(io.BytesIO(sample_bytes)).convert("RGB")
                ext = extract_style_from_image(im)
                style["fills"] = ext.get("fills", DEFAULT_COLORS)
            except Exception:
                pass
        pack = generate_matrix_reasoning(rng_local, difficulty=difficulty, style=style)
        q_id = f"mimic-{int(time.time()*1000)}-{rng_local.randrange(1_000_000)}"
        problem_img = compose_grid(pack["grid_imgs"], pack["grid_size"])
        choices_imgs = pack["choices_imgs"]
        correct_index = pack["correct_index"]
        prompt_text = pack["prompt"]
        rule_desc = pack["rule_desc"]
        meta = pack["meta"]
        qtype_meta = "mimic"

    labels = [chr(ord('A') + i) for i in range(len(choices_imgs))]
    labeled_choices = []
    for i, img in enumerate(choices_imgs):
        overlay = img.copy()
        d = ImageDraw.Draw(overlay)
        box_w, box_h = 64, 54
        d.rectangle([10, 10, 10 + box_w, 10 + box_h], fill=(245, 245, 245), outline=(30, 30, 30), width=2)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 32)
        except Exception:
            font = ImageFont.load_default()
        label = labels[i]
        try:
            bbox = d.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                tw = int(d.textlength(label, font=font))
            except Exception:
                tw = int(len(label) * 20 * 0.6)
            th = 24
        tx = 10 + (box_w - tw) / 2
        ty = 10 + (box_h - th) / 2
        d.text((tx, ty), label, fill=(10, 10, 10), font=font)
        labeled_choices.append(overlay)

    llm_expl = None
    if include_llm:
        llm_prompt = build_llm_prompt(qtype_meta, rule_desc, prompt_text, len(labels), labels[correct_index])
        llm_expl = ollama_chat(ollama_host, ollama_model, llm_prompt, temperature=temperature)

    return {
        "id": q_id,
        "type": qtype_meta,
        "difficulty": difficulty,
        "seed": seed if seed is not None else 0,
        "prompt": prompt_text,
        "rule_description": rule_desc,
        "llm_explanation": llm_expl,
        "correct_label": labels[correct_index],
        "labels": labels,
        "correct_index": correct_index,
        "problem_img": problem_img,
        "choices_imgs": labeled_choices,
        "meta": meta
    }

# Generate now or on click
if ("qpack" not in st.session_state) or gen_btn:
    st.session_state.qpack = _make_one(rng)

# Render
qp = st.session_state.get("qpack")
if qp:
    colQ, colA = st.columns([2.1, 1.4])
    with colQ:
        st.subheader("Question")
        st.image(qp["problem_img"], use_container_width=True)
        st.write(qp["prompt"])
    with colA:
        st.subheader("Choices")
        chosen = st.radio("Select your answer:", qp["labels"], index=0, horizontal=True, label_visibility="collapsed")
        cols2 = st.columns(2)
        for i, img in enumerate(qp["choices_imgs"]):
            with cols2[i % 2]:
                st.image(img, caption=f"Option {qp['labels'][i]}", use_container_width=True)
        if st.button("Check answer"):
            if chosen == qp["correct_label"]:
                st.success(f"Correct. Answer: {qp['correct_label']}")
            else:
                st.error(f"Not quite. Correct answer: {qp['correct_label']}")
            st.markdown("Why: " + qp["rule_description"])
            if qp["llm_explanation"]:
                st.markdown("LLM explanation: " + qp["llm_explanation"])

    st.markdown("---")
    st.subheader("Export")
    filename = f"{qp['id']}.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pb = io.BytesIO()
        qp["problem_img"].save(pb, format="PNG")
        zf.writestr("problem.png", pb.getvalue())
        choice_items = []
        for i, img in enumerate(qp["choices_imgs"]):
            cfname = f"choice_{qp['labels'][i]}.png"
            cb = io.BytesIO()
            img.save(cb, format="PNG")
            zf.writestr(cfname, cb.getvalue())
            choice_items.append(ChoiceItem(label=qp["labels"][i], is_correct=(i == qp["correct_index"]), image_filename=cfname))
        qpkg = QuestionPackage(
            id=qp["id"],
            type=qp["type"],
            difficulty=qp["difficulty"],
            seed=qp["seed"],
            prompt=qp["prompt"],
            rule_description=qp["rule_description"],
            llm_explanation=qp["llm_explanation"],
            correct_label=qp["correct_label"],
            choices=choice_items,
            meta=qp["meta"]
        )
        zf.writestr("question.json", json.dumps(asdict(qpkg), indent=2))
    st.download_button("Download ZIP", data=buf.getvalue(), file_name=filename, mime="application/zip")

    if batch_btn:
        batch_buf = io.BytesIO()
        with zipfile.ZipFile(batch_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            master_meta = []
            for k in range(int(batch_n)):
                rng_local = make_rng((seed or 0) + k + 1)
                pack = _make_one(rng_local)
                pb = io.BytesIO()
                pack["problem_img"].save(pb, format="PNG")
                zf.writestr(f"{pack['id']}/problem.png", pb.getvalue())
                for i, img in enumerate(pack["choices_imgs"]):
                    cb = io.BytesIO()
                    img.save(cb, format="PNG")
                    zf.writestr(f"{pack['id']}/choice_{pack['labels'][i]}.png", cb.getvalue())
                choice_items = [
                    {"label": pack["labels"][i], "is_correct": (i == pack["correct_index"]), "image_filename": f"choice_{pack['labels'][i]}.png"}
                    for i in range(len(pack["choices_imgs"]))
                ]
                meta_one = {
                    "id": pack["id"],
                    "type": pack["type"],
                    "difficulty": pack["difficulty"],
                    "seed": pack["seed"],
                    "prompt": pack["prompt"],
                    "rule_description": pack["rule_description"],
                    "llm_explanation": pack["llm_explanation"],
                    "correct_label": pack["correct_label"],
                    "choices": choice_items,
                    "meta": pack["meta"]
                }
                zf.writestr(f"{pack['id']}/question.json", json.dumps(meta_one, indent=2))
                master_meta.append(meta_one)
            zf.writestr("index.json", json.dumps(master_meta, indent=2))
        st.download_button("Download Batch ZIP", data=batch_buf.getvalue(), file_name=f"batch_{qp['type']}_{int(time.time())}.zip", mime="application/zip")
