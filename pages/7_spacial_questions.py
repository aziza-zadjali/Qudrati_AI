# -*- coding: utf-8 -*-
# File: pages/7_spacial_questions.py
# Title: Spatial IQ Question Generator (Matrix, Rotation, Mirror, Sample, Mimic)
# Notes:
#   - NEW: "Mimic Sample (Procedural)" mode extracts palette/stroke from a sample image,
#           then generates NEW questions in that style.
#   - Sidebar Generate button + optional auto-generate on settings change.
#   - ASCII-only strings (no emojis).
#   - Pillow 10+ compatible (uses textbbox; no textsize).
#   - Consistent tile sizes; use_container_width everywhere.
#   - Export ZIP with images + JSON metadata.
#   - Optional LLM explanations via Ollama.

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
# Utilities and draw helpers
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

def _is_light(rgb):
    r, g, b = rgb
    # simple luminance heuristic
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y > 186

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

def _rotate_points(points, angle_deg, origin):
    angle = math.radians(angle_deg)
    ox, oy = origin
    out = []
    for x, y in points:
        qx = ox + math.cos(angle) * (x - ox) - math.sin(angle) * (y - oy)
        qy = oy + math.sin(angle) * (x - ox) + math.cos(angle) * (y - oy)
        out.append((qx, qy))
    return out

def _draw_polygon(draw, pts, fill, outline, width):
    draw.polygon(pts, fill=fill)
    pts_closed = list(pts) + [pts[0]]
    try:
        draw.line(pts_closed, fill=outline, width=width, joint="curve")
    except TypeError:
        draw.line(pts_closed, fill=outline, width=width)

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

# ------------ Style extraction (from sample image) ------------

def extract_style_from_image(img: Image.Image, n_colors: int = 6) -> Dict:
    """
    Extract a simple style profile from an image:
      - background: most frequent color
      - outline: darkest among palette
      - fills: other dominant colors (excluding bg), sorted by saturation/luma mix
      - stroke_width: heuristic based on image width
    Uses PIL adaptive quantization (no external deps).
    """
    # Downscale for robustness
    small = img.convert("RGB")
    if max(small.size) > 512:
        scale = 512 / max(small.size)
        small = small.resize((int(small.width * scale), int(small.height * scale)), Image.BILINEAR)

    # Quantize to n_colors
    pal_img = small.quantize(colors=n_colors, method=Image.MEDIANCUT)
    palette = pal_img.getpalette()[:n_colors * 3]
    counts = pal_img.getcolors()  # list of (count, index)

    # Map palette indices to RGB and sort by frequency desc
    colors_freq = []
    if counts:
        for count, idx in counts:
            rgb = tuple(palette[idx * 3: idx * 3 + 3])
            colors_freq.append((count, rgb))
        colors_freq.sort(key=lambda x: x[0], reverse=True)

    # Pick background as most common color
    bg = colors_freq[0][1] if colors_freq else (255, 255, 255)

    # Determine outline as darkest color among palette
    unique_rgbs = [rgb for _, rgb in colors_freq]
    if not unique_rgbs:
        unique_rgbs = [(30, 30, 30), (0, 0, 0), (200, 200, 200)]
    darkest = min(unique_rgbs, key=lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])

    # Fills = remaining colors excluding bg; sort by perceived saturation-ish
    def sat_like(c):
        r, g, b = c
        mx, mn = max(c), min(c)
        return (mx - mn, -(0.2126*r + 0.7152*g + 0.0722*b))  # prioritize spread, then darker
    fills = [c for c in unique_rgbs if c != bg]
    fills.sort(key=sat_like, reverse=True)

    # Ensure at least a few fills
    if len(fills) < 3:
        fills = fills + [(0, 88, 155), (200, 0, 0), (0, 140, 70)]
        seen = set()
        fills = [x for x in fills if not (x in seen or seen.add(x))]

    # Stroke width heuristic
    base_w = img.width
    stroke_width = 5 if base_w < 800 else (6 if base_w < 1400 else 8)

    # Ensure good text contrast
    text_color = (20, 20, 20) if _is_light(bg) else (240, 240, 240)

    return {
        "bg": bg,
        "outline": darkest,
        "fills": fills,
        "stroke_width": stroke_width,
        "text_color": text_color
    }


# ----------------------------
# Data model for export
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
# Puzzle generators (style-aware)
# ----------------------------

DEFAULT_SHAPES = ["circle", "square", "triangle", "pentagon"]
DEFAULT_COLORS = [
    (30, 30, 30),
    (0, 88, 155),
    (200, 0, 0),
    (0, 140, 70),
    (220, 120, 0),
    (120, 0, 160),
]

def generate_matrix_reasoning(rng: random.Random, img_size=(380, 380), cell_shape_size=160, difficulty="Medium",
                              style: Optional[Dict] = None, forced_rules: Optional[Dict] = None):
    """
    Generate a 3x3 matrix; bottom-right is missing.
    style: dict with keys bg, outline, fills(list), stroke_width.
    forced_rules: optional dict like {"rotation": True, "count": True, "color": False, "size": False,
                                      "rotation_step": 45, "size_step": 20, "base_count": 1}
    """
    fills = (style.get("fills") if style else None) or DEFAULT_COLORS
    outline_color = (style.get("outline") if style else None) or (10, 10, 10)
    bg_color = (style.get("bg") if style else None) or (255, 255, 255)
    stroke_w = (style.get("stroke_width") if style else None) or 4

    # Determine rules
    if forced_rules:
        use_rotation = bool(forced_rules.get("rotation", False))
        use_count = bool(forced_rules.get("count", False))
        use_color = bool(forced_rules.get("color", False))
        use_size = bool(forced_rules.get("size", False))
        # Parameters (fallbacks)
        rotation_step = int(forced_rules.get("rotation_step", 45)) if use_rotation else 0
        size_step = int(forced_rules.get("size_step", 20)) if use_size else 0
        base_count = int(forced_rules.get("base_count", 1)) if use_count else 1
    else:
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
            elif s == "count": use_count = True
            elif s == "color": use_color = True
            elif s == "size": use_size = True
        rotation_step = rng.choice(rotation_step_choices) if use_rotation else 0
        base_count = rng.randint(1, 2) if use_count else 1
        size_step = rng.choice([-20, 20]) if use_size else 0

    base_size = cell_shape_size
    base_color_idx = rng.randrange(len(fills)) if use_color else 0
    color_by_row = (rng.random() < 0.5) if use_color else True
    shape = rng.choice(DEFAULT_SHAPES)

    grid_imgs = []
    rule_desc_parts = []
    # Precompute text color for placeholder
    txt_col = (style.get("text_color") if style else None) or (30, 30, 30)

    for r in range(3):
        for c in range(3):
            if r == 2 and c == 2:
                grid_imgs.append(text_image("?", size=img_size, font_size=int(img_size[0] * 0.32),
                                            color=txt_col, bg=bg_color))
                continue

            img = Image.new("RGB", img_size, bg_color)
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
                               fill=fills[color_idx], outline=outline_color, width=stroke_w)
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
        img = Image.new("RGB", img_size, bg_color)
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
                           fill=fills[color_idx], outline=outline_color, width=stroke_w)
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

    if use_rotation: rule_desc_parts.append(f"Rotation increases by {rotation_step} deg across columns.")
    if use_count:    rule_desc_parts.append("Number of shapes increases by 1 down the rows.")
    if use_color:    rule_desc_parts.append(f"Color alternates by {'row' if color_by_row else 'column'}.")
    if use_size:     rule_desc_parts.append(f"Shape size changes by {size_step:+d} px per row.")
    rule_desc = " ".join(rule_desc_parts) if rule_desc_parts else "Follow the visual pattern."
    prompt_text = "Which option completes the 3x3 matrix?"

    return {
        "grid_imgs": grid_imgs,
        "grid_size": (3, 3),
        "choices_imgs": choices_imgs,
        "correct_index": choices_imgs.index(correct_img),
        "shape": shape,
        "rule_desc": rule_desc,
        "prompt": prompt_text,
        "meta": {
            "type": "matrix",
            "shape": shape,
            "use_rotation": use_rotation,
            "use_count": use_count,
            "use_color": use_color,
            "use_size": use_size,
            "rotation_step": rotation_step,
            "size_step": size_step,
            "base_count": base_count,
            "color_by_row": color_by_row,
            "style_used": bool(style)
        }
    }

def generate_rotation_sequence(rng: random.Random, img_size=(420, 420), shape_size=200, difficulty="Medium",
                               style: Optional[Dict] = None):
    fills = (style.get("fills") if style else None) or DEFAULT_COLORS
    outline_color = (style.get("outline") if style else None) or (10, 10, 10)
    bg_color = (style.get("bg") if style else None) or (255, 255, 255)
    stroke_w = (style.get("stroke_width") if style else None) or 5

    steps = 3 if difficulty == "Easy" else (4 if difficulty == "Medium" else 5)
    shape = random.choice(DEFAULT_SHAPES)
    step_angle = random.choice([30, 45, 60, 90])
    start_angle = random.choice([0, 15, 30, 45])
    color = random.choice(fills)

    seq_imgs = []
    for i in range(steps):
        img = Image.new("RGB", img_size, bg_color)
        d = ImageDraw.Draw(img)
        draw_shape(d, shape, (img_size[0] // 2, img_size[1] // 2), shape_size,
                   rotation_deg=(start_angle + i * step_angle) % 360,
                   fill=color, outline=outline_color, width=stroke_w)
        seq_imgs.append(img)

    correct_rot = (start_angle + steps * step_angle) % 360
    correct_img = Image.new("RGB", img_size, bg_color)
    d = ImageDraw.Draw(correct_img)
    draw_shape(d, shape, (img_size[0] // 2, img_size[1] // 2), shape_size,
               rotation_deg=correct_rot, fill=color, outline=outline_color, width=stroke_w)

    choices = [correct_img]
    for delta_mult in [1, -1, 2]:
        img = Image.new("RGB", img_size, bg_color)
        d = ImageDraw.Draw(img)
        wrong_rot = (correct_rot + delta_mult * step_angle) % 360
        draw_shape(d, shape, (img_size[0] // 2, img_size[1] // 2), shape_size,
                   rotation_deg=wrong_rot, fill=color, outline=outline_color, width=stroke_w)
        choices.append(img)
    random.shuffle(choices)

    rule_desc = f"The figure rotates by {step_angle} deg each step."
    prompt_text = "Select the next figure in the rotation sequence."

    return {
        "sequence_imgs": seq_imgs,
        "choices_imgs": choices,
        "correct_index": choices.index(correct_img),
        "rule_desc": rule_desc,
        "prompt": prompt_text,
        "meta": {
            "type": "rotation",
            "shape": shape,
            "step_angle": step_angle,
            "start_angle": start_angle,
            "steps_shown": steps,
            "style_used": bool(style)
        }
    }

def generate_mirror_choice(rng: random.Random, img_size=(420, 420), shape_size=220, difficulty="Medium",
                           style: Optional[Dict] = None):
    fills = (style.get("fills") if style else None) or DEFAULT_COLORS
    outline_color = (style.get("outline") if style else None) or (10, 10, 10)
    bg_color = (style.get("bg") if style else None) or (255, 255, 255)
    stroke_w = (style.get("stroke_width") if style else None) or 5

    shape = random.choice(DEFAULT_SHAPES)
    mirror_axis = random.choice(["horizontal", "vertical"])
    rotation_deg = random.choice([0, 15, 30, 45, 60, 75, 90])
    color = random.choice(fills)

    base = Image.new("RGB", img_size, bg_color)
    d = ImageDraw.Draw(base)
    draw_shape(d, shape, (img_size[0] // 2, img_size[1] // 2), shape_size,
               rotation_deg=rotation_deg, fill=color, outline=outline_color, width=stroke_w)

    correct = base.transpose(Image.FLIP_LEFT_RIGHT) if mirror_axis == "vertical" else base.transpose(Image.FLIP_TOP_BOTTOM)
    wrong1 = base.transpose(Image.FLIP_TOP_BOTTOM) if mirror_axis == "vertical" else base.transpose(Image.FLIP_LEFT_RIGHT)
    wrong2 = base.rotate(180)
    wrong3 = base.rotate(90)

    choices = [correct, wrong1, wrong2, wrong3]
    random.shuffle(choices)

    rule_desc = f"The correct answer is the {mirror_axis} mirror of the base figure."
    prompt_text = "Which option is the mirror image of the base figure?"

    return {
        "base_img": base,
        "choices_imgs": choices,
        "correct_index": choices.index(correct),
        "rule_desc": rule_desc,
        "prompt": prompt_text,
        "meta": {
            "type": "mirror",
            "shape": shape,
            "mirror_axis": mirror_axis,
            "rotation_deg": rotation_deg,
            "style_used": bool(style)
        }
    }


# ----------------------------
# Optional LLM via Ollama
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
    st.header("Settings")

    q_type = st.selectbox(
        "Question type",
        ["Matrix (3x3)", "Rotation Sequence", "Mirror Image", "Sample (Image)", "Mimic Sample (Procedural)"],
        index=0
    )
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
    seed_input = st.text_input("Seed (optional for reproducibility)", value="", help="Leave empty for a fresh random seed each time.")

    st.markdown("---")
    st.subheader("Output Options")
    show_guides = st.checkbox("Show section labels (Question, Choices)", value=True)
    include_llm = st.checkbox("Use open-source LLM for explanations (via Ollama)", value=False)
    ollama_host = st.text_input("Ollama endpoint", value=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ollama_model = st.text_input("Model", value=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown("---")
    st.subheader("Reference (optional)")
    ref_files = st.file_uploader("Upload sample question images (to display or use as a question)",
                                 type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # Sample (Image) simple settings
    sample_options_n = 4
    sample_prompt = "Answer the question shown in the image."
    sample_rule = "As shown in the image."
    if q_type.startswith("Sample"):
        st.markdown("---")
        st.subheader("Sample Settings")
        if ref_files:
            names = [f.name for f in ref_files]
            default_idx = 0
            if "sample_sel_name" in st.session_state and st.session_state.sample_sel_name in names:
                default_idx = names.index(st.session_state.sample_sel_name)
            sel_name = st.selectbox("Choose image", names, index=default_idx)
            st.session_state.sample_sel_name = sel_name
            for f in ref_files:
                if f.name == sel_name:
                    st.session_state.sample_image_bytes = f.getvalue()
                    break
        else:
            st.info("Upload the sample question image above to use it as the problem.")

        sample_options_n = st.number_input("Number of options", min_value=2, max_value=6, value=4, step=1)
        letters_tmp = [chr(ord('A') + i) for i in range(int(sample_options_n))]
        default_label = st.session_state.get("sample_correct_label", "A")
        if default_label not in letters_tmp:
            default_label = letters_tmp[0]
        sample_correct_label = st.selectbox("Correct label", letters_tmp, index=letters_tmp.index(default_label))
        st.session_state.sample_correct_label = sample_correct_label

        sample_prompt = st.text_input("Prompt", sample_prompt)
        sample_rule = st.text_input("Rule/Why", sample_rule)

    # Mimic Sample (Procedural) settings
    mimic_forced_rules = None
    mimic_use_style = False
    if q_type.startswith("Mimic"):
        st.markdown("---")
        st.subheader("Mimic Settings")
        if ref_files:
            names = [f.name for f in ref_files]
            default_idx = 0
            if "mimic_sel_name" in st.session_state and st.session_state.mimic_sel_name in names:
                default_idx = names.index(st.session_state.mimic_sel_name)
            sel_name2 = st.selectbox("Choose sample image for style", names, index=default_idx, key="mimic_image_select")
            st.session_state.mimic_sel_name = sel_name2
            for f in ref_files:
                if f.name == sel_name2:
                    st.session_state.mimic_image_bytes = f.getvalue()
                    break
            mimic_use_style = True
        else:
            st.info("Upload a sample image above. The generator will extract palette and stroke to mimic it.")

        st.caption("Optional: Force certain rules to match the sample's logic more closely.")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            r_rotation = st.checkbox("Rule: rotation", value=True)
            r_count = st.checkbox("Rule: count", value=True)
        with col_m2:
            r_color = st.checkbox("Rule: color alternation", value=False)
            r_size = st.checkbox("Rule: size change", value=False)

        # Advanced parameters (small, only shown if rule checked)
        rotation_step_val = st.selectbox("Rotation step (deg)", [0, 30, 45, 60, 90], index=2) if r_rotation else 0
        size_step_val = st.selectbox("Size step per row (px)", [-30, -20, -10, 10, 20, 30], index=4) if r_size else 0
        base_count_val = st.number_input("Base count (top row)", min_value=1, max_value=4, value=1, step=1) if r_count else 1

        mimic_forced_rules = {
            "rotation": r_rotation,
            "count": r_count,
            "color": r_color,
            "size": r_size,
            "rotation_step": rotation_step_val,
            "size_step": size_step_val,
            "base_count": base_count_val
        }

    st.markdown("---")
    st.subheader("Generate")
    auto_generate = st.checkbox("Auto-generate when settings change", value=False)
    trigger_sidebar = st.button("Generate New Question", type="primary", use_container_width=True)
    st.caption("Tip: Change settings above, then click Generate. Or enable auto-generate.")

# RNG init
seed = None
if seed_input.strip():
    try:
        seed = int(seed_input.strip())
    except Exception:
        seed = abs(hash(seed_input)) % (2**31)
rng = make_rng(seed)

# Detect settings changes for optional auto-generate
current_settings = {
    "q_type": q_type,
    "difficulty": difficulty,
    "seed_input": seed_input,
    "include_llm": include_llm,
    "ollama_host": ollama_host,
    "ollama_model": ollama_model,
    "temperature": temperature,
    # sample
    "sample_sel_name": st.session_state.get("sample_sel_name"),
    "sample_options_n": int(sample_options_n),
    "sample_correct_label": st.session_state.get("sample_correct_label", "A"),
    "sample_prompt": sample_prompt,
    "sample_rule": sample_rule,
    # mimic
    "mimic_sel_name": st.session_state.get("mimic_sel_name"),
    "mimic_forced_rules": mimic_forced_rules,
}
prev_settings = st.session_state.get("prev_settings")
settings_changed = (prev_settings is not None) and (prev_settings != current_settings)
st.session_state.prev_settings = current_settings

# Decide trigger
trigger = ("qpack" not in st.session_state) or trigger_sidebar or (auto_generate and settings_changed)

if trigger:
    # Default style (None) or extracted from sample for Mimic mode
    extracted_style = None
    if q_type.startswith("Mimic"):
        sample_bytes = st.session_state.get("mimic_image_bytes")
        if (sample_bytes is None) and ref_files:
            sample_bytes = ref_files[0].getvalue()
            st.session_state.mimic_sel_name = ref_files[0].name
            st.session_state.mimic_image_bytes = sample_bytes
        if sample_bytes:
            try:
                style_img = Image.open(io.BytesIO(sample_bytes)).convert("RGB")
                extracted_style = extract_style_from_image(style_img, n_colors=6)
            except Exception:
                extracted_style = None

    # Generate by mode
    if q_type.startswith("Matrix"):
        pack = generate_matrix_reasoning(rng, difficulty=difficulty)
        q_id = f"matrix-{int(time.time())}"
        problem_img = compose_grid(pack["grid_imgs"], pack["grid_size"])
        choices_imgs = pack["choices_imgs"]
        correct_index = pack["correct_index"]
        prompt_text = pack["prompt"]
        rule_desc = pack["rule_desc"]
        meta = pack["meta"]
        qtype_meta = "matrix"

    elif q_type.startswith("Rotation Sequence"):
        pack = generate_rotation_sequence(rng, difficulty=difficulty)
        q_id = f"rotation-{int(time.time())}"
        seq = pack["sequence_imgs"]
        tile_size = seq[0].size
        q_font = max(28, int(tile_size[0] * 0.28))
        # Use same bg as tiles
        problem_img = compose_grid(seq + [text_image("?", size=tile_size, font_size=q_font)], (1, len(seq) + 1))
        choices_imgs = pack["choices_imgs"]
        correct_index = pack["correct_index"]
        prompt_text = pack["prompt"]
        rule_desc = pack["rule_desc"]
        meta = pack["meta"]
        qtype_meta = "rotation"

    elif q_type.startswith("Mirror Image"):
        pack = generate_mirror_choice(rng, difficulty=difficulty)
        q_id = f"mirror-{int(time.time())}"
        base = pack["base_img"]
        tile_size = base.size
        m_font = max(22, int(tile_size[0] * 0.12))
        problem_img = compose_grid([base, text_image("Mirror ->", size=tile_size, font_size=m_font)], (1, 2))
        choices_imgs = pack["choices_imgs"]
        correct_index = pack["correct_index"]
        prompt_text = pack["prompt"]
        rule_desc = pack["rule_desc"]
        meta = pack["meta"]
        qtype_meta = "mirror"

    elif q_type.startswith("Sample (Image)"):
        q_id = f"sample-{int(time.time())}"
        sample_bytes = st.session_state.get("sample_image_bytes")
        if (sample_bytes is None) and ref_files:
            sample_bytes = ref_files[0].getvalue()
            st.session_state.sample_sel_name = ref_files[0].name
            st.session_state.sample_image_bytes = sample_bytes
        if sample_bytes:
            try:
                problem_img = Image.open(io.BytesIO(sample_bytes)).convert("RGB")
            except Exception:
                problem_img = text_image("Could not open image", size=(800, 600), font_size=36)
        else:
            problem_img = text_image("Upload a sample image in the sidebar", size=(1000, 320), font_size=36)

        n_opts = int(sample_options_n)
        labels_sample = [chr(ord('A') + i) for i in range(n_opts)]
        choices_imgs = [text_image(lbl, size=(420, 420), font_size=140) for lbl in labels_sample]
        correct_label_now = st.session_state.get("sample_correct_label", "A")
        if correct_label_now not in labels_sample:
            correct_label_now = labels_sample[0]
        correct_index = labels_sample.index(correct_label_now)

        prompt_text = sample_prompt
        rule_desc = sample_rule
        qtype_meta = "sample"
        meta = {
            "type": "sample",
            "sample_filename": st.session_state.get("sample_sel_name"),
            "options_count": n_opts
        }

    else:
        # Mimic Sample (Procedural) -> generate a matrix using extracted style and optionally forced rules
        q_id = f"mimic-{int(time.time())}"
        # If style not available, fall back gracefully
        style_used = extracted_style if extracted_style else None

        # Generate a matrix (most common pattern for samples)
        pack = generate_matrix_reasoning(
            rng,
            difficulty=difficulty,
            style=style_used,
            forced_rules=mimic_forced_rules
        )
        # Compose with mimic background if present
        bg_for_grid = style_used.get("bg") if style_used else (255, 255, 255)
        problem_img = compose_grid(pack["grid_imgs"], pack["grid_size"], bg=bg_for_grid)
        choices_imgs = pack["choices_imgs"]
        correct_index = pack["correct_index"]
        prompt_text = "Which option completes the matrix in the style of the sample?"
        rule_desc = pack["rule_desc"]
        meta = pack["meta"]
        if style_used:
            meta["mimic_style"] = {
                "bg": style_used["bg"],
                "outline": style_used["outline"],
                "fills": style_used["fills"],
                "stroke_width": style_used["stroke_width"]
            }
        qtype_meta = "mimic"

    # Prepare labeled choices overlay (uniform UX)
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

    # Optional LLM explanation
    llm_expl = None
    if include_llm:
        llm_prompt = build_llm_prompt(qtype_meta, rule_desc, prompt_text, len(labels), labels[correct_index])
        llm_expl = ollama_chat(ollama_host, ollama_model, llm_prompt, temperature=temperature)

    st.session_state.qpack = {
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

# Render
qp = st.session_state.get("qpack")
if qp:
    colQ, colA = st.columns([2.1, 1.4])
    with colQ:
        if show_guides:
            st.subheader("Question")
        st.image(qp["problem_img"], use_container_width=True)
        st.write(qp["prompt"])

        if ref_files:
            with st.expander("Reference samples (uploaded)"):
                st.image(ref_files, use_container_width=True)

    with colA:
        if show_guides:
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
                choice_items.append(ChoiceItem(
                    label=qp["labels"][i],
                    is_correct=(i == qp["correct_index"]),
                    image_filename=cfname
                ))

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
