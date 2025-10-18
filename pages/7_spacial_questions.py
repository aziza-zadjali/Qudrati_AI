# -*- coding: utf-8 -*-
# File: pages/7_spacial_questions.py
# Title: Spatial IQ Question Generator (Matrix, Rotation, Mirror)
# Notes:
#   - Button relocated to sidebar near settings for better UX.
#   - Optional: Auto-generate when settings change.
#   - ASCII-only strings to avoid parsing issues in some environments.
#   - Fixes Pillow deprecation (uses textbbox instead of textsize).
#   - Ensures placeholder tiles match neighbor tile sizes.
#   - Procedural generation (offline), optional LLM explanations via Ollama.
#   - Export ZIP with images + JSON metadata.
#   - Updated: replaced deprecated use_column_width with use_container_width.

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

def text_image(text: str, size=(380, 380), font_size=42,
               color=(30, 30, 30), bg=(255, 255, 255)) -> Image.Image:
    """
    Create an image with centered text. Compatible with Pillow 10+ (no textsize).
    Uses textbbox; fallback to textlength approximation.
    """
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
# Puzzle generators
# ----------------------------

SHAPES = ["circle", "square", "triangle", "pentagon"]
COLORS = [
    (30, 30, 30),      # dark gray
    (0, 88, 155),      # blue
    (200, 0, 0),       # red
    (0, 140, 70),      # green
    (220, 120, 0),     # orange
    (120, 0, 160),     # purple
]

def generate_matrix_reasoning(rng: random.Random, img_size=(380, 380), cell_shape_size=160, difficulty="Medium"):
    """
    Generate a 3x3 matrix where the bottom-right tile is missing (question mark).
    Rules considered:
      - Rotation increases by fixed step per column
      - Count of shapes increases per row
      - Color alternates by row or by column
      - Size changes by a fixed step per row
    """
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
        if s == "rotation":
            use_rotation = True
        elif s == "count":
            use_count = True
        elif s == "color":
            use_color = True
        elif s == "size":
            use_size = True

    rotation_step = rng.choice(rotation_step_choices) if use_rotation else 0
    base_count = rng.randint(1, 2) if use_count else 1
    base_size = cell_shape_size
    size_step = rng.choice([-20, 20]) if use_size else 0
    base_color_idx = rng.randrange(len(COLORS)) if use_color else 0
    color_by_row = (rng.random() < 0.5) if use_color else True

    shape = rng.choice(SHAPES)

    grid_imgs = []
    rule_desc_parts = []
    for r in range(3):
        for c in range(3):
            if r == 2 and c == 2:
                grid_imgs.append(text_image("?", size=img_size, font_size=int(img_size[0] * 0.32)))
                continue

            img = Image.new("RGB", img_size, (255, 255, 255))
            d = ImageDraw.Draw(img)

            rot = (c * rotation_step) % 360 if use_rotation else 0
            count = base_count + r if use_count else 1
            size_px = max(50, base_size + r * size_step) if use_size else base_size
            if use_color:
                shift = r if color_by_row else c
                color_idx = (base_color_idx + shift) % len(COLORS)
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
                    if k >= count:
                        break
                    cx = margin + cc * cell_w + cell_w // 2
                    cy = margin + rr * cell_h + cell_h // 2
                    draw_shape(d, shape, (cx, cy), mini_size, rotation_deg=rot,
                               fill=COLORS[color_idx], outline=(10, 10, 10), width=4)
                    k += 1
            grid_imgs.append(img)

    correct_rot = (2 * rotation_step) % 360 if use_rotation else 0
    correct_count = base_count + 2 if use_count else 1
    correct_size = max(50, base_size + 2 * size_step) if use_size else base_size
    if use_color:
        shift = 2 if color_by_row else 2
        color_idx_correct = (base_color_idx + shift) % len(COLORS)
    else:
        color_idx_correct = base_color_idx

    def render_multi(count, rot, sz, color_idx):
        img = Image.new("RGB", img_size, (255, 255, 255))
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
                if k2 >= count:
                    break
                cx = margin + cc * cell_w + cell_w // 2
                cy = margin + rr * cell_h + cell_h // 2
                draw_shape(d, shape, (cx, cy), mini_size2, rotation_deg=rot,
                           fill=COLORS[color_idx], outline=(10, 10, 10), width=4)
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
            col = (col + rng.choice([1, -1])) % len(COLORS)
        return render_multi(cnt, rot, sz, col)

    distractor_kinds = []
    if use_rotation:
        distractor_kinds.append("rotation")
    if use_count:
        distractor_kinds.append("count")
    if use_size:
        distractor_kinds.append("size")
    if use_color:
        distractor_kinds.append("color")
    while len(distractor_kinds) < 3:
        distractor_kinds.append(rng.choice(["rotation", "count", "size", "color"]))

    choices_imgs = [correct_img] + [make_distractor(k) for k in distractor_kinds]
    rng.shuffle(choices_imgs)

    if use_rotation:
        rule_desc_parts.append(f"Rotation increases by {rotation_step} deg across columns.")
    if use_count:
        rule_desc_parts.append("Number of shapes increases by 1 down the rows.")
    if use_color:
        rule_desc_parts.append(f"Color alternates by {'row' if color_by_row else 'column'}.")
    if use_size:
        rule_desc_parts.append(f"Shape size changes by {size_step:+d} px per row.")

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
        }
    }

def generate_rotation_sequence(rng: random.Random, img_size=(420, 420), shape_size=200, difficulty="Medium"):
    steps = 3 if difficulty == "Easy" else (4 if difficulty == "Medium" else 5)
    shape = rng.choice(SHAPES)
    step_angle = rng.choice([30, 45, 60, 90])
    start_angle = rng.choice([0, 15, 30, 45])
    color = rng.choice(COLORS)

    seq_imgs = []
    for i in range(steps):
        img = Image.new("RGB", img_size, (255, 255, 255))
        d = ImageDraw.Draw(img)
        draw_shape(d, shape, (img_size[0] // 2, img_size[1] // 2), shape_size,
                   rotation_deg=(start_angle + i * step_angle) % 360, fill=color, outline=(10, 10, 10), width=5)
        seq_imgs.append(img)

    correct_rot = (start_angle + steps * step_angle) % 360
    correct_img = Image.new("RGB", img_size, (255, 255, 255))
    d = ImageDraw.Draw(correct_img)
    draw_shape(d, shape, (img_size[0] // 2, img_size[1] // 2), shape_size,
               rotation_deg=correct_rot, fill=color, outline=(10, 10, 10), width=5)

    choices = [correct_img]
    for delta_mult in [1, -1, 2]:
        img = Image.new("RGB", img_size, (255, 255, 255))
        d = ImageDraw.Draw(img)
        wrong_rot = (correct_rot + delta_mult * step_angle) % 360
        draw_shape(d, shape, (img_size[0] // 2, img_size[1] // 2), shape_size,
                   rotation_deg=wrong_rot, fill=color, outline=(10, 10, 10), width=5)
        choices.append(img)
    rng.shuffle(choices)

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
            "steps_shown": steps
        }
    }

def generate_mirror_choice(rng: random.Random, img_size=(420, 420), shape_size=220, difficulty="Medium"):
    shape = rng.choice(SHAPES)
    mirror_axis = rng.choice(["horizontal", "vertical"])
    rotation_deg = rng.choice([0, 15, 30, 45, 60, 75, 90])
    color = rng.choice(COLORS)

    base = Image.new("RGB", img_size, (255, 255, 255))
    d = ImageDraw.Draw(base)
    draw_shape(d, shape, (img_size[0] // 2, img_size[1] // 2), shape_size,
               rotation_deg=rotation_deg, fill=color, outline=(10, 10, 10), width=5)

    correct = base.transpose(Image.FLIP_LEFT_RIGHT) if mirror_axis == "vertical" else base.transpose(Image.FLIP_TOP_BOTTOM)
    wrong1 = base.transpose(Image.FLIP_TOP_BOTTOM) if mirror_axis == "vertical" else base.transpose(Image.FLIP_LEFT_RIGHT)
    wrong2 = base.rotate(180)
    wrong3 = base.rotate(90)

    choices = [correct, wrong1, wrong2, wrong3]
    rng.shuffle(choices)

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
            "rotation_deg": rotation_deg
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
        ["Matrix (3x3)", "Rotation Sequence", "Mirror Image"],
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
    ref_files = st.file_uploader("Upload sample question images (to display as reference)",
                                 type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Generate")
    auto_generate = st.checkbox("Auto-generate when settings change", value=False)
    trigger_sidebar = st.button("Generate New Question", type="primary", use_container_width=True)
    st.caption("Tip: Change settings above, then click Generate. Or enable auto-generate.")

colQ, colA = st.columns([2.1, 1.4])

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
}
prev_settings = st.session_state.get("prev_settings")
settings_changed = (prev_settings is not None) and (prev_settings != current_settings)
st.session_state.prev_settings = current_settings  # update for next run

# Decide trigger: first load OR sidebar button OR auto-generate on change
trigger = ("qpack" not in st.session_state) or trigger_sidebar or (auto_generate and settings_changed)

if trigger:
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

    elif q_type.startswith("Rotation"):
        pack = generate_rotation_sequence(rng, difficulty=difficulty)
        q_id = f"rotation-{int(time.time())}"
        seq = pack["sequence_imgs"]
        tile_size = seq[0].size
        q_font = max(28, int(tile_size[0] * 0.28))
        problem_img = compose_grid(seq + [text_image("?", size=tile_size, font_size=q_font)], (1, len(seq) + 1))
        choices_imgs = pack["choices_imgs"]
        correct_index = pack["correct_index"]
        prompt_text = pack["prompt"]
        rule_desc = pack["rule_desc"]
        meta = pack["meta"]
        qtype_meta = "rotation"

    else:
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
        prompt = build_llm_prompt(qtype_meta, rule_desc, prompt_text, len(labels), labels[correct_index])
        llm_expl = ollama_chat(ollama_host, ollama_model, prompt, temperature=temperature)

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
