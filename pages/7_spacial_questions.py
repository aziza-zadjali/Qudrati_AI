# -*- coding: utf-8 -*-
# pages/7_spacial_questions.py
# Matrix reasoning (3x3) with optional style mimic from an uploaded sample.
# ASCII-only, Pillow 10+ safe.

import io
import time
import math
import random
import zipfile
from typing import List, Tuple, Optional, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

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

def text_image(text: str, size=(380, 380), font_size=42,
               color=(30, 30, 30), bg=(255, 255, 255)) -> Image.Image:
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    font = _load_font(font_size)
    try:
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top
        x = (size[0] - w) / 2 - left
        y = (size[1] - h) / 2 - top
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

def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)

def _draw_polygon(draw, pts, fill, outline, width):
    draw.polygon(pts, fill=fill)
    pts_closed = list(pts) + [pts[0]]
    try:
        draw.line(pts_closed, fill=outline, width=width, joint="curve")
    except TypeError:
        draw.line(pts_closed, fill=outline, width=width)

def _rotate_points(points, angle_deg, origin):
    ox, oy = origin
    ang = math.radians(angle_deg)
    out = []
    for x, y in points:
        qx = ox + math.cos(ang) * (x - ox) - math.sin(ang) * (y - oy)
        qy = oy + math.sin(ang) * (x - ox) + math.cos(ang) * (y - oy)
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
            ang = 90 + 72 * k
            pts.append((cx + r * math.cos(math.radians(ang)), cy + r * math.sin(math.radians(ang))))
        pts = _rotate_points(pts, rotation_deg, (cx, cy))
        _draw_polygon(draw, pts, fill=fill, outline=outline, width=width)
    else:
        draw.line([cx - r, cy, cx + r, cy], fill=outline, width=width)
        draw.line([cx, cy - r, cx, cy + r], fill=outline, width=width)

def compose_grid(images, grid_size, pad=16, bg=(255,255,255)):
    rows, cols = grid_size
    assert len(images) == rows * cols
    w, h = images[0].size
    for im in images:
        if im.size != (w, h):
            raise ValueError("Tile size mismatch")
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
# Optional style extraction to mimic sample palette
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
    unique = [rgb for _, rgb in colors_freq] or [(30,30,30), (240,240,240)]
    outline = min(unique, key=lum)
    fills = [c for c in unique if c != bg] or [(30,30,30), (0,88,155), (200,0,0)]
    stroke_width = 5 if img.width < 800 else (6 if img.width < 1400 else 8)
    text_color = (20,20,20) if lum(bg) > 186 else (240,240,240)
    return {"bg": bg, "outline": outline, "fills": fills, "stroke_width": stroke_width, "text_color": text_color}

# ----------------------------
# Matrix generator (3x3)
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

def generate_matrix_reasoning(rng: random.Random, img_size=(380,380), cell_shape_size=160,
                              difficulty="Medium", style: Optional[Dict] = None) -> Dict:
    fills = (style.get("fills") if style else None) or DEFAULT_COLORS
    outline = (style.get("outline") if style else None) or (10, 10, 10)
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    stroke = (style.get("stroke_width") if style else None) or 4
    txt_col = (style.get("text_color") if style else None) or (30, 30, 30)

    if difficulty == "Easy":
        rules_to_use = 1; step_choices = [0, 45, 90]
    elif difficulty == "Hard":
        rules_to_use = 3; step_choices = [30, 45, 60, 90]
    else:
        rules_to_use = 2; step_choices = [30, 45, 60, 90]

    use_rotation = use_count = use_color = use_size = False
    for s in rng.sample(["rotation", "count", "color", "size"], rules_to_use):
        if s == "rotation": use_rotation = True
        elif s == "count":  use_count = True
        elif s == "color":  use_color = True
        elif s == "size":   use_size = True

    rotation_step = rng.choice(step_choices) if use_rotation else 0
    base_count = rng.randint(1, 2) if use_count else 1
    base_size = cell_shape_size
    size_step = rng.choice([-20, 20]) if use_size else 0
    base_color_idx = rng.randrange(len(fills)) if use_color else 0
    color_by_row = (rng.random() < 0.5) if use_color else True
    shape = rng.choice(DEFAULT_SHAPES)

    tiles = []
    desc = []
    for r in range(3):
        for c in range(3):
            if r == 2 and c == 2:
                tiles.append(text_image("?", size=img_size, font_size=int(img_size[0] * 0.32),
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
            mini = min(size_px, int(0.8 * min(cell_w, cell_h)))
            k = 0
            for rr in range(rows_cnt):
                for cc in range(cols_cnt):
                    if k >= count: break
                    cx = margin + cc * cell_w + cell_w // 2
                    cy = margin + rr * cell_h + cell_h // 2
                    draw_shape(d, shape, (cx, cy), mini, rotation_deg=rot,
                               fill=fills[color_idx], outline=outline, width=stroke)
                    k += 1
            tiles.append(img)

    correct_rot = (2 * rotation_step) % 360 if use_rotation else 0
    correct_count = base_count + 2 if use_count else 1
    correct_size = max(50, base_size + 2 * size_step) if use_size else base_size
    if use_color:
        color_idx_correct = (base_color_idx + 2) % len(fills)
    else:
        color_idx_correct = base_color_idx

    def render_multi(cnt, rot, sz, col_idx):
        img = Image.new("RGB", img_size, bg)
        d = ImageDraw.Draw(img)
        cols_cnt = int(math.ceil(math.sqrt(cnt)))
        rows_cnt = int(math.ceil(cnt / cols_cnt))
        grid_w, grid_h = img_size
        margin = 30
        cell_w = (grid_w - 2 * margin) // cols_cnt
        cell_h = (grid_h - 2 * margin) // rows_cnt
        mini = min(sz, int(0.8 * min(cell_w, cell_h)))
        k = 0
        for rr in range(rows_cnt):
            for cc in range(cols_cnt):
                if k >= cnt: break
                cx = margin + cc * cell_w + cell_w // 2
                cy = margin + rr * cell_h + cell_h // 2
                draw_shape(d, shape, (cx, cy), mini, rotation_deg=rot,
                           fill=fills[col_idx], outline=outline, width=stroke)
                k += 1
        return img

    correct_img = render_multi(correct_count, correct_rot, correct_size, color_idx_correct)

    def make_distractor(var: str):
        cnt, rot, sz, col = correct_count, correct_rot, correct_size, color_idx_correct
        if var == "rotation":
            step = rotation_step if rotation_step != 0 else rng.choice([30,45,60,90])
            rot = (rot + rng.choice([-step, step])) % 360
        elif var == "count":
            cnt = max(1, cnt + rng.choice([-1, 1]))
        elif var == "size":
            sz = max(50, sz + rng.choice([-20, 20]))
        elif var == "color":
            col = (col + rng.choice([1, -1])) % len(fills)
        return render_multi(cnt, rot, sz, col)

    kinds = []
    if use_rotation: kinds.append("rotation")
    if use_count:    kinds.append("count")
    if use_size:     kinds.append("size")
    if use_color:    kinds.append("color")
    while len(kinds) < 3:
        kinds.append(rng.choice(["rotation","count","size","color"]))

    choices = [correct_img] + [make_distractor(k) for k in kinds]
    rng.shuffle(choices)

    if use_rotation: desc.append("Rotation increases by %d deg." % rotation_step)
    if use_count:    desc.append("Count increases by 1 down rows.")
    if use_color:    desc.append("Color alternates.")
    if use_size:     desc.append("Size changes by %+d px per row." % size_step)
    rule_desc = " ".join(desc) if desc else "Follow the visual pattern."

    return {
        "grid_imgs": tiles,
        "grid_size": (3, 3),
        "choices_imgs": choices,
        "correct_index": choices.index(correct_img),
        "rule_desc": rule_desc,
        "prompt": "Which option completes the 3x3 matrix?",
        "meta": {"type": "matrix"}
    }

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Matrix (Mimic Sample)", layout="wide")
st.title("Matrix Reasoning (3x3)")

with st.sidebar:
    st.header("Controls")
    difficulty = st.select_slider("Difficulty", options=["Easy","Medium","Hard"], value="Medium")
    seed_str = st.text_input("Seed (optional)", value="")
    seed = None
    if seed_str.strip():
        try:
            seed = int(seed_str.strip())
        except Exception:
            seed = abs(hash(seed_str)) % (2**31)
    st.header("Style")
    bg_hex = st.text_input("Background", "#FFFFFF")
    outline_hex = st.text_input("Outline", "#141414")
    text_hex = st.text_input("Text", "#141414")
    stroke_w = st.number_input("Stroke width", min_value=2, max_value=12, value=4, step=1)

    style = {
        "bg": tuple(int(bg_hex.strip().lstrip("#")[i:i+2],16) for i in (0,2,4)) if len(bg_hex.strip().lstrip("#")) in (3,6) else (255,255,255),
        "outline": tuple(int(outline_hex.strip().lstrip("#")[i:i+2],16) for i in (0,2,4)) if len(outline_hex.strip().lstrip("#")) in (3,6) else (20,20,20),
        "text_color": tuple(int(text_hex.strip().lstrip("#")[i:i+2],16) for i in (0,2,4)) if len(text_hex.strip().lstrip("#")) in (3,6) else (30,30,30),
        "stroke_width": int(stroke_w)
    }

    st.header("Mimic from Sample (optional)")
    sample_bytes = None
    file_up = st.file_uploader("Upload image to mimic palette", type=["png","jpg","jpeg"], accept_multiple_files=False)
    if file_up:
        sample_bytes = file_up.getvalue()

    st.markdown("---")
    gen_btn = st.button("Generate New Question", type="primary", use_container_width=True)

    st.subheader("Batch")
    batch_n = st.number_input("How many to generate", min_value=2, max_value=100, value=10, step=1)
    batch_btn = st.button("Generate Batch ZIP", use_container_width=True)

rng = make_rng(seed)

def make_one():
    local_style = style.copy()
    if sample_bytes:
        try:
            im = Image.open(io.BytesIO(sample_bytes)).convert("RGB")
            ext = extract_style_from_image(im)
            local_style["fills"] = ext.get("fills")
        except Exception:
            pass
    pack = generate_matrix_reasoning(rng, difficulty=difficulty, style=local_style)
    problem = compose_grid(pack["grid_imgs"], pack["grid_size"])
    return {
        "problem_img": problem,
        "choices_imgs": pack["choices_imgs"],
        "correct_index": pack["correct_index"],
        "prompt": pack["prompt"],
        "rule_desc": pack["rule_desc"]
    }

if ("mx_qpack" not in st.session_state) or gen_btn:
    st.session_state.mx_qpack = make_one()

qp = st.session_state.get("mx_qpack")
if qp:
    colQ, colA = st.columns([2.1, 1.4])
    with colQ:
        st.subheader("Question")
        show_image(qp["problem_img"])
        st.write(qp["prompt"])
    with colA:
        st.subheader("Choices")
        labels = [chr(ord('A') + i) for i in range(len(qp["choices_imgs"]))]
        chosen = st.radio("Select your answer:", labels, index=0, horizontal=True, label_visibility="collapsed")
        cols2 = st.columns(2)
        for i, im in enumerate(qp["choices_imgs"]):
            with cols2[i % 2]:
                show_image(im, caption="Option " + labels[i])
        if st.button("Check answer"):
            if chosen == labels[qp["correct_index"]]:
                st.success("Correct. Answer: " + labels[qp["correct_index"]])
            else:
                st.error("Not quite. Correct answer: " + labels[qp["correct_index"]])
            st.markdown("Why: " + qp["rule_desc"])

    st.markdown("---")
    st.subheader("Export")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG")
        zf.writestr("problem.png", pb.getvalue())
        labels = [chr(ord('A') + i) for i in range(len(qp["choices_imgs"]))]
        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG")
            zf.writestr("choice_%s.png" % labels[i], cb.getvalue())
        meta = {
            "type": "matrix",
            "correct_label": labels[qp["correct_index"]],
            "rule": qp["rule_desc"]
        }
        zf.writestr("question.json", json.dumps(meta, indent=2))
    st.download_button("Download ZIP", data=buf.getvalue(),
                       file_name="matrix_%d.zip" % int(time.time()), mime="application/zip")

    if batch_btn:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(batch_n)):
                rng_local = make_rng((seed or 0) + k + 1)
                # reuse current style; mimic fills again if provided
                loc_style = style.copy()
                if sample_bytes:
                    try:
                        im = Image.open(io.BytesIO(sample_bytes)).convert("RGB")
                        ext = extract_style_from_image(im)
                        loc_style["fills"] = ext.get("fills")
                    except Exception:
                        pass
                pack = generate_matrix_reasoning(rng_local, difficulty=difficulty, style=loc_style)
                qid = "matrix_%d_%d" % (int(time.time()*1000), k)
                pb = io.BytesIO(); compose_grid(pack["grid_imgs"], pack["grid_size"]).save(pb, format="PNG")
                zf.writestr("%s/problem.png" % qid, pb.getvalue())
                labels = [chr(ord('A') + i) for i in range(len(pack["choices_imgs"]))]
                for i, im in enumerate(pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG")
                    zf.writestr("%s/choice_%s.png" % (qid, labels[i]), cb.getvalue())
                meta = {
                    "id": qid,
                    "type": "matrix",
                    "correct_label": labels[pack["correct_index"]],
                    "rule": pack["rule_desc"]
                }
                zf.writestr("%s/question.json" % qid, json.dumps(meta, indent=2))
                index.append(meta)
            zf.writestr("index.json", json.dumps(index, indent=2))
        st.download_button("Download Batch ZIP", data=bbuf.getvalue(),
                           file_name="batch_matrix_%d.zip" % int(time.time()),
                           mime="application/zip")
