# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py
# Stable, clear, large visuals:
# - Single-fold (L/R/U/D)
# - Curved arrow + dotted fold line + solid punch
# - Larger example and choices; example width matches grid width
# - Arabic prompt (exact) outside images; English UI
# - Clean export: composite question.png + JSON (ensure_ascii=False)

import io
import time
import math
import random
import zipfile
import json
from typing import List, Tuple, Optional, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ----------------------------
# Basic utils
# ----------------------------
def make_rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        seed = int(time.time() * 1000) % 2147483647
    return random.Random(seed)

def _try_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

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

def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)

# ----------------------------
# Geometry (single fold)
# ----------------------------
def reflect_point(p: Tuple[float, float], axis: str) -> Tuple[float, float]:
    x, y = p
    if axis == "V":  # across x=0
        return (-x, y)
    if axis == "H":  # across y=0
        return (x, -y)
    return (x, y)

def dir_to_axis_and_half(direction: str):
    """
    Returns (axis, shaded_half) where shaded_half in {'left','right','top','bottom'}
    indicates the half that folds over the other BEFORE punching.
    """
    if direction == "L":   # fold left: right half folds onto left
        return "V", "right"
    if direction == "R":   # fold right: left half folds onto right
        return "V", "left"
    if direction == "U":   # fold up: bottom half folds onto top
        return "H", "bottom"
    # "D": fold down: top half folds onto bottom
    return "H", "top"

def norm_to_px(xy, rect):
    """Map (x,y) in [-1,1] to pixel inside rect=(left,top,right,bottom) with (0,0) at top-left."""
    x, y = xy
    left, top, right, bottom = rect
    px = (x + 1) / 2.0 * (right - left) + left
    py = (1 - (y + 1) / 2.0) * (bottom - top) + top
    return px, py

# ----------------------------
# Drawing helpers
# ----------------------------
def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline, width=3):
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=width)

def paper_shadow(base_img: Image.Image, rect, blur=12, offset=(8, 10), opacity=90):
    l, t, r, b = rect
    w, h = r - l, b - t
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rectangle([0, 0, w, h], fill=(0, 0, 0, opacity))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    base_img.paste(shadow, (l + offset[0], t + offset[1]), shadow)

def dotted_line_circles(draw: ImageDraw.ImageDraw, p1, p2, color, dot_r=3, gap=10):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    steps = max(1, int(length // gap))
    for i in range(steps + 1):
        t = i / steps
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=color, outline=None)

def draw_arc_arrow(draw: ImageDraw.ImageDraw, bbox, start_deg, end_deg, color, width=6, head_len=18, head_w=12):
    draw.arc(bbox, start=start_deg, end=end_deg, fill=color, width=width)
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    rx = abs(bbox[2] - bbox[0]) / 2
    ry = abs(bbox[3] - bbox[1]) / 2
    t = math.radians(end_deg)
    tip_x = cx + rx * math.cos(t)
    tip_y = cy + ry * math.sin(t)
    vx = -rx * math.sin(t)
    vy =  ry * math.cos(t)
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    base_x = tip_x - ux * head_len
    base_y = tip_y - uy * head_len
    p1 = (base_x + px * head_w, base_y + py * head_w)
    p2 = (base_x - px * head_w, base_y - py * head_w)
    draw.polygon([(tip_x, tip_y), p1, p2], fill=color)

def overlay_label(img: Image.Image, label: str, spot="tr",
                  circle_fill=(255,255,255), circle_outline=(20,20,20),
                  text_color=(20,20,20)) -> Image.Image:
    im = img.copy()
    d = ImageDraw.Draw(im)
    r = 18
    margin = 10
    if spot == "tl":
        cx, cy = margin + r, margin + r
    elif spot == "tr":
        cx, cy = im.width - margin - r, margin + r
    elif spot == "bl":
        cx, cy = margin + r, im.height - margin - r
    else:  # br
        cx, cy = im.width - margin - r, im.height - margin - r
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=circle_fill, outline=circle_outline, width=2)
    font = _try_font(18)
    tw = d.textlength(label, font=font)
    d.text((cx - tw/2, cy - 9), label, fill=text_color, font=font)
    return im

def compose_2x2_grid(choices, labels, pad=22, tile_border=2, bg=(255,255,255), label_spot="tr"):
    tiles = choices[:4]
    labeled = [overlay_label(tiles[i], labels[i], spot=label_spot) for i in range(4)]
    w, h = labeled[0].size
    cols, rows = 2, 2
    W = cols * w + (cols + 1) * pad
    H = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(canvas)
    for idx, im in enumerate(labeled):
        r, c = divmod(idx, 2)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(im, (x, y))
        d.rectangle([x, y, x + w, y + h], outline=(20,20,20), width=tile_border)
    return canvas

def stack_vertical(top_img, bottom_img, pad=28, bg=(255,255,255)):
    W = max(top_img.width, bottom_img.width) + 2*pad
    H = top_img.height + bottom_img.height + 3*pad
    canvas = Image.new("RGB", (W, H), bg)
    x_top = (W - top_img.width)//2
    canvas.paste(top_img, (x_top, pad))
    x_bot = (W - bottom_img.width)//2
    canvas.paste(bottom_img, (x_bot, top_img.height + 2*pad))
    return canvas

# ----------------------------
# Reference-style example (folded paper)
# ----------------------------
def draw_example(direction: str, hole_xy_norm: Tuple[float,float],
                 out_width: int,  # final 1x width in px
                 bg=(255,255,255), paper_fill=(250,250,250),
                 outline=(20,20,20), fold_line_color=(60,60,60),
                 stroke=5, dpi_scale=2) -> Image.Image:
    # Layout: left = paper block, right = arrow block
    paper_block_w = int(out_width * 0.64)
    arrow_block_w = out_width - paper_block_w
    # We'll make example height proportional to paper_block_w for a tall portrait look
    out_height = int(paper_block_w * 1.2)

    # Hi-DPI canvas
    W, H = out_width * dpi_scale, out_height * dpi_scale
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # Paper rectangle (portrait with comfortable margins)
    margin = int(0.08 * paper_block_w) * dpi_scale
    left = margin
    right = paper_block_w * dpi_scale - margin
    paper_h = int((right - left) * 1.45)  # portrait ratio ~ 0.69 (close to A4 look)
    top = (H - paper_h) // 2
    bottom = top + paper_h
    paper_rect = (left, top, right, bottom)

    # Shadow + rounded paper
    paper_shadow(img, paper_rect, blur=14, offset=(8, 10), opacity=80)
    rounded_rect(d, paper_rect, radius=16, fill=paper_fill, outline=outline, width=stroke)

    # Fold line (dotted) + shaded half
    axis, shaded_half = dir_to_axis_and_half(direction)
    l, t, r, b = paper_rect
    cx = (l + r) // 2
    cy = (t + b) // 2
    if axis == "V":
        dotted_line_circles(d, (cx, t + 12), (cx, b - 12), fold_line_color, dot_r=4, gap=16)
    else:
        dotted_line_circles(d, (l + 12, cy), (r - 12, cy), fold_line_color, dot_r=4, gap=16)

    shade = Image.new("RGBA", (r - l, b - t), (0,0,0,0))
    sd = ImageDraw.Draw(shade)
    alpha = 28
    if axis == "V":
        if shaded_half == "left":
            sd.rectangle([0, 0, (cx - l), b - t], fill=(0,0,0,alpha))
        else:
            sd.rectangle([(cx - l), 0, r - l, b - t], fill=(0,0,0,alpha))
    else:
        if shaded_half == "top":
            sd.rectangle([0, 0, r - l, (cy - t)], fill=(0,0,0,alpha))
        else:
            sd.rectangle([0, (cy - t), r - l, b - t], fill=(0,0,0,alpha))
    img.paste(shade, (l, t), shade)

    # Punch (solid)
    hx, hy = norm_to_px(hole_xy_norm, paper_rect)
    pr = 14
    d.ellipse([hx - pr, hy - pr, hx + pr, hy + pr], fill=outline, outline=outline)

    # Curved arrow (in right block)
    arrow_left = paper_block_w * dpi_scale + 10
    arrow_right = out_width * dpi_scale - 10
    arc_hh = int((b - t) * 0.45)
    arc_top = cy - arc_hh
    arc_bottom = cy + arc_hh

    if axis == "V":
        start, end = (-20, 215) if direction == "L" else (200, -20)
        draw_arc_arrow(d, (arrow_left, arc_top, arrow_right, arc_bottom),
                       start, end, outline, width=8, head_len=26, head_w=16)
    else:
        arc_left = (l + r)//2 - (arrow_right - arrow_left)//3
        arc_right = (l + r)//2 + (arrow_right - arrow_left)//3
        arc_top2 = b + 24
        arc_bottom2 = arc_top2 + (arc_bottom - arc_top)
        start, end = (160, 342) if direction == "U" else (-20, 160)
        draw_arc_arrow(d, (arc_left, arc_top2, arc_right, arc_bottom2),
                       start, end, outline, width=8, head_len=26, head_w=16)

    # Downsample
    if dpi_scale != 1:
        img = img.resize((out_width, out_height), Image.LANCZOS)
    return img

# ----------------------------
# Unfolded (choices) — large, portrait tiles
# ----------------------------
def draw_choice(holes_norm: List[Tuple[float,float]],
                tile_size=(420, 520),  # larger tiles
                bg=(255,255,255), paper_fill=(250,250,250),
                outline=(20,20,20), stroke=5, dpi_scale=2) -> Image.Image:
    W, H = tile_size[0]*dpi_scale, tile_size[1]*dpi_scale
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # Paper rectangle with comfortable margins (portrait)
    margin = int(0.10 * W)
    left = margin
    right = W - margin
    paper_h = int((right - left) * 1.45)  # same portrait look as example
    top = max(20, (H - paper_h) // 2)
    bottom = top + paper_h
    paper_rect = (left, top, right, bottom)

    paper_shadow(img, paper_rect, blur=12, offset=(6, 8), opacity=80)
    rounded_rect(d, paper_rect, radius=14, fill=paper_fill, outline=outline, width=stroke)

    r = 14
    for (x, y) in holes_norm:
        px, py = norm_to_px((x, y), paper_rect)
        d.ellipse([px - r, py - r, px + r, py + r], fill=outline, outline=outline)

    if dpi_scale != 1:
        img = img.resize(tile_size, Image.LANCZOS)
    return img

# ----------------------------
# Question generator (single fold)
# ----------------------------
def generate_single_fold_question(rng: random.Random, style: Optional[Dict]=None) -> Dict:
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    fold_line_color = (style.get("fold_line") if style else None) or (60, 60, 60)
    stroke = (style.get("stroke_width") if style else None) or 6
    dpi_scale = int((style.get("dpi_scale") if style else None) or 2)

    direction = rng.choice(["L", "R", "U", "D"])
    axis, _ = dir_to_axis_and_half(direction)

    # Keep punch away from edges & fold axis for clarity
    def safe_rand():
        return rng.uniform(-0.68, 0.68)
    px = safe_rand()
    py = safe_rand()
    if axis == "V" and abs(px) < 0.12:
        px = 0.12 * (1 if px >= 0 else -1)
    if axis == "H" and abs(py) < 0.12:
        py = 0.12 * (1 if py >= 0 else -1)
    p_folded = (px, py)
    p_reflected = reflect_point(p_folded, axis)

    # Build choices (large tiles)
    choice_size = (420, 520)
    correct = draw_choice([p_folded, p_reflected], choice_size, bg, paper_fill, outline, stroke, dpi_scale)
    d1 = draw_choice([p_folded], choice_size, bg, paper_fill, outline, stroke, dpi_scale)
    wrong_axis = "H" if axis == "V" else "V"
    d2 = draw_choice([p_folded, reflect_point(p_folded, wrong_axis)], choice_size, bg, paper_fill, outline, stroke, dpi_scale)
    rot = lambda pt: (pt[1], -pt[0])
    d3 = draw_choice([rot(p_folded), rot(p_reflected)], choice_size, bg, paper_fill, outline, stroke, dpi_scale)

    choices = [correct, d1, d2, d3]
    rng.shuffle(choices)
    correct_index = choices.index(correct)

    # Grid first (to get width), then example with the same width
    labels_ar = ["أ", "ب", "ج", "د"]
    grid = compose_2x2_grid(choices, labels_ar, pad=24, tile_border=2, bg=bg, label_spot="tr")
    example = draw_example(direction, p_folded, out_width=grid.width,
                           bg=bg, paper_fill=paper_fill, outline=outline,
                           fold_line_color=fold_line_color, stroke=stroke, dpi_scale=dpi_scale)

    prompt_ar = "ما رمز البديل الذي يحتوي على صورة الورقة بعد إعادة فتحها من بين البدائل الأربعة؟"
    meta = {
        "type": "folding_single",
        "direction": direction,
        "axis": axis,
        "punch": (round(px, 3), round(py, 3))
    }

    return {
        "example_img": example,
        "choices_imgs": choices,
        "grid_img": grid,
        "correct_index": correct_index,
        "labels_ar": labels_ar,
        "prompt": prompt_ar,
        "meta": meta
    }

# ----------------------------
# Streamlit UI (English UI, Arabic prompt)
# ----------------------------
st.set_page_config(page_title="Paper Folding — Clear & Large", layout="wide")
st.title("Paper Folding — Clear & Large (Single Fold)")

with st.sidebar:
    st.header("Controls")
    seed_str = st.text_input("Seed (optional)", value="")
    seed = None
    if seed_str.strip():
        try:
            seed = int(seed_str.strip())
        except Exception:
            seed = abs(hash(seed_str)) % (2**31)

    st.header("Visual Style")
    bg_hex = st.text_input("Background", "#FFFFFF")
    paper_hex = st.text_input("Paper fill", "#FAFAFA")
    outline_hex = st.text_input("Outline", "#1A1A1A")
    fold_line_hex = st.text_input("Fold-line (dots)", "#3C3C3C")
    stroke_w = st.number_input("Paper border thickness (px)", min_value=2, max_value=12, value=6, step=1)
    dpi_scale = st.slider("Quality (internal render scale)", min_value=1, max_value=3, value=2)

    st.markdown("---")
    gen_btn = st.button("Generate New Question", type="primary", use_container_width=True)

    st.subheader("Batch")
    batch_n = st.number_input("How many?", min_value=2, max_value=100, value=8, step=1)
    batch_btn = st.button("Generate Batch ZIP", use_container_width=True)

style = {
    "bg": _hex_to_rgb(bg_hex),
    "paper_fill": _hex_to_rgb(paper_hex),
    "outline": _hex_to_rgb(outline_hex),
    "fold_line": _hex_to_rgb(fold_line_hex),
    "stroke_width": int(stroke_w),
    "dpi_scale": int(dpi_scale),
}

rng = make_rng(seed)

def make_one():
    return generate_single_fold_question(rng, style=style)

if ("fold_qpack" not in st.session_state) or gen_btn:
    st.session_state.fold_qpack = make_one()

qp = st.session_state.get("fold_qpack")

if qp:
    # Composite (example above, 2x2 grid below)
    composite = stack_vertical(qp["example_img"], qp["grid_img"], pad=28, bg=style["bg"])

    st.subheader("Question")
    # Exact Arabic text (outside image for correct shaping)
    st.write(qp["prompt"])
    show_image(composite)

    chosen = st.radio("Pick your answer:", qp["labels_ar"], index=0, horizontal=True)
    if st.button("Check answer"):
        if chosen == qp["labels_ar"][qp["correct_index"]]:
            st.success(f"Correct. Answer: {qp['labels_ar'][qp['correct_index']]}")
        else:
            st.error(f"Not quite. Correct answer: {qp['labels_ar'][qp['correct_index']]}")

    st.markdown("---")
    st.subheader("Export")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        eb = io.BytesIO(); qp["example_img"].save(eb, format="PNG")
        zf.writestr("example.png", eb.getvalue())

        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG")
            zf.writestr(f"choice_{qp['labels_ar'][i]}.png", cb.getvalue())

        gb = io.BytesIO(); qp["grid_img"].save(gb, format="PNG")
        zf.writestr("grid.png", gb.getvalue())

        qb = io.BytesIO(); composite.save(qb, format="PNG")
        zf.writestr("question.png", qb.getvalue())

        meta = {
            "type": qp["meta"]["type"],
            "direction": qp["meta"]["direction"],
            "axis": qp["meta"]["axis"],
            "punch": qp["meta"]["punch"],
            "prompt": qp["prompt"],
            "labels": qp["labels_ar"],
            "correct_label": qp["labels_ar"][qp["correct_index"]],
        }
        zf.writestr("question.json", json.dumps(meta, indent=2, ensure_ascii=False))

    st.download_button(
        "Download ZIP",
        data=buf.getvalue(),
        file_name=f"folding_clear_{int(time.time())}.zip",
        mime="application/zip"
    )

    if batch_btn:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(batch_n)):
                local_rng = make_rng((seed or 0) + k + 1)
                local_pack = generate_single_fold_question(local_rng, style=style)
                qid = f"folding_clear_{int(time.time()*1000)}_{k}"

                eb = io.BytesIO(); local_pack["example_img"].save(eb, format="PNG")
                zf.writestr(f"{qid}/example.png", eb.getvalue())

                grid = local_pack["grid_img"]
                gb = io.BytesIO(); grid.save(gb, format="PNG")
                zf.writestr(f"{qid}/grid.png", gb.getvalue())

                comp = stack_vertical(local_pack["example_img"], grid, pad=28, bg=style["bg"])
                qb = io.BytesIO(); comp.save(qb, format="PNG")
                zf.writestr(f"{qid}/question.png", qb.getvalue())

                labels_ar = local_pack["labels_ar"]
                for i, im in enumerate(local_pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG")
                    zf.writestr(f"{qid}/choice_{labels_ar[i]}.png", cb.getvalue())

                meta = {
                    "id": qid,
                    "type": local_pack["meta"]["type"],
                    "direction": local_pack["meta"]["direction"],
                    "axis": local_pack["meta"]["axis"],
                    "punch": local_pack["meta"]["punch"],
                    "prompt": local_pack["prompt"],
                    "labels": labels_ar,
                    "correct_label": labels_ar[local_pack["correct_index"]],
                }
                zf.writestr(f"{qid}/question.json", json.dumps(meta, indent=2, ensure_ascii=False))
                index.append(meta)

            zf.writestr("index.json", json.dumps(index, indent=2, ensure_ascii=False))

        st.download_button(
            "Download Batch ZIP",
            data=bbuf.getvalue(),
            file_name=f"batch_folding_clear_{int(time.time())}.zip",
            mime="application/zip"
        )
