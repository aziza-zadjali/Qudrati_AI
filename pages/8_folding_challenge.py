# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py
# Single-fold "paper folding" question — enhanced reference style:
# - Rounded-rectangle paper with soft shadow
# - True dotted fold line (round dots)
# - Smooth curved arrow with proper head
# - Solid punch circle
# - 2x2 options (أ/ب/ج/د) with labeled badges
# - Hi-DPI render (2x) + downsample for sharpness
# - Arabic prompt text exactly as requested (outside image)
# - Exports composite question.png + JSON (ensure_ascii=False)

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
    # Use rounded_rectangle if available, else fallback.
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    else:
        # Simple fallback: regular rectangle
        draw.rectangle(box, fill=fill, outline=outline, width=width)

def paper_shadow(base_img: Image.Image, rect, blur=12, offset=(8, 10), opacity=90):
    """Render a soft shadow and apply to base_img."""
    l, t, r, b = rect
    w, h = r - l, b - t
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rectangle([0, 0, w, h], fill=(0, 0, 0, opacity))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    # Paste with offset
    base_img.paste(shadow, (l + offset[0], t + offset[1]), shadow)

def dotted_line_circles(draw: ImageDraw.ImageDraw, p1, p2, color, dot_r=3, gap=10):
    """True dotted line using circles along a straight segment (supports V/H lines cleanly)."""
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
    """Draw arc + arrow head at end angle (degrees)."""
    draw.arc(bbox, start=start_deg, end=end_deg, fill=color, width=width)
    # Arrow head
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    rx = abs(bbox[2] - bbox[0]) / 2
    ry = abs(bbox[3] - bbox[1]) / 2
    t = math.radians(end_deg)
    tip_x = cx + rx * math.cos(t)
    tip_y = cy + ry * math.sin(t)
    # Tangent direction
    vx = -rx * math.sin(t)
    vy =  ry * math.cos(t)
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L
    # Perp for head width
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

def compose_2x2_grid(choices: List[Image.Image], labels: List[str],
                     pad=18, tile_border=2, bg=(255,255,255),
                     label_spot="tr") -> Image.Image:
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

def stack_vertical(top_img: Image.Image, bottom_img: Image.Image, pad=24, bg=(255,255,255)) -> Image.Image:
    W = max(top_img.width, bottom_img.width) + 2*pad
    H = top_img.height + bottom_img.height + 3*pad
    canvas = Image.new("RGB", (W, H), bg)
    x_top = (W - top_img.width)//2
    canvas.paste(top_img, (x_top, pad))
    x_bot = (W - bottom_img.width)//2
    canvas.paste(bottom_img, (x_bot, top_img.height + 2*pad))
    return canvas

# ----------------------------
# Reference-style problem (Hi-DPI)
# ----------------------------
def draw_problem(direction: str, hole_xy_norm: Tuple[float,float],
                 size=(700, 440), paper_size=(320, 440),
                 bg=(255,255,255), paper_fill=(250,250,250),
                 outline=(20,20,20), fold_line_color=(60,60,60),
                 stroke=4, dpi_scale=2) -> Image.Image:
    # Hi-DPI canvas
    W, H = size[0]*dpi_scale, size[1]*dpi_scale
    pW, pH = paper_size[0]*dpi_scale, paper_size[1]*dpi_scale
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    margin = 22 * dpi_scale
    paper_left = margin
    paper_top = (H - pH) // 2
    paper_rect = (paper_left, paper_top, paper_left + pW, paper_top + pH)

    # Soft shadow
    paper_shadow(img, paper_rect, blur=14, offset=(10*dpi_scale//2, 12*dpi_scale//2), opacity=85)

    # Paper with rounded corners
    rounded_rect(d, paper_rect, radius=16*dpi_scale, fill=paper_fill, outline=outline, width=stroke*dpi_scale)

    # Fold axis & dotted line
    axis, shaded_half = dir_to_axis_and_half(direction)
    l, t, r, b = paper_rect
    cx = (l + r) // 2
    cy = (t + b) // 2
    # Dotted fold line (round dots)
    if axis == "V":
        dotted_line_circles(d, (cx, t + 10*dpi_scale), (cx, b - 10*dpi_scale),
                            fold_line_color, dot_r=3*dpi_scale, gap=13*dpi_scale)
    else:
        dotted_line_circles(d, (l + 10*dpi_scale, cy), (r - 10*dpi_scale, cy),
                            fold_line_color, dot_r=3*dpi_scale, gap=13*dpi_scale)

    # Subtle shaded half (semi-transparent overlay)
    shade = Image.new("RGBA", (r - l, b - t), (0,0,0,0))
    sd = ImageDraw.Draw(shade)
    alpha = 28  # subtle
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

    # Punch (solid circle)
    hx, hy = norm_to_px(hole_xy_norm, paper_rect)
    pr = 11 * dpi_scale
    d.ellipse([hx - pr, hy - pr, hx + pr, hy + pr], fill=outline, outline=outline)

    # Curved arrow (to the right/bottom of paper)
    ac = outline
    if axis == "V":
        arc_left = r + 34*dpi_scale
        arc_right = arc_left + 200*dpi_scale
        arc_top = cy - 110*dpi_scale
        arc_bottom = cy + 110*dpi_scale
        if direction == "L":
            start, end = -20, 210
        else:  # "R"
            start, end = 200, -20
        draw_arc_arrow(d, (arc_left, arc_top, arc_right, arc_bottom), start, end,
                       ac, width=6*dpi_scale, head_len=22*dpi_scale, head_w=14*dpi_scale)
    else:
        arc_left = cx - 110*dpi_scale
        arc_right = cx + 110*dpi_scale
        arc_top = b + 28*dpi_scale
        arc_bottom = arc_top + 200*dpi_scale
        if direction == "U":
            start, end = 160, 342
        else:  # "D"
            start, end = -20, 160
        draw_arc_arrow(d, (arc_left, arc_top, arc_right, arc_bottom), start, end,
                       ac, width=6*dpi_scale, head_len=22*dpi_scale, head_w=14*dpi_scale)

    # Downsample for crisp edges
    if dpi_scale != 1:
        img = img.resize((size[0], size[1]), Image.LANCZOS)
    return img

def draw_unfolded(holes_norm: List[Tuple[float,float]],
                  size=(360, 360), paper_margin=36,
                  bg=(255,255,255), paper_fill=(250,250,250),
                  outline=(20,20,20), stroke=5, dpi_scale=2) -> Image.Image:
    W, H = size[0]*dpi_scale, size[1]*dpi_scale
    m = paper_margin * dpi_scale
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    # Shadow + rounded paper
    left, top, right, bottom = m, m, W - m, H - m
    paper_shadow(img, (left, top, right, bottom), blur=12, offset=(8*dpi_scale//2, 10*dpi_scale//2), opacity=80)
    rounded_rect(d, (left, top, right, bottom), radius=14*dpi_scale,
                 fill=paper_fill, outline=outline, width=stroke*dpi_scale)
    # Holes
    r = 11 * dpi_scale
    for (x, y) in holes_norm:
        px = (x + 1) / 2.0 * (right - left) + left
        py = (1 - (y + 1) / 2.0) * (bottom - top) + top
        d.ellipse([px - r, py - r, px + r, py + r], fill=outline, outline=outline)
    if dpi_scale != 1:
        img = img.resize((size[0], size[1]), Image.LANCZOS)
    return img

# ----------------------------
# Question generator (single fold)
# ----------------------------
def generate_single_fold_question(rng: random.Random, style: Optional[Dict]=None) -> Dict:
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    fold_line_color = (style.get("fold_line") if style else None) or (60, 60, 60)
    stroke = (style.get("stroke_width") if style else None) or 5
    dpi_scale = (style.get("dpi_scale") if style else None) or 2

    # Single direction
    direction = rng.choice(["L", "R", "U", "D"])
    axis, _ = dir_to_axis_and_half(direction)

    # Punch location — keep a safe margin from edges and fold axis
    def safe_rand():
        # Keep at least 0.15 away from edges and 0.12 from axis for clarity
        return rng.uniform(-0.85 + 0.3, 0.85 - 0.3)
    px = safe_rand()
    py = safe_rand()
    # If it's too close to the active axis, nudge it:
    if axis == "V" and abs(px) < 0.12:
        px = 0.12 * (1 if px >= 0 else -1)
    if axis == "H" and abs(py) < 0.12:
        py = 0.12 * (1 if py >= 0 else -1)
    p_folded = (px, py)
    p_reflected = reflect_point(p_folded, axis)

    # Problem image
    problem_img = draw_problem(direction, p_folded, size=(720, 460), paper_size=(340, 460),
                               bg=bg, paper_fill=paper_fill, outline=outline,
                               fold_line_color=fold_line_color, stroke=stroke, dpi_scale=dpi_scale)

    # Choices
    correct = draw_unfolded([p_folded, p_reflected], size=(380, 380),
                            bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke, dpi_scale=dpi_scale)
    # Missed mirror
    d1 = draw_unfolded([p_folded], size=(380, 380),
                       bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke, dpi_scale=dpi_scale)
    # Wrong axis mirror
    wrong_axis = "H" if axis == "V" else "V"
    d2 = draw_unfolded([p_folded, reflect_point(p_folded, wrong_axis)], size=(380, 380),
                       bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke, dpi_scale=dpi_scale)
    # Rotated misconception
    rot = lambda pt: (pt[1], -pt[0])
    d3 = draw_unfolded([rot(p_folded), rot(p_reflected)], size=(380, 380),
                       bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke, dpi_scale=dpi_scale)

    choices = [correct, d1, d2, d3]
    rng.shuffle(choices)
    correct_index = choices.index(correct)

    prompt_ar = "ما رمز البديل الذي يحتوي على صورة الورقة بعد إعادة فتحها من بين البدائل الأربعة؟"
    meta = {
        "type": "folding_single",
        "direction": direction,
        "axis": axis,
        "punch": (round(px, 3), round(py, 3))
    }
    return {
        "problem_img": problem_img,
        "choices_imgs": choices,
        "correct_index": correct_index,
        "prompt": prompt_ar,
        "meta": meta
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Paper Folding — Enhanced", layout="wide")
st.title("Paper Folding — Enhanced Reference Style")

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
    stroke_w = st.number_input("Paper border thickness (px)", min_value=2, max_value=12, value=5, step=1)
    dpi_scale = st.slider("Quality (internal render scale)", min_value=1, max_value=3, value=2)

    st.markdown("---")
    gen_btn = st.button("Generate New Question", type="primary", use_container_width=True)

    st.subheader("Batch")
    batch_n = st.number_input("How many?", min_value=2, max_value=100, value=10, step=1)
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
labels_ar = ["أ", "ب", "ج", "د"]

if qp:
    # Compose grid + stack
    grid = compose_2x2_grid(qp["choices_imgs"], labels_ar, pad=18, tile_border=2, bg=style["bg"], label_spot="tr")
    composite = stack_vertical(qp["problem_img"], grid, pad=24, bg=style["bg"])

    st.subheader("Question")
    st.write(qp["prompt"])
    show_image(composite)

    chosen = st.radio("Pick your answer:", labels_ar, index=0, horizontal=True)
    if st.button("Check answer"):
        if chosen == labels_ar[qp["correct_index"]]:
            st.success(f"Correct. Answer: {labels_ar[qp['correct_index']]}")
        else:
            st.error(f"Not quite. Correct answer: {labels_ar[qp['correct_index']]}")

    st.markdown("---")
    st.subheader("Export")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Separate assets
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG")
        zf.writestr("problem.png", pb.getvalue())
        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG")
            zf.writestr(f"choice_{labels_ar[i]}.png", cb.getvalue())

        # Composite question.png
        qb = io.BytesIO(); composite.save(qb, format="PNG")
        zf.writestr("question.png", qb.getvalue())

        meta = {
            "type": qp["meta"]["type"],
            "direction": qp["meta"]["direction"],
            "axis": qp["meta"]["axis"],
            "punch": qp["meta"]["punch"],
            "prompt": qp["prompt"],
            "labels": labels_ar,
            "correct_label": labels_ar[qp["correct_index"]],
        }
        zf.writestr("question.json", json.dumps(meta, indent=2, ensure_ascii=False))

    st.download_button(
        "Download ZIP",
        data=buf.getvalue(),
        file_name=f"folding_enh_{int(time.time())}.zip",
        mime="application/zip"
    )

    if batch_btn:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(batch_n)):
                local_rng = make_rng((seed or 0) + k + 1)
                local_pack = generate_single_fold_question(local_rng, style=style)
                qid = f"folding_enh_{int(time.time()*1000)}_{k}"

                # Separate images
                pb = io.BytesIO(); local_pack["problem_img"].save(pb, format="PNG")
                zf.writestr(f"{qid}/problem.png", pb.getvalue())
                for i, im in enumerate(local_pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG")
                    zf.writestr(f"{qid}/choice_{labels_ar[i]}.png", cb.getvalue())

                # Composite
                grid = compose_2x2_grid(local_pack["choices_imgs"], labels_ar, pad=18, tile_border=2, bg=style["bg"], label_spot="tr")
                composite = stack_vertical(local_pack["problem_img"], grid, pad=24, bg=style["bg"])
                qb = io.BytesIO(); composite.save(qb, format="PNG")
                zf.writestr(f"{qid}/question.png", qb.getvalue())

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
            file_name=f"batch_folding_enh_{int(time.time())}.zip",
            mime="application/zip"
        )
