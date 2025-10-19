# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py
# Single-fold "paper folding" question in the reference style:
# - Portrait rectangular paper
# - Dotted fold line at the center
# - Curved arrow indicating fold/unfold direction
# - Solid circle for the punch (no "Punch" text)
# - 2x2 options with Arabic labels inside tiles
# - Exports composite question.png and JSON with Arabic preserved

import io
import time
import math
import random
import zipfile
import json
from typing import List, Tuple, Optional, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ----------------------------
# Basic utils
# ----------------------------
def make_rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        seed = int(time.time() * 1000) % 2147483647
    return random.Random(seed)

def _load_font(font_size: int):
    # Try DejaVu (usually present), then Arial, else default
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        try:
            return ImageFont.truetype("Arial.ttf", font_size)
        except Exception:
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
# Geometry & transforms (single fold)
# ----------------------------
def reflect_point_single_fold(p: Tuple[float, float], axis: str) -> Tuple[float, float]:
    """Reflect across axis: 'V' (vertical x=0) or 'H' (horizontal y=0)."""
    x, y = p
    if axis == "V":
        return (-x, y)
    if axis == "H":
        return (x, -y)
    return (x, y)

def fold_dir_to_axis_and_shaded_half(direction: str):
    """
    Returns (axis, shaded_half) where shaded_half in {'left','right','top','bottom'}
    indicates the half that folds over the other BEFORE punching.
    """
    if direction == "L":   # fold leftwards: right half folds onto left
        return "V", "right"
    if direction == "R":   # fold rightwards: left half folds onto right
        return "V", "left"
    if direction == "U":   # fold upwards: bottom half folds onto top
        return "H", "bottom"
    # "D"   # fold downwards: top half folds onto bottom
    return "H", "top"

# ----------------------------
# Drawing helpers (reference look)
# ----------------------------
def draw_dotted_line(d: ImageDraw.ImageDraw, p1, p2, color, dash_len=8, gap_len=8, width=3):
    """Simple dashed line between p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    dist = 0.0
    on = True
    while dist < length:
        seg = dash_len if on else gap_len
        seg = min(seg, length - dist)
        x_start = x1 + ux * dist
        y_start = y1 + uy * dist
        x_end = x1 + ux * (dist + seg) if on else x_start
        y_end = y1 + uy * (dist + seg) if on else y_start
        if on:
            d.line([(x_start, y_start), (x_end, y_end)], fill=color, width=width)
        on = not on
        dist += seg

def draw_arc_arrow(d: ImageDraw.ImageDraw, bbox, start_deg, end_deg, color, width=5, head_len=16, head_w=10):
    """
    Draw an arc with an arrow head at the end angle.
    bbox: (left, top, right, bottom) ellipse bounding box
    angles in degrees (PIL uses degrees, clockwise from +x axis)
    """
    # Draw arc
    d.arc(bbox, start=start_deg, end=end_deg, fill=color, width=width)

    # Compute arrow tip point and tangent direction at end angle
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    rx = abs(bbox[2] - bbox[0]) / 2
    ry = abs(bbox[3] - bbox[1]) / 2
    t = math.radians(end_deg)
    # Point on ellipse
    tx = cx + rx * math.cos(t)
    ty = cy + ry * math.sin(t)
    # Tangent direction (derivative): (-rx*sin t, ry*cos t)
    vx = -rx * math.sin(t)
    vy =  ry * math.cos(t)
    # Normalize
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L

    # Build arrow head triangle
    # Perp vector
    px, py = -uy, ux
    tip = (tx, ty)
    base = (tx - ux * head_len, ty - uy * head_len)
    p1 = (base[0] + px * head_w, base[1] + py * head_w)
    p2 = (base[0] - px * head_w, base[1] - py * head_w)
    d.polygon([tip, p1, p2], fill=color)

def norm_to_px(xy, rect):
    """Map (x,y) in [-1,1] to pixel inside rect=(left,top,right,bottom) with (0,0) at top-left."""
    x, y = xy
    left, top, right, bottom = rect
    px = (x + 1) / 2.0 * (right - left) + left
    py = (1 - (y + 1) / 2.0) * (bottom - top) + top
    return px, py

def overlay_label(img: Image.Image, label: str, spot="tr",
                  circle_fill=(255,255,255), circle_outline=(20,20,20),
                  text_color=(20,20,20)) -> Image.Image:
    """Draw a small labeled circle (أ/ب/ج/د) inside the image."""
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
    else:  # "br"
        cx, cy = im.width - margin - r, im.height - margin - r
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=circle_fill, outline=circle_outline, width=2)
    try:
        font = _load_font(18)
    except Exception:
        font = ImageFont.load_default()
    # Rough centering
    tw = d.textlength(label, font=font)
    d.text((cx - tw/2, cy - 9), label, fill=text_color, font=font)
    return im

def compose_2x2_grid(choices: List[Image.Image], labels: List[str],
                     pad=18, tile_border=2, bg=(255,255,255),
                     label_spot="tr") -> Image.Image:
    """Return a 2x2 grid image with labeled tiles."""
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

def stack_vertical(top_img: Image.Image, bottom_img: Image.Image, pad=22, bg=(255,255,255)) -> Image.Image:
    W = max(top_img.width, bottom_img.width) + 2*pad
    H = top_img.height + bottom_img.height + 3*pad
    canvas = Image.new("RGB", (W, H), bg)
    x_top = (W - top_img.width)//2
    canvas.paste(top_img, (x_top, pad))
    x_bot = (W - bottom_img.width)//2
    canvas.paste(bottom_img, (x_bot, top_img.height + 2*pad))
    return canvas

# ----------------------------
# Reference-style visuals
# ----------------------------
def draw_reference_problem(direction: str, hole_xy_norm: Tuple[float,float],
                           size=(660, 380),
                           paper_size=(300, 420),
                           bg=(255,255,255), paper_fill=(250,250,250),
                           outline=(20,20,20), fold_line_color=(60,60,60),
                           stroke=4) -> Image.Image:
    """
    Create the top "problem" image that matches the reference:
    - Portrait paper on the left with dotted fold-line at center
    - Shaded folding half
    - Solid circle hole
    - Large curved arrow indicating fold direction
    """
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)

    # Place the paper on the left
    margin = 22
    paper_W, paper_H = paper_size
    paper_left = margin
    paper_top = (H - paper_H) // 2
    paper_rect = (paper_left, paper_top, paper_left + paper_W, paper_top + paper_H)

    # Paper
    d.rectangle(paper_rect, outline=outline, width=stroke, fill=paper_fill)

    # Fold axis & shaded half
    axis, shaded_half = fold_dir_to_axis_and_shaded_half(direction)
    left, top, right, bottom = paper_rect
    cx = (left + right) // 2
    cy = (top + bottom) // 2

    # Dotted fold line
    if axis == "V":
        draw_dotted_line(d, (cx, top+8), (cx, bottom-8), fold_line_color, dash_len=10, gap_len=8, width=3)
        # Shade the half that folds over
        if shaded_half == "left":
            shade_rect = (left+2, top+2, cx-2, bottom-2)
        else:
            shade_rect = (cx+2, top+2, right-2, bottom-2)
    else:
        draw_dotted_line(d, (left+8, cy), (right-8, cy), fold_line_color, dash_len=10, gap_len=8, width=3)
        if shaded_half == "top":
            shade_rect = (left+2, top+2, right-2, cy-2)
        else:
            shade_rect = (left+2, cy+2, right-2, bottom-2)

    # Semi-transparent shading effect via hatch (simple)
    hatch = Image.new("RGBA", (paper_W-4, paper_H-4), (0,0,0,0))
    hd = ImageDraw.Draw(hatch)
    # diagonal light hatch
    for x in range(-paper_H, paper_W, 12):
        hd.line([(x, 0), (x+paper_H, paper_H)], fill=(0,0,0,28), width=2)
    # paste only over the shaded half
    sx1, sy1, sx2, sy2 = shade_rect
    mask = Image.new("L", (paper_W, paper_H), 0)
    md = ImageDraw.Draw(mask)
    md.rectangle((sx1-left, sy1-top, sx2-left, sy2-top), fill=190)
    img.paste(hatch, (left+2, top+2), mask.crop((2,2,paper_W-2,paper_H-2)))

    # Hole (solid circle)
    hx, hy = norm_to_px(hole_xy_norm, paper_rect)
    r = 10
    d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=outline, outline=outline)

    # Big curved arrow on the right, indicating the fold direction
    # We'll place an ellipse to the right side of the paper and draw an arc with an arrow head.
    arrow_color = outline
    if axis == "V":
        # Vertical fold -> arrow that curves horizontally around the mid-height
        # Arc bbox right of paper
        arc_left = right + 30
        arc_right = arc_left + 180
        arc_top = cy - 90
        arc_bottom = cy + 90
        if direction == "L":
            # arrow heads pointing leftwards at ~180 deg
            start, end = -20, 200  # sweeping counter-clockwise
        else:  # "R"
            # arrow heads pointing rightwards at ~0 deg
            start, end = 200, -20
        draw_arc_arrow(d, (arc_left, arc_top, arc_right, arc_bottom), start, end, arrow_color, width=6, head_len=18, head_w=12)
    else:
        # Horizontal fold -> arrow that curves vertically around the mid-width
        arc_left = cx - 90
        arc_right = cx + 90
        arc_top = bottom + 20
        arc_bottom = arc_top + 180
        if direction == "U":
            # arrow pointing up at ~90 deg
            start, end = 160, 340
        else:  # "D"
            # arrow pointing down at ~270 deg
            start, end = -20, 160
        draw_arc_arrow(d, (arc_left, arc_top, arc_right, arc_bottom), start, end, arrow_color, width=6, head_len=18, head_w=12)

    return img

def draw_unfolded_paper_with_holes(holes_norm: List[Tuple[float,float]],
                                   size=(420, 420), paper_margin=40,
                                   bg=(255,255,255), paper_fill=(250,250,250),
                                   outline=(20,20,20), stroke=5) -> Image.Image:
    """Final unfolded paper with all hole positions."""
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    left, top, right, bottom = paper_margin, paper_margin, W - paper_margin, H - paper_margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    r = 11
    for (x, y) in holes_norm:
        px = (x + 1) / 2.0 * (right - left) + left
        py = (1 - (y + 1) / 2.0) * (bottom - top) + top
        d.ellipse([px - r, py - r, px + r, py + r], fill=outline, outline=outline)
    return img

# ----------------------------
# Question generator (single fold)
# ----------------------------
def generate_single_fold_question(rng: random.Random,
                                  style: Optional[Dict] = None) -> Dict:
    """
    Returns:
      problem_img: reference-style prompt image
      choices_imgs: list of 4 options (correct + 3 distractors)
      correct_index: index of the correct option
      prompt: Arabic prompt text (fixed)
      meta: dict with fold direction, axis, punch point
    """
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    fold_line_color = (style.get("fold_line") if style else None) or (60, 60, 60)
    stroke = (style.get("stroke_width") if style else None) or 5

    # Direction: single fold
    direction = rng.choice(["L", "R", "U", "D"])
    axis, _ = fold_dir_to_axis_and_shaded_half(direction)

    # Pick a punch location on the folded stack; after unfolding we mirror across axis
    px = rng.uniform(-0.65, 0.65)
    py = rng.uniform(-0.65, 0.65)
    p_folded = (px, py)
    p_reflected = reflect_point_single_fold(p_folded, axis)

    # Problem image (reference look)
    problem_img = draw_reference_problem(
        direction, p_folded, size=(660, 420), paper_size=(300, 420),
        bg=bg, paper_fill=paper_fill, outline=outline, fold_line_color=fold_line_color, stroke=stroke
    )

    # Build choices:
    # Correct: both holes after unfolding
    correct = draw_unfolded_paper_with_holes([p_folded, p_reflected], bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    # Distractor A: only one hole (forgets mirroring)
    d1 = draw_unfolded_paper_with_holes([p_folded], bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    # Distractor B: mirror across the WRONG axis
    wrong_axis = "H" if axis == "V" else "V"
    wrong_reflect = reflect_point_single_fold(p_folded, wrong_axis)
    d2 = draw_unfolded_paper_with_holes([p_folded, wrong_reflect], bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    # Distractor C: rotate 90° (common misconception)
    # Rotation in normalized coords: (x, y) -> (y, -x)
    rot = lambda pt: (pt[1], -pt[0])
    d3 = draw_unfolded_paper_with_holes([rot(p_folded), rot(p_reflected)], bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

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
# Streamlit UI (English; Arabic prompt preserved)
# ----------------------------
st.set_page_config(page_title="Paper Folding (Reference Style)", layout="wide")
st.title("Paper Folding — Reference Style (Single Fold)")

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
    outline_hex = st.text_input("Outline", "#141414")
    fold_line_hex = st.text_input("Fold-line (dotted)", "#3C3C3C")
    stroke_w = st.number_input("Border thickness (px)", min_value=2, max_value=12, value=5, step=1)

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
}

rng = make_rng(seed)

def make_one():
    return generate_single_fold_question(rng, style=style)

if ("fold_qpack" not in st.session_state) or gen_btn:
    st.session_state.fold_qpack = make_one()

qp = st.session_state.get("fold_qpack")
labels_ar = ["أ", "ب", "ج", "د"]

if qp:
    # Compose 2x2 grid with labels inside the tiles
    grid = compose_2x2_grid(qp["choices_imgs"], labels_ar, pad=18, tile_border=2, bg=style["bg"], label_spot="tr")
    composite = stack_vertical(qp["problem_img"], grid, pad=22, bg=style["bg"])

    # Arabic prompt (exact)
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
        # separate images
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG")
        zf.writestr("problem.png", pb.getvalue())
        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG")
            zf.writestr(f"choice_{labels_ar[i]}.png", cb.getvalue())

        # composite question.png
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
        file_name=f"folding_ref_{int(time.time())}.zip",
        mime="application/zip"
    )

    if batch_btn:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(batch_n)):
                local_rng = make_rng((seed or 0) + k + 1)
                local_pack = generate_single_fold_question(local_rng, style=style)
                qid = f"folding_ref_{int(time.time()*1000)}_{k}"

                # separate images
                pb = io.BytesIO(); local_pack["problem_img"].save(pb, format="PNG")
                zf.writestr(f"{qid}/problem.png", pb.getvalue())
                for i, im in enumerate(local_pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG")
                    zf.writestr(f"{qid}/choice_{labels_ar[i]}.png", cb.getvalue())

                # composite question.png
                grid = compose_2x2_grid(local_pack["choices_imgs"], labels_ar, pad=18, tile_border=2, bg=style["bg"], label_spot="tr")
                composite = stack_vertical(local_pack["problem_img"], grid, pad=22, bg=style["bg"])
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
            file_name=f"batch_folding_ref_{int(time.time())}.zip",
            mime="application/zip"
        )
