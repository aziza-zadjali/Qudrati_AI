# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py
#
# Paper Folding — “Book style” (like the screenshot)
# - Folded inset on the left + dotted empty panel + large paper with fold arrow
# - Landscape paper (like the sample), dotted fold line, shapes are OUTLINES
# - Shapes: circle + triangle (exact vibe of the screenshot)
# - Single fold (L/R/U/D) with correct mirroring
# - 4 choices (2x2), Arabic labels (أ ب ج د)
# - Exports: problem.png, grid.png, question.png, question.json (ensure_ascii=False)
# - Fits container width in Streamlit

import io
import time
import math
import random
import zipfile
import json
from typing import List, Tuple, Optional, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
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
    st.image(img, caption=caption, use_container_width=True)

# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------
def reflect_point(p: Tuple[float, float], axis: str) -> Tuple[float, float]:
    x, y = p
    if axis == "V":
        return (-x, y)
    if axis == "H":
        return (x, -y)
    return (x, y)

def dir_to_axis_and_half(direction: str):
    """
    Returns (axis, shaded_half) where shaded_half in {'left','right','top','bottom'}
    indicates the half that folds over BEFORE opening.
    """
    if direction == "L":
        return "V", "right"   # right folds onto left
    if direction == "R":
        return "V", "left"    # left folds onto right
    if direction == "U":
        return "H", "bottom"  # bottom folds onto top
    return "H", "top"         # "D": top folds onto bottom

def norm_to_px(xy, rect):
    """Map (x,y) in [-1,1] to pixel inside rect=(l,t,r,b) with origin top-left."""
    x, y = xy
    l, t, r, b = rect
    px = (x + 1) / 2.0 * (r - l) + l
    py = (1 - (y + 1) / 2.0) * (b - t) + t
    return px, py

# ------------------------------------------------------------
# Drawing primitives
# ------------------------------------------------------------
def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline, width=3):
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=width)

def paper_shadow(base_img: Image.Image, rect, blur=10, offset=(8, 10), opacity=90):
    l, t, r, b = rect
    w, h = r - l, b - t
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rectangle([0, 0, w, h], fill=(0, 0, 0, opacity))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    base_img.paste(shadow, (l + offset[0], t + offset[1]), shadow)

def dotted_line_circles(draw: ImageDraw.ImageDraw, p1, p2, color, dot_r=3, gap=12):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    steps = max(1, int(length // gap))
    for i in range(steps + 1):
        t = i / steps
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=color, outline=None)

def draw_arc_arrow(draw: ImageDraw.ImageDraw, bbox, start_deg, end_deg, color, width=7, head_len=22, head_w=14):
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

def draw_shape_outline(draw: ImageDraw.ImageDraw, center, size, shape, color, width=6):
    cx, cy = center
    if shape == "circle":
        r = size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)
    elif shape == "triangle":
        s = size * 1.6
        # isosceles, base horizontal
        p1 = (cx, cy - s * 0.9)
        p2 = (cx - s, cy + s * 0.9)
        p3 = (cx + s, cy + s * 0.9)
        draw.line([p1, p2, p3, p1], fill=color, width=width, joint="curve")
    else:
        # default to small square
        s = size * 1.5
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], outline=color, width=width)

def overlay_label_below(tile: Image.Image, label: str, text_color=(20,20,20)) -> Image.Image:
    """Add Arabic label centered **below** the tile (closer to the screenshot style)."""
    font = _try_font(24)
    tw = ImageDraw.Draw(tile).textlength(label, font=font)
    pad = 10
    canvas = Image.new("RGB", (tile.width, tile.height + 40), (255,255,255))
    canvas.paste(tile, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text(((canvas.width - tw)/2, tile.height + pad), label, fill=text_color, font=font)
    return canvas

def compose_2x2_grid(choices, labels, pad=24, tile_border=2, bg=(255,255,255)):
    tiles = []
    for i in range(4):
        t = choices[i].copy()
        d = ImageDraw.Draw(t)
        d.rectangle([3, 3, t.width-3, t.height-3], outline=(20,20,20), width=tile_border)
        tiles.append(overlay_label_below(t, labels[i]))

    w, h = tiles[0].size
    cols, rows = 2, 2
    W = cols * w + (cols + 1) * pad
    H = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    for idx, im in enumerate(tiles):
        r, c = divmod(idx, 2)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(im, (x, y))
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

# ------------------------------------------------------------
# Drawing the example area (folded inset + big paper)
# ------------------------------------------------------------
def draw_example(direction: str,
                 shapes_norm: List[Tuple[str, Tuple[float,float]]],
                 out_width: int,
                 bg=(255,255,255), paper_fill=(250,250,250),
                 outline=(20,20,20), fold_line_color=(60,60,60),
                 stroke=6, dpi_scale=2) -> Image.Image:

    # Layout proportions
    big_w = int(out_width * 0.62)
    inset_w = out_width - big_w
    out_height = int(out_width * 0.38)  # landscape band

    W, H = out_width * dpi_scale, out_height * dpi_scale
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    axis, shaded_half = dir_to_axis_and_half(direction)

    # ---- Left inset (folded + dotted empty half) ----
    inset_pad = int(12 * dpi_scale)
    inset_area = (inset_pad, inset_pad,
                  inset_w * dpi_scale - inset_pad,
                  H - inset_pad)
    il, it, ir, ib = inset_area

    # Decide if inset boxes are side-by-side (V) or stacked (H)
    if axis == "V":
        midx = (il + ir) // 2
        left_box  = (il, it, midx - 6, ib)
        right_box = (midx + 6, it, ir, ib)
        box_with_shapes = left_box if shaded_half == "left" else right_box
        empty_box       = right_box if shaded_half == "left" else left_box
    else:
        midy = (it + ib) // 2
        top_box    = (il, it, ir, midy - 6)
        bottom_box = (il, midy + 6, ir, ib)
        box_with_shapes = top_box if shaded_half == "top" else bottom_box
        empty_box       = bottom_box if shaded_half == "top" else top_box

    # Draw boxes
    rd = 12
    rounded_rect(d, box_with_shapes, rd, fill=paper_fill, outline=outline, width=stroke)
    # empty box dotted border
    dotted = Image.new("RGBA", (ir - il, ib - it), (0,0,0,0))
    dd = ImageDraw.Draw(dotted)
    if axis == "V":
        eb = (0, 0, empty_box[2]-empty_box[0], empty_box[3]-empty_box[1])
        ex, ey = empty_box[0], empty_box[1]
    else:
        eb = (0, 0, empty_box[2]-empty_box[0], empty_box[3]-empty_box[1])
        ex, ey = empty_box[0], empty_box[1]
    # draw dotted rectangle
    x0, y0, x1, y1 = eb
    step = 20
    for x in range(x0+8, x1-8, step):
        dd.line([(x, y0+8), (x+8, y0+8)], fill=fold_line_color, width=3)
        dd.line([(x, y1-8), (x+8, y1-8)], fill=fold_line_color, width=3)
    for y in range(y0+8, y1-8, step):
        dd.line([(x0+8, y), (x0+8, y+8)], fill=fold_line_color, width=3)
        dd.line([(x1-8, y), (x1-8, y+8)], fill=fold_line_color, width=3)
    img.paste(dotted, (ex, ey), dotted)

    # Map shapes (given in full [-1,1] coords on the FOLDING HALF) into the inset half box
    def map_half_to_box(point_xy: Tuple[float,float], bx):
        bl, bt, br, bb = bx
        # determine source range for x or y depending on axis
        x, y = point_xy
        if axis == "V":
            if shaded_half == "left":   # x in [-1, 0]
                x0, x1 = -1.0, 0.0
            else:                       # x in [0, 1]
                x0, x1 = 0.0, 1.0
            X = (x - x0) / (x1 - x0)  # 0..1 in half
            Y = (y + 1) / 2.0         # -1..1 -> 0..1
            px = bl + X * (br - bl)
            py = bt + (1 - Y) * (bb - bt)
        else:
            if shaded_half == "top":    # y in [0, -1] (top half)
                y0, y1 = -1.0, 0.0
            else:                        # bottom half y in [0, 1]
                y0, y1 = 0.0, 1.0
            X = (x + 1) / 2.0
            Y = (y - y0) / (y1 - y0)
            px = bl + X * (br - bl)
            py = bt + (1 - Y) * (bb - bt)
        return px, py

    for shp, (sx, sy) in shapes_norm:
        px, py = map_half_to_box((sx, sy), box_with_shapes)
        draw_shape_outline(d, (px, py), size=12 * dpi_scale, shape=shp, color=outline, width=max(4, stroke-1))

    # ---- Big paper on the right with fold line and arrow ----
    big_l = inset_w * dpi_scale + int(12 * dpi_scale)
    big_r = W - int(12 * dpi_scale)
    big_t = int(10 * dpi_scale)
    big_b = H - int(10 * dpi_scale)

    # inside big area we draw a landscape paper
    margin = int(14 * dpi_scale)
    pl = big_l + margin
    pr = big_r - margin
    # landscape ratio (width:height ~ 2.3:1 like sample)
    p_w = pr - pl
    p_h = int(p_w / 2.3)
    # vertically center
    pt = (H - p_h) // 2
    pb = pt + p_h
    paper_rect = (pl, pt, pr, pb)

    paper_shadow(img, paper_rect, blur=12, offset=(8, 10), opacity=75)
    rounded_rect(d, paper_rect, radius=16, fill=paper_fill, outline=outline, width=stroke)

    # dotted fold line
    l, t, r, b = paper_rect
    cx = (l + r) // 2
    cy = (t + b) // 2
    if axis == "V":
        dotted_line_circles(d, (cx, t + 10), (cx, b - 10), fold_line_color, dot_r=4, gap=16)
        # curved arrow (horizontal arc)
        arc_left, arc_right = r - int(0.32 * (r - l)), r - int(0.04 * (r - l))
        arc_top, arc_bottom = t, b
        if direction == "L":
            start, end = (210, -20)
        else:  # "R"
            start, end = (-20, 210)
        draw_arc_arrow(d, (arc_left, arc_top, arc_right, arc_bottom),
                       start, end, outline, width=7, head_len=22, head_w=14)
    else:
        dotted_line_circles(d, (l + 10, cy), (r - 10, cy), fold_line_color, dot_r=4, gap=16)
        # vertical arc under the paper
        arc_w = int((r - l) * 0.45)
        axl = (l + r)//2 - arc_w//2
        axr = axl + arc_w
        ayt = b + int(0.06*(b-t))
        ayb = ayt + int((b - t) * 0.9)
        if direction == "U":
            start, end = (160, 340)
        else:
            start, end = (-20, 160)
        draw_arc_arrow(d, (axl, ayt, axr, ayb),
                       start, end, outline, width=7, head_len=22, head_w=14)

    # Downsample
    if dpi_scale != 1:
        img = img.resize((out_width, out_height), Image.LANCZOS)
    return img

# ------------------------------------------------------------
# Choices (wide landscape tiles like the sample)
# ------------------------------------------------------------
def draw_choice(shapes_norm: List[Tuple[str, Tuple[float,float]]],
                axis: str,
                tile_size=(560, 260),  # wide tiles
                bg=(255,255,255), paper_fill=(250,250,250),
                outline=(20,20,20), stroke=6, dpi_scale=2) -> Image.Image:

    W, H = tile_size[0]*dpi_scale, tile_size[1]*dpi_scale
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # Paper rectangle (landscape)
    margin = int(0.09 * W)
    l = margin
    r = W - margin
    p_w = r - l
    p_h = int(p_w / 2.3)
    t = max(16, (H - p_h)//2)
    b = t + p_h
    paper_rect = (l, t, r, b)

    paper_shadow(img, paper_rect, blur=10, offset=(6, 8), opacity=75)
    rounded_rect(d, paper_rect, radius=14, fill=paper_fill, outline=outline, width=stroke)

    # Draw shapes as outlines
    for shp, (x, y) in shapes_norm:
        px, py = norm_to_px((x, y), paper_rect)
        draw_shape_outline(d, (px, py), size=10 * dpi_scale, shape=shp, color=outline, width=max(4, stroke-2))

    if dpi_scale != 1:
        img = img.resize(tile_size, Image.LANCZOS)
    return img

# ------------------------------------------------------------
# Question generator
# ------------------------------------------------------------
def generate_single_fold_question(rng: random.Random, style: Optional[Dict]=None) -> Dict:
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    fold_line_color = (style.get("fold_line") if style else None) or (60, 60, 60)
    stroke = (style.get("stroke_width") if style else None) or 6
    dpi_scale = int((style.get("dpi_scale") if style else None) or 2)

    direction = rng.choice(["L", "R", "U", "D"])
    axis, shaded_half = dir_to_axis_and_half(direction)

    # Generate 2 shapes on the folding half (exact vibe of screenshot: circle + triangle)
    def sample_point_on_half():
        # keep good margins from edges and fold axis
        x = rng.uniform(-0.85, 0.85)
        y = rng.uniform(-0.70, 0.70)
        if axis == "V":
            # constrain x to correct half
            if shaded_half == "left":
                x = rng.uniform(-0.85, -0.12)
            else:
                x = rng.uniform(0.12, 0.85)
        else:
            if shaded_half == "top":
                y = rng.uniform(-0.85, -0.12)
            else:
                y = rng.uniform(0.12, 0.85)
        return (round(x, 3), round(y, 3))

    pts = []
    for _ in range(2):
        p = sample_point_on_half()
        # keep some separation
        tries = 0
        while any(math.hypot(p[0]-q[0], p[1]-q[1]) < 0.35 for q in pts) and tries < 20:
            p = sample_point_on_half(); tries += 1
        pts.append(p)

    shapes_half: List[Tuple[str, Tuple[float,float]]] = []
    # ensure one circle + one triangle
    base_shapes = ["circle", "triangle"]
    rng.shuffle(base_shapes)
    shapes_half.append((base_shapes[0], pts[0]))
    shapes_half.append((base_shapes[1], pts[1]))

    # Build correct list after unfolding: mirror across axis — we get 4 shapes total
    mirrored = [(shp, reflect_point(pt, axis)) for (shp, pt) in shapes_half]
    shapes_correct = shapes_half + mirrored

    # WRONG choices:
    # 1) Only originals (no mirror)
    shapes_wrong1 = list(shapes_half)
    # 2) Mirror across wrong axis
    wrong_axis = "H" if axis == "V" else "V"
    shapes_wrong2 = shapes_half + [(shp, reflect_point(pt, wrong_axis)) for (shp, pt) in shapes_half]
    # 3) Swap one shape type after mirroring (triangle<->circle)
    swap = {"circle": "triangle", "triangle": "circle"}
    shapes_wrong3 = shapes_half + [(swap[shp], reflect_point(pt, axis)) for (shp, pt) in shapes_half]

    # Render choices (wide landscape)
    choice_size = (560, 260)
    c0 = draw_choice(shapes_correct, axis, choice_size, bg, paper_fill, outline, stroke, dpi_scale)
    c1 = draw_choice(shapes_wrong1, axis, choice_size, bg, paper_fill, outline, stroke, dpi_scale)
    c2 = draw_choice(shapes_wrong2, axis, choice_size, bg, paper_fill, outline, stroke, dpi_scale)
    c3 = draw_choice(shapes_wrong3, axis, choice_size, bg, paper_fill, outline, stroke, dpi_scale)

    choices = [c0, c1, c2, c3]
    rng.shuffle(choices)
    correct_index = choices.index(c0)

    # Compose grid first to get width, then example to match exactly
    labels_ar = ["أ", "ب", "ج", "د"]
    grid = compose_2x2_grid(choices, labels_ar, pad=24, tile_border=2, bg=bg)
    example = draw_example(direction, shapes_half, out_width=grid.width,
                           bg=bg, paper_fill=paper_fill, outline=outline,
                           fold_line_color=fold_line_color, stroke=stroke, dpi_scale=dpi_scale)

    prompt_ar = "ما رمز البديل الذي يحتوي على صورة الورقة بعد إعادة فتحها من بين البدائل الأربعة؟"
    meta = {
        "type": "folding_single_shapes",
        "direction": direction,
        "axis": axis,
        "half": shaded_half,
        "shapes_half": shapes_half  # stored in normalized coords
    }

    return {
        "problem_img": example,
        "choices_imgs": choices,
        "correct_index": correct_index,
        "labels_ar": labels_ar,
        "prompt": prompt_ar,
        "meta": meta
    }

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Paper Folding — Book Style (Single Fold)", layout="wide")
st.title("Paper Folding — Book Style (Single Fold)")

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
    # Rebuild grid/composite each render so it always matches container width nicely
    grid = compose_2x2_grid(qp["choices_imgs"], qp["labels_ar"], pad=24, tile_border=2, bg=style["bg"])
    composite = stack_vertical(qp["problem_img"], grid, pad=28, bg=style["bg"])

    st.subheader("Question")
    # Exact Arabic text outside images (so RTL is rendered correctly)
    st.write(qp["prompt"])
    show_image(composite)

    chosen = st.radio("Pick your answer:", qp["labels_ar"], index=0, horizontal=True)
    if st.button("Check answer"):
        if chosen == qp["labels_ar"][qp["correct_index"]]:
            st.success(f"إجابة صحيحة: {qp['labels_ar'][qp['correct_index']]}")
        else:
            st.error(f"غير صحيح. الإجابة الصحيحة: {qp['labels_ar'][qp['correct_index']]}")

    st.markdown("---")
    st.subheader("Export")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Save example (problem)
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG")
        zf.writestr("problem.png", pb.getvalue())

        # Save choices (with labels in filenames)
        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG")
            zf.writestr(f"choice_{qp['labels_ar'][i]}.png", cb.getvalue())

        # Save grid and composite
        gb = io.BytesIO(); grid.save(gb, format="PNG")
        zf.writestr("grid.png", gb.getvalue())

        qb = io.BytesIO(); composite.save(qb, format="PNG")
        zf.writestr("question.png", qb.getvalue())

        # Metadata (Arabic preserved)
        meta = {
            "type": qp["meta"]["type"],
            "direction": qp["meta"]["direction"],
            "axis": qp["meta"]["axis"],
            "half": qp["meta"]["half"],
            "shapes_half": qp["meta"]["shapes_half"],
            "prompt": qp["prompt"],
            "labels": qp["labels_ar"],
            "correct_label": qp["labels_ar"][qp["correct_index"]],
        }
        zf.writestr("question.json", json.dumps(meta, indent=2, ensure_ascii=False))

    st.download_button(
        "Download ZIP",
        data=buf.getvalue(),
        file_name=f"folding_bookstyle_{int(time.time())}.zip",
        mime="application/zip"
    )

    if batch_btn:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(batch_n)):
                local_rng = make_rng((seed or 0) + k + 1)
                local_pack = generate_single_fold_question(local_rng, style=style)
                qid = f"folding_bookstyle_{int(time.time()*1000)}_{k}"

                local_grid = compose_2x2_grid(local_pack["choices_imgs"], local_pack["labels_ar"], pad=24, tile_border=2, bg=style["bg"])
                local_composite = stack_vertical(local_pack["problem_img"], local_grid, pad=28, bg=style["bg"])

                pb = io.BytesIO(); local_pack["problem_img"].save(pb, format="PNG")
                zf.writestr(f"{qid}/problem.png", pb.getvalue())

                for i, im in enumerate(local_pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG")
                    zf.writestr(f"{qid}/choice_{local_pack['labels_ar'][i]}.png", cb.getvalue())

                gb = io.BytesIO(); local_grid.save(gb, format="PNG")
                zf.writestr(f"{qid}/grid.png", gb.getvalue())

                qb = io.BytesIO(); local_composite.save(qb, format="PNG")
                zf.writestr(f"{qid}/question.png", qb.getvalue())

                meta = {
                    "id": qid,
                    "type": local_pack["meta"]["type"],
                    "direction": local_pack["meta"]["direction"],
                    "axis": local_pack["meta"]["axis"],
                    "half": local_pack["meta"]["half"],
                    "shapes_half": local_pack["meta"]["shapes_half"],
                    "prompt": local_pack["prompt"],
                    "labels": local_pack["labels_ar"],
                    "correct_label": local_pack["labels_ar"][local_pack["correct_index"]],
                }
                zf.writestr(f"{qid}/question.json", json.dumps(meta, indent=2, ensure_ascii=False))
                index.append(meta)

            zf.writestr("index.json", json.dumps(index, indent=2, ensure_ascii=False))

        st.download_button(
            "Download Batch ZIP",
            data=bbuf.getvalue(),
            file_name=f"batch_folding_bookstyle_{int(time.time())}.zip",
            mime="application/zip"
        )
