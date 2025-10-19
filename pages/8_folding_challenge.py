# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py

# Paper Folding — Two-Panel Example (Single Fold)
# - Horizontal: Bottom → Top (U) → square paper
# - Vertical:   Right  → Left (L) → landscape paper
# LEFT tile: folding half has dotted outer edges (3 sides); center fold line is SOLID.
# Shapes are drawn on the VISIBLE (non-dotted) half.
# RIGHT tile: plain paper, center dotted line, big curved arrow that CROSSES the line.
# Choices: mirror reflection across the fold axis (after unfolding).
# Export: single or batch .zip

import io, time, math, random, zipfile, json
from typing import List, Tuple, Optional, Dict
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# -------------------------- Config --------------------------
DPI = 2                 # render scale for crisp lines
DISPLAY_WIDTH = 680     # page render width for the composite

# -------------------------- utils --------------------------
def make_rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        seed = int(time.time() * 1000) % 2147483647
    return random.Random(seed)

def _try_font(size: int):
    for n in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except:
            pass
    return ImageFont.load_default()

def _hex_to_rgb(h):
    try:
        h = h.strip().lstrip("#")
        if len(h) == 3:
            h = "".join([c * 2 for c in h])
        if len(h) != 6:
            return (255, 255, 255)
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except:
        return (255, 255, 255)

def dir_to_axis_and_half(direction: str):
    if direction == "L":  # right → left
        return "V", "right"
    if direction == "U":  # bottom → top
        return "H", "bottom"
    raise ValueError("Direction must be 'L' or 'U'.")

def norm_to_px(xy, rect):
    # Map normalized (-1..1) to canvas pixels (rect)
    x, y = xy
    l, t, r, b = rect
    px = (x + 1) / 2 * (r - l) + l
    py = (1 - (y + 1) / 2) * (b - t) + t
    return px, py

# -------------------------- drawing helpers --------------------------
def rounded_rect(draw, box, radius, fill, outline, width=3):
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=width)

def paper_shadow(img, rect, blur=10, offset=(6, 8), opacity=80):
    l, t, r, b = rect
    w, h = r - l, b - t
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(sh)
    d.rectangle([0, 0, w, h], fill=(0, 0, 0, opacity))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    img.paste(sh, (l + offset[0], t + offset[1]), sh)

def dotted_line_circles(draw, p1, p2, color, dot_r=3, gap=12):
    x1, y1 = p1
    x2, y2 = p2
    L = math.hypot(x2 - x1, y2 - y1)
    steps = max(1, int(L // gap))
    for i in range(steps + 1):
        t = i / steps
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=color)

def draw_arc_arrow(draw, bbox, start_deg, end_deg, color, width=6, head_len=18, head_w=12):
    # arc body
    draw.arc(bbox, start=start_deg, end=end_deg, fill=color, width=width)
    # arrow head at end
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    rx = abs(bbox[2] - bbox[0]) / 2
    ry = abs(bbox[3] - bbox[1]) / 2
    t = math.radians(end_deg)
    tip_x = cx + rx * math.cos(t)
    tip_y = cy + ry * math.sin(t)
    vx = -rx * math.sin(t)
    vy =  ry * math.cos(t)
    L = math.hypot(vx, vy) or 1
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    bx = tip_x - ux * head_len
    by = tip_y - uy * head_len
    draw.polygon([(tip_x, tip_y),
                  (bx + px * head_w, by + py * head_w),
                  (bx - px * head_w, by - py * head_w)],
                 fill=color)

def draw_shape_outline(draw, center, size, shape, color, width):
    cx, cy = center
    if shape == "circle":
        r = size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)
    elif shape == "triangle":
        s = size * 1.6
        p1 = (cx,         cy - s * 0.9)
        p2 = (cx - s,     cy + s * 0.9)
        p3 = (cx + s,     cy + s * 0.9)
        draw.line([p1, p2, p3, p1], fill=color, width=width)
    else:
        # square
        s = size * 1.4
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], outline=color, width=width)

# -------------------------- adaptive canvas/paper --------------------------
def get_tile_size(axis: str) -> Tuple[int, int]:
    """Compact tiles for tidy page layout."""
    if axis == "V":    # landscape when vertical fold
        return (340, 140)
    else:              # square when horizontal fold
        return (220, 220)

def paper_rect_on_canvas(W: int, H: int, ratio: float) -> Tuple[int, int, int, int]:
    m = int(0.09 * min(W, H))
    max_w = W - 2 * m
    max_h = H - 2 * m
    pw = max_w
    ph = int(round(pw / ratio))
    if ph > max_h:
        ph = max_h
        pw = int(round(ph * ratio))
    l = (W - pw) // 2
    t = (H - ph) // 2
    return (l, t, l + pw, t + ph)

# -------- folded half styling: dotted 3 edges + solid fold line at center ----
def style_folded_edges(draw: ImageDraw.ImageDraw, rect, axis: str, fold_half: str,
                       paper_fill, outline_color, dotted_color, stroke):
    l, t, r, b = rect
    cx = (l + r) // 2
    cy = (t + b) // 2
    erase_w = stroke + 4  # override solid border

    if axis == "H":
        if fold_half == "bottom":
            # dotted: bottom edge + lower parts of left/right
            draw.line([(l, b), (r, b)], fill=paper_fill, width=erase_w)
            draw.line([(l, cy), (l, b)], fill=paper_fill, width=erase_w)
            draw.line([(r, cy), (r, b)], fill=paper_fill, width=erase_w)
            dotted_line_circles(draw, (l, b), (r, b), dotted_color, 4, 16)
            dotted_line_circles(draw, (l, cy), (l, b), dotted_color, 4, 16)
            dotted_line_circles(draw, (r, cy), (r, b), dotted_color, 4, 16)
            draw.line([(l, cy), (r, cy)], fill=outline_color, width=stroke)  # center solid
        else:
            draw.line([(l, t), (r, t)], fill=paper_fill, width=erase_w)
            draw.line([(l, t), (l, cy)], fill=paper_fill, width=erase_w)
            draw.line([(r, t), (r, cy)], fill=paper_fill, width=erase_w)
            dotted_line_circles(draw, (l, t), (r, t), dotted_color, 4, 16)
            dotted_line_circles(draw, (l, t), (l, cy), dotted_color, 4, 16)
            dotted_line_circles(draw, (r, t), (r, cy), dotted_color, 4, 16)
            draw.line([(l, cy), (r, cy)], fill=outline_color, width=stroke)
    else:  # axis == "V"
        if fold_half == "right":
            draw.line([(r, t), (r, b)], fill=paper_fill, width=erase_w)
            draw.line([(cx, t), (r, t)], fill=paper_fill, width=erase_w)
            draw.line([(cx, b), (r, b)], fill=paper_fill, width=erase_w)
            dotted_line_circles(draw, (r, t), (r, b), dotted_color, 4, 16)
            dotted_line_circles(draw, (cx, t), (r, t), dotted_color, 4, 16)
            dotted_line_circles(draw, (cx, b), (r, b), dotted_color, 4, 16)
            draw.line([(cx, t), (cx, b)], fill=outline_color, width=stroke)
        else:
            draw.line([(l, t), (l, b)], fill=paper_fill, width=erase_w)
            draw.line([(l, t), (cx, t)], fill=paper_fill, width=erase_w)
            draw.line([(l, b), (cx, b)], fill=paper_fill, width=erase_w)
            dotted_line_circles(draw, (l, t), (l, b), dotted_color, 4, 16)
            dotted_line_circles(draw, (l, t), (cx, t), dotted_color, 4, 16)
            dotted_line_circles(draw, (l, b), (cx, b), dotted_color, 4, 16)
            draw.line([(cx, t), (cx, b)], fill=outline_color, width=stroke)

# -------------------------- example (two tiles) --------------------------
def draw_example(direction: str,
                 shapes_visible: List[Tuple[str, Tuple[float, float]]],
                 tile_size: Tuple[int, int],
                 ratio: float,
                 bg=(255, 255, 255), paper_fill=(250, 250, 250),
                 outline=(20, 20, 20), fold_line_color=(60, 60, 60)) -> Image.Image:

    TW, TH = [d * DPI for d in tile_size]
    pad = int(16 * DPI)

    # Left tile: folded reference (folding half dotted; center line solid)
    left = Image.new("RGB", (TW, TH), bg)
    pl, pt, pr, pb = paper_rect_on_canvas(TW, TH, ratio)
    ld = ImageDraw.Draw(left)
    paper_shadow(left, (pl, pt, pr, pb))
    rounded_rect(ld, (pl, pt, pr, pb), 14, paper_fill, outline, 6)

    axis, fold_half = dir_to_axis_and_half(direction)
    style_folded_edges(ld, (pl, pt, pr, pb), axis, fold_half,
                       paper_fill, outline, fold_line_color, 6)

    # shapes on the VISIBLE half
    for shp, (sx, sy) in shapes_visible:
        px, py = norm_to_px((sx, sy), (pl, pt, pr, pb))
        draw_shape_outline(ld, (px, py), size=10 * DPI, shape=shp, color=outline, width=5)

    # Right tile: plain paper with center dotted + arrow that crosses line
    right = Image.new("RGB", (TW, TH), bg)
    prl, prt, prr, prb = paper_rect_on_canvas(TW, TH, ratio)
    rd = ImageDraw.Draw(right)
    paper_shadow(right, (prl, prt, prr, prb))
    rounded_rect(rd, (prl, prt, prr, prb), 14, paper_fill, outline, 6)

    cx = (prl + prr) // 2
    cy = (prt + prb) // 2
    Wp = prr - prl
    Hp = prb - prt

    # stronger crossing: pad the arc box so it clearly crosses the center
    if axis == "V":
        dotted_line_circles(rd, (cx, prt + 10), (cx, prb - 10), fold_line_color, dot_r=5, gap=18)
        aL = prl + int(0.08 * Wp); aR = prr - int(0.04 * Wp)
        aT = prt + int(0.15 * Hp); aB = prt + int(0.85 * Hp)
        draw_arc_arrow(rd, (aL, aT, aR, aB), -25, 180, outline, width=7, head_len=22, head_w=14)
    else:
        dotted_line_circles(rd, (prl + 10, cy), (prr - 10, cy), fold_line_color, dot_r=5, gap=18)
        aL = prl + int(0.12 * Wp); aR = prr - int(0.12 * Wp)
        aT = prt + int(0.08 * Hp); aB = prb - int(0.04 * Hp)
        draw_arc_arrow(rd, (aL, aT, aR, aB), 110, 270, outline, width=7, head_len=22, head_w=14)

    # compose example
    example = Image.new("RGB", (TW * 2 + pad, TH), bg)
    example.paste(left, (0, 0))
    example.paste(right, (TW + pad, 0))
    if DPI != 1:
        example = example.resize((example.width // DPI, example.height // DPI), Image.LANCZOS)
    return example

# -------------------------- choices (one paper per tile) --------------------------
def draw_choice(shapes_px: List[Tuple[str, Tuple[float, float]]],
                tile_size: Tuple[int, int], ratio: float,
                bg=(255, 255, 255), paper_fill=(250, 250, 250),
                outline=(20, 20, 20),
                show_center=False, axis: Optional[str]=None) -> Image.Image:

    TW, TH = [d * DPI for d in tile_size]
    img = Image.new("RGB", (TW, TH), bg)
    d = ImageDraw.Draw(img)
    l, t, r, b = paper_rect_on_canvas(TW, TH, ratio)
    paper_shadow(img, (l, t, r, b))
    rounded_rect(d, (l, t, r, b), 14, paper_fill, outline, 6)

    # optional center guide (QA)
    if show_center and axis:
        if axis == "V":
            cx = (l + r) // 2
            dotted_line_circles(d, (cx, t + 8), (cx, b - 8), (130,130,130), dot_r=4, gap=14)
        else:
            cy = (t + b) // 2
            dotted_line_circles(d, (l + 8, cy), (r - 8, cy), (130,130,130), dot_r=4, gap=14)

    for shp, (px, py) in shapes_px:
        draw_shape_outline(d, (px, py), size=10 * DPI, shape=shp, color=outline, width=4)

    return img.resize(tile_size, Image.LANCZOS) if DPI != 1 else img

# -------------------------- compose helpers --------------------------
def overlay_label_below(tile: Image.Image, label: str, color=(20, 20, 20)) -> Image.Image:
    font = _try_font(22)
    tw = ImageDraw.Draw(tile).textlength(label, font=font)
    canvas = Image.new("RGB", (tile.width, tile.height + 36), (255, 255, 255))
    canvas.paste(tile, (0, 0))
    ImageDraw.Draw(canvas).text(((canvas.width - tw) / 2, tile.height + 8), label, fill=color, font=font)
    return canvas

def compose_2x2_grid(choices, labels, pad=18, bg=(255, 255, 255)):
    tiles = [overlay_label_below(choices[i], labels[i]) for i in range(4)]
    w, h = tiles[0].size
    W = 2 * w + 3 * pad
    H = 2 * h + 3 * pad
    canvas = Image.new("RGB", (W, H), bg)
    canvas.paste(tiles[0], (pad, pad))
    canvas.paste(tiles[1], (2 * pad + w, pad))
    canvas.paste(tiles[2], (pad, 2 * pad + h))
    canvas.paste(tiles[3], (2 * pad + w, 2 * pad + h))
    return canvas

def stack_vertical(top_img, bottom_img, pad=24, bg=(255, 255, 255)):
    W = max(top_img.width, bottom_img.width) + 2 * pad
    H = top_img.height + bottom_img.height + 3 * pad
    canvas = Image.new("RGB", (W, H), bg)
    canvas.paste(top_img, ((W - top_img.width) // 2, pad))
    canvas.paste(bottom_img, ((W - bottom_img.width) // 2, top_img.height + 2 * pad))
    return canvas

# -------------------------- geometry helpers --------------------------
def reflect_points_px(points_px: List[Tuple[str, Tuple[float, float]]],
                      rect, axis: str) -> List[Tuple[str, Tuple[float, float]]]:
    l, t, r, b = rect
    cx = (l + r) / 2
    cy = (t + b) / 2
    out = []
    for shp, (px, py) in points_px:
        if axis == "V":
            out.append((shp, (l + r - px, py)))  # reflect across vertical midline
        else:
            out.append((shp, (px, t + b - py)))  # reflect across horizontal midline
    return out

def normalized_to_px_list(shapes_norm, rect):
    return [(s, norm_to_px(p, rect)) for (s, p) in shapes_norm]

# -------------------------- generator --------------------------
def generate_single_fold_question(rng: random.Random,
                                  style: Optional[Dict] = None,
                                  include_original_on_unfold: bool = True,
                                  show_center_in_choices: bool = False) -> Dict:
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    fold_color = (style.get("fold_line") if style else None) or (60, 60, 60)

    # Random orientation
    axis_choice = rng.choice(["V", "H"])
    direction = "L" if axis_choice == "V" else "U"
    axis, folding_half = dir_to_axis_and_half(direction)

    # Tile size & paper ratio by axis
    tile_size = get_tile_size(axis)
    ratio = 2.3 if axis == "V" else 1.0

    # Where shapes are allowed (visible half only in the example)
    visible_half = (
        {"left": "right", "right": "left"}[folding_half] if axis == "V"
        else {"top": "bottom", "bottom": "top"}[folding_half]
    )

    # Safety margins so shapes are never near the fold line
    AXIS_MARGIN = 0.18  # keep shapes away from the fold axis in normalized space
    EDGE_MARGIN = 0.12

    def sample_point_on_half():
        # normalized sampling with axis margin
        x = rng.uniform(-0.85, 0.85)
        y = rng.uniform(-0.70, 0.70)
        if axis == "V":
            if visible_half == "left":
                x = rng.uniform(-0.85, -AXIS_MARGIN)
            else:
                x = rng.uniform(AXIS_MARGIN, 0.85)
            # keep decent vertical margin
            y = rng.uniform(-0.70, 0.70)
        else:
            if visible_half == "top":
                y = rng.uniform(AXIS_MARGIN, 0.85)
            else:
                y = rng.uniform(-0.85, -AXIS_MARGIN)
            x = rng.uniform(-0.85, 0.85)
        # edge margin clamp
        x = max(-1 + EDGE_MARGIN, min(1 - EDGE_MARGIN, x))
        y = max(-1 + EDGE_MARGIN, min(1 - EDGE_MARGIN, y))
        return (round(x, 3), round(y, 3))

    # Two shapes, spaced apart
    pts = []
    for _ in range(2):
        p = sample_point_on_half(); tries = 0
        while any(math.hypot(p[0]-q[0], p[1]-q[1]) < 0.40 for q in pts) and tries < 50:
            p = sample_point_on_half(); tries += 1
        pts.append(p)

    shapes = ["circle", "triangle"]; rng.shuffle(shapes)
    shapes_visible = [(shapes[0], pts[0]), (shapes[1], pts[1])]

    # Build all four choice panels in PIXEL space to avoid rounding drift
    TW, TH = [d * DPI for d in tile_size]
    l, t, r, b = paper_rect_on_canvas(TW, TH, ratio)

    # Visible shapes in pixels
    shapes_visible_px = normalized_to_px_list(shapes_visible, (l, t, r, b))
    # Mirrored (correct) in pixels
    mirrored_px = reflect_points_px(shapes_visible_px, (l, t, r, b), axis)

    if include_original_on_unfold:
        shapes_correct_px = shapes_visible_px + mirrored_px
    else:
        shapes_correct_px = mirrored_px

    # Distractors:
    wrong_axis = "H" if axis == "V" else "V"

    # (1) No reflection (just originals)
    shapes_wrong1_px = list(shapes_visible_px)

    # (2) Wrong-axis reflection (with originals)
    shapes_wrong2_px = shapes_visible_px + reflect_points_px(shapes_visible_px, (l, t, r, b), wrong_axis)

    # (3) Swap shape type on the mirrored half (correct axis)
    swap = {"circle": "triangle", "triangle": "circle"}
    swapped_mirror_px = [(swap[s], pt) for (s, pt) in reflect_points_px(shapes_visible_px, (l, t, r, b), axis)]
    shapes_wrong3_px = shapes_visible_px + swapped_mirror_px if include_original_on_unfold else swapped_mirror_px

    # Render choices
    c0 = draw_choice(shapes_correct_px, tile_size, ratio, bg, paper_fill, outline,
                     show_center=show_center_in_choices, axis=axis)
    c1 = draw_choice(shapes_wrong1_px, tile_size, ratio, bg, paper_fill, outline,
                     show_center=show_center_in_choices, axis=axis)
    c2 = draw_choice(shapes_wrong2_px, tile_size, ratio, bg, paper_fill, outline,
                     show_center=show_center_in_choices, axis=axis)
    c3 = draw_choice(shapes_wrong3_px, tile_size, ratio, bg, paper_fill, outline,
                     show_center=show_center_in_choices, axis=axis)

    choices = [c0, c1, c2, c3]; random.shuffle(choices); correct_index = choices.index(c0)

    labels_ar = ["أ", "ب", "ج", "د"]
    grid = compose_2x2_grid(choices, labels_ar, pad=18, bg=bg)
    example = draw_example(direction, shapes_visible, tile_size, ratio, bg, paper_fill, outline, fold_color)

    prompt_ar = "ما رمز البديل الذي يحتوي على صورة الورقة بعد إعادة فتحها من بين البدائل الأربعة؟"
    meta = {
        "type": "folding_single_shapes",
        "direction": direction,
        "axis": axis,
        "folding_half": folding_half,
        "visible_half": visible_half,
        "ratio": ratio,
        "tile_size": tile_size,
        "shapes_visible": shapes_visible,
        "include_original_on_unfold": include_original_on_unfold
    }
    return {
        "problem_img": example,
        "choices_imgs": choices,
        "correct_index": correct_index,
        "labels_ar": labels_ar,
        "prompt": prompt_ar,
        "meta": meta
    }

# -------------------------- Sidebar builder --------------------------
def build_sidebar():
    sb = st.sidebar
    with sb:
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

        st.header("Behavior")
        include_original = st.toggle("Show originals on unfolded page (double-print)", value=True,
                                     help="If off, answers show only the mirrored half (transfer-only).")
        show_center = st.toggle("Show center guide in answers (QA)", value=False)

        gen_btn = st.button("Generate New Question", type="primary", use_container_width=True)

        st.subheader("Batch")
        batch_n = st.number_input("How many?", min_value=2, max_value=100, value=8, step=1)
        batch_btn = st.button("Generate Batch ZIP", use_container_width=True)

    return dict(seed=seed, bg_hex=bg_hex, paper_hex=paper_hex, outline_hex=outline_hex,
                fold_line_hex=fold_line_hex, gen_btn=gen_btn, batch_n=batch_n, batch_btn=batch_btn,
                include_original=include_original, show_center=show_center)

# -------------------------- Streamlit UI --------------------------
st.set_page_config(page_title="Paper Folding — Two-Panel Example", layout="wide")
st.title("Paper Folding — Two-Panel Example (Single Fold)")

ui = build_sidebar()
style = {
    "bg": _hex_to_rgb(ui["bg_hex"]),
    "paper_fill": _hex_to_rgb(ui["paper_hex"]),
    "outline": _hex_to_rgb(ui["outline_hex"]),
    "fold_line": _hex_to_rgb(ui["fold_line_hex"]),
}
rng = make_rng(ui["seed"])

if ("fold_qpack" not in st.session_state) or ui["gen_btn"]:
    st.session_state.fold_qpack = generate_single_fold_question(
        rng, style=style,
        include_original_on_unfold=ui["include_original"],
        show_center_in_choices=ui["show_center"]
    )

qp = st.session_state.get("fold_qpack")
if qp:
    grid = compose_2x2_grid(qp["choices_imgs"], qp["labels_ar"], pad=18, bg=style["bg"])
    composite = stack_vertical(qp["problem_img"], grid, pad=24, bg=style["bg"])
    st.subheader("Question")
    st.write(qp["prompt"])
    st.image(composite, width=DISPLAY_WIDTH)

    chosen = st.radio("Pick your answer:", qp["labels_ar"], index=0, horizontal=True)
    if st.button("Check answer"):
        ans = qp["labels_ar"][qp["correct_index"]]
        if chosen == ans:
            st.success(f"الإجابة الصحيحة: {ans}")
        else:
            st.error(f"غير صحيح. الإجابة الصحيحة: {ans}")

    st.markdown("---")
    st.subheader("Export")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG")
        zf.writestr("problem.png", pb.getvalue())
        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG")
            zf.writestr(f"choice_{qp['labels_ar'][i]}.png", cb.getvalue())
        gb = io.BytesIO(); grid.save(gb, format="PNG")
        zf.writestr("grid.png", gb.getvalue())
        qb = io.BytesIO(); composite.save(qb, format="PNG")
        zf.writestr("question.png", qb.getvalue())
        meta = {
            "type": qp["meta"]["type"],
            "direction": qp["meta"]["direction"],
            "axis": qp["meta"]["axis"],
            "folding_half": qp["meta"]["folding_half"],
            "visible_half": qp["meta"]["visible_half"],
            "ratio": qp["meta"]["ratio"],
            "tile_size": qp["meta"]["tile_size"],
            "shapes_visible": qp["meta"]["shapes_visible"],
            "include_original_on_unfold": qp["meta"]["include_original_on_unfold"],
            "prompt": qp["prompt"],
            "labels": qp["labels_ar"],
            "correct_label": qp["labels_ar"][qp["correct_index"]],
        }
        zf.writestr("question.json", json.dumps(meta, indent=2, ensure_ascii=False))
    st.download_button(
        "Download ZIP",
        data=buf.getvalue(),
        file_name=f"folding_two_panel_{int(time.time())}.zip",
        mime="application/zip"
    )

    if ui["batch_btn"]:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(ui["batch_n"])):
                local_rng = make_rng((ui["seed"] or 0) + k + 1)
                local_pack = generate_single_fold_question(
                    local_rng, style=style,
                    include_original_on_unfold=ui["include_original"],
                    show_center_in_choices=ui["show_center"]
                )
                qid = f"folding_two_panel_{int(time.time()*1000)}_{k}"
                local_grid = compose_2x2_grid(local_pack["choices_imgs"], local_pack["labels_ar"], pad=18, bg=style["bg"])
                local_comp = stack_vertical(local_pack["problem_img"], local_grid, pad=24, bg=style["bg"])

                pb = io.BytesIO(); local_pack["problem_img"].save(pb, format="PNG")
                zf.writestr(f"{qid}/problem.png", pb.getvalue())
                for i, im in enumerate(local_pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG")
                    zf.writestr(f"{qid}/choice_{local_pack['labels_ar'][i]}.png", cb.getvalue())
                gb = io.BytesIO(); local_grid.save(gb, format="PNG")
                zf.writestr(f"{qid}/grid.png", gb.getvalue())
                qb = io.BytesIO(); local_comp.save(qb, format="PNG")
                zf.writestr(f"{qid}/question.png", qb.getvalue())

                meta = {
                    "id": qid,
                    "type": local_pack["meta"]["type"],
                    "direction": local_pack["meta"]["direction"],
                    "axis": local_pack["meta"]["axis"],
                    "folding_half": local_pack["meta"]["folding_half"],
                    "visible_half": local_pack["meta"]["visible_half"],
                    "ratio": local_pack["meta"]["ratio"],
                    "tile_size": local_pack["meta"]["tile_size"],
                    "shapes_visible": local_pack["meta"]["shapes_visible"],
                    "include_original_on_unfold": local_pack["meta"]["include_original_on_unfold"],
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
            file_name=f"batch_folding_two_panel_{int(time.time())}.zip",
            mime="application/zip"
        )
