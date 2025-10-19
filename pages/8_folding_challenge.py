# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py
#
# Paper Folding — Two-Panel Example (Single Fold)
# - Horizontal: Bottom → Top (U)  → square paper
# - Vertical:   Right  → Left (L) → landscape paper
# LEFT tile: folding half has dotted outer edges (3 sides); center fold line is SOLID.
#            Shapes are drawn on the VISIBLE (non-dotted) half.
# RIGHT tile: plain paper, center dotted line, big curved arrow that CROSSES the line.
# Choices: the correct option is a TRUE MIRROR of the visible-half shapes across the fold axis.
#          We guarantee uniqueness so one choice is always correct.
# Compact visuals so the page stays tidy.

import io, time, math, random, zipfile, json
from typing import List, Tuple, Optional, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------- Config ----------------
DPI = 2                       # internal render scale for crisp lines
DISPLAY_WIDTH = 680           # page render width for the composite (smaller = more compact)

# ---------------- utils ----------------
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

# normalized coordinate mirror (positions only)
def reflect_point(p: Tuple[float, float], axis: str) -> Tuple[float, float]:
    x, y = p
    return (-x, y) if axis == "V" else (x, -y)

def dir_to_axis_and_half(direction: str):
    if direction == "L":  # right → left
        return "V", "right"
    if direction == "U":  # bottom → top
        return "H", "bottom"
    raise ValueError("Direction must be 'L' or 'U'.")

def norm_to_px(xy, rect):
    x, y = xy
    l, t, r, b = rect
    px = (x + 1) / 2 * (r - l) + l
    py = (1 - (y + 1) / 2) * (b - t) + t
    return px, py

# ---------------- drawing helpers ----------------
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
    L = math.hypot(vx, vy) or 1
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    bx = tip_x - ux * head_len
    by = tip_y - uy * head_len
    draw.polygon([(tip_x, tip_y), (bx + px * head_w, by + py * head_w),
                  (bx - px * head_w, by - py * head_w)], fill=color)

def draw_shape_outline(draw, center, size, shape, color, width):
    cx, cy = center
    if shape == "circle":
        r = size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)
    elif shape == "triangle":
        s = size * 1.6
        p1 = (cx, cy - s * 0.9)
        p2 = (cx - s, cy + s * 0.9)
        p3 = (cx + s, cy + s * 0.9)
        draw.line([p1, p2, p3, p1], fill=color, width=width, joint="curve")
    else:
        s = size * 1.4
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], outline=color, width=width)

# ---------------- adaptive canvas/paper ----------------
def get_tile_size(axis: str) -> Tuple[int, int]:
    """Smaller tiles for a compact page layout."""
    if axis == "V":     # landscape like Example (2)
        return (340, 140)
    else:               # square like Example (1)
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

# --- folded half styling: dotted 3 edges + solid fold line at center ---
def style_folded_edges(draw: ImageDraw.ImageDraw, rect, axis: str, fold_half: str,
                       paper_fill, outline_color, dotted_color, stroke):
    l, t, r, b = rect
    cx = (l + r) // 2
    cy = (t + b) // 2
    erase_w = stroke + 4  # override solid border

    if axis == "H":
        if fold_half == "bottom":
            draw.line([(l, b), (r, b)], fill=paper_fill, width=erase_w)
            draw.line([(l, cy), (l, b)], fill=paper_fill, width=erase_w)
            draw.line([(r, cy), (r, b)], fill=paper_fill, width=erase_w)
            dotted_line_circles(draw, (l, b), (r, b), dotted_color, 3, 12)
            dotted_line_circles(draw, (l, cy), (l, b), dotted_color, 3, 12)
            dotted_line_circles(draw, (r, cy), (r, b), dotted_color, 3, 12)
            draw.line([(l, cy), (r, cy)], fill=outline_color, width=stroke)  # center solid
        else:
            draw.line([(l, t), (r, t)], fill=paper_fill, width=erase_w)
            draw.line([(l, t), (l, cy)], fill=paper_fill, width=erase_w)
            draw.line([(r, t), (r, cy)], fill=paper_fill, width=erase_w)
            dotted_line_circles(draw, (l, t), (r, t), dotted_color, 3, 12)
            dotted_line_circles(draw, (l, t), (l, cy), dotted_color, 3, 12)
            dotted_line_circles(draw, (r, t), (r, cy), dotted_color, 3, 12)
            draw.line([(l, cy), (r, cy)], fill=outline_color, width=stroke)
    else:  # axis == "V"
        if fold_half == "right":
            draw.line([(r, t), (r, b)], fill=paper_fill, width=erase_w)
            draw.line([(cx, t), (r, t)], fill=paper_fill, width=erase_w)
            draw.line([(cx, b), (r, b)], fill=paper_fill, width=erase_w)
            dotted_line_circles(draw, (r, t), (r, b), dotted_color, 3, 12)
            dotted_line_circles(draw, (cx, t), (r, t), dotted_color, 3, 12)
            dotted_line_circles(draw, (cx, b), (r, b), dotted_color, 3, 12)
            draw.line([(cx, t), (cx, b)], fill=outline_color, width=stroke)
        else:
            draw.line([(l, t), (l, b)], fill=paper_fill, width=erase_w)
            draw.line([(l, t), (cx, t)], fill=paper_fill, width=erase_w)
            draw.line([(l, b), (cx, b)], fill=paper_fill, width=erase_w)
            dotted_line_circles(draw, (l, t), (l, b), dotted_color, 3, 12)
            dotted_line_circles(draw, (l, t), (cx, t), dotted_color, 3, 12)
            dotted_line_circles(draw, (l, b), (cx, b), dotted_color, 3, 12)
            draw.line([(cx, t), (cx, b)], fill=outline_color, width=stroke)

# -------------- example (two tiles) ----------------
def draw_example(direction: str,
                 shapes_visible: List[Tuple[str, Tuple[float, float]]],
                 tile_size: Tuple[int, int],
                 ratio: float,
                 bg=(255, 255, 255), paper_fill=(250, 250, 250),
                 outline=(20, 20, 20), fold_line_color=(60, 60, 60)) -> Image.Image:

    TW, TH = [d * DPI for d in tile_size]
    pad = int(16 * DPI)

    # Left tile
    left = Image.new("RGB", (TW, TH), bg)
    pl, pt, pr, pb = paper_rect_on_canvas(TW, TH, ratio)
    ld = ImageDraw.Draw(left)
    paper_shadow(left, (pl, pt, pr, pb))
    rounded_rect(ld, (pl, pt, pr, pb), 14, paper_fill, outline, 6)

    axis, fold_half = dir_to_axis_and_half(direction)
    style_folded_edges(ld, (pl, pt, pr, pb), axis, fold_half,
                       paper_fill, outline, fold_line_color, 6)

    # shapes on the VISIBLE (non-dotted) half
    for shp, (sx, sy) in shapes_visible:
        px, py = norm_to_px((sx, sy), (pl, pt, pr, pb))
        draw_shape_outline(ld, (px, py), size=10 * DPI, shape=shp, color=outline, width=5)

    # Right tile
    right = Image.new("RGB", (TW, TH), bg)
    prl, prt, prr, prb = paper_rect_on_canvas(TW, TH, ratio)
    rd = ImageDraw.Draw(right)
    paper_shadow(right, (prl, prt, prr, prb))
    rounded_rect(rd, (prl, prt, prr, prb), 14, paper_fill, outline, 6)

    cx = (prl + prr) // 2
    cy = (prt + prb) // 2
    Wp = prr - prl
    Hp = prb - prt

    if axis == "V":
        dotted_line_circles(rd, (cx, prt + 10), (cx, prb - 10), fold_line_color, dot_r=4, gap=16)
        # Arrow head pointing LEFT (→ fold right→left).
        aL = prl + int(0.06 * Wp); aR = prr - int(0.06 * Wp)
        aT = prt + int(0.18 * Hp); aB = prt + int(0.78 * Hp)
        draw_arc_arrow(rd, (aL, aT, aR, aB), -30, 180, outline, width=6, head_len=18, head_w=12)
    else:
        dotted_line_circles(rd, (prl + 10, cy), (prr - 10, cy), fold_line_color, dot_r=4, gap=16)
        # Arrow head pointing UP (→ fold bottom→top).
        aL = prl + int(0.12 * Wp); aR = prr - int(0.12 * Wp)
        aT = prt + int(0.06 * Hp); aB = prb - int(0.06 * Hp)
        draw_arc_arrow(rd, (aL, aT, aR, aB), 120, 270, outline, width=6, head_len=18, head_w=12)

    example = Image.new("RGB", (TW * 2 + pad, TH), bg)
    example.paste(left, (0, 0))
    example.paste(right, (TW + pad, 0))

    if DPI != 1:
        example = example.resize((example.width // DPI, example.height // DPI), Image.LANCZOS)
    return example

# -------------- choices (one paper per tile) --------------
def draw_choice(shapes_norm: List[Tuple[str, Tuple[float, float]]],
                tile_size: Tuple[int, int], ratio: float,
                bg=(255, 255, 255), paper_fill=(250, 250, 250),
                outline=(20, 20, 20)) -> Image.Image:
    TW, TH = [d * DPI for d in tile_size]
    img = Image.new("RGB", (TW, TH), bg)
    d = ImageDraw.Draw(img)
    l, t, r, b = paper_rect_on_canvas(TW, TH, ratio)
    paper_shadow(img, (l, t, r, b))
    rounded_rect(d, (l, t, r, b), 14, paper_fill, outline, 6)
    for shp, (x, y) in shapes_norm:
        px, py = norm_to_px((x, y), (l, t, r, b))
        draw_shape_outline(d, (px, py), size=10 * DPI, shape=shp, color=outline, width=4)
    return img.resize(tile_size, Image.LANCZOS) if DPI != 1 else img

# ----- shape list signatures (to ensure uniqueness / correctness) -----
def signature(shapes: List[Tuple[str, Tuple[float,float]]]) -> Tuple:
    # normalized & sorted signature (rounded to 3 decimals)
    norm = [(s, round(p[0],3), round(p[1],3)) for s,p in shapes]
    return tuple(sorted(norm))

# -------------- grid & compose --------------
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

# -------------- generator (guaranteed mirror) --------------
def generate_single_fold_question(rng: random.Random, style: Optional[Dict] = None) -> Dict:
    bg          = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill  = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline     = (style.get("outline") if style else None) or (20, 20, 20)
    fold_color  = (style.get("fold_line") if style else None) or (60, 60, 60)

    # pick fold type
    axis_choice = rng.choice(["V", "H"])
    direction   = "L" if axis_choice == "V" else "U"
    axis, folding_half = dir_to_axis_and_half(direction)

    tile_size = get_tile_size(axis)
    ratio     = 2.3 if axis == "V" else 1.0

    # shapes go on the visible (non-dotted) half
    visible_half = (
        {"left":"right","right":"left"}[folding_half] if axis == "V"
        else {"top":"bottom","bottom":"top"}[folding_half]
    )

    def sample_point_on_half():
        x = rng.uniform(-0.85, 0.85)
        y = rng.uniform(-0.70, 0.70)
        if axis == "V":
            x = rng.uniform(-0.85, -0.12) if visible_half == "left" else rng.uniform(0.12, 0.85)
        else:
            y = rng.uniform(0.12, 0.85) if visible_half == "top" else rng.uniform(-0.85, -0.12)
        return (round(x, 3), round(y, 3))

    # try until we get unique and valid choices
    while True:
        pts = []
        for _ in range(2):
            p = sample_point_on_half(); tries = 0
            while any(math.hypot(p[0]-q[0], p[1]-q[1]) < 0.35 for q in pts) and tries < 40:
                p = sample_point_on_half(); tries += 1
            pts.append(p)

        shapes = ["circle", "triangle"]; rng.shuffle(shapes)
        shapes_visible = [(shapes[0], pts[0]), (shapes[1], pts[1])]

        # build all four answer sets
        correct = shapes_visible + [(s, reflect_point(p, axis)) for s, p in shapes_visible]
        wrong_axis = "H" if axis == "V" else "V"
        wrong1 = list(shapes_visible)
        wrong2 = shapes_visible + [(s, reflect_point(p, wrong_axis)) for s, p in shapes_visible]
        swap = {"circle": "triangle", "triangle": "circle"}
        wrong3 = shapes_visible + [(swap[s], reflect_point(p, axis)) for s, p in shapes_visible]

        sigs = { "C": signature(correct),
                 "A": signature(wrong1),
                 "B": signature(wrong2),
                 "D": signature(wrong3) }

        # ensure uniqueness and correctness (no distractor equals the correct signature)
        if len(set(sigs.values())) == 4:
            all_sets = [correct, wrong1, wrong2, wrong3]
            break

    # draw choices
    cimgs = [draw_choice(s, tile_size, ratio, bg, paper_fill, outline) for s in all_sets]

    # shuffle with our rng and remember where the correct one went
    order = list(range(4))
    rng.shuffle(order)
    choices = [cimgs[i] for i in order]
    correct_index = order.index(0)  # index of "correct" after shuffling

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
        "shapes_visible": shapes_visible
    }
    return {
        "problem_img": example,
        "choices_imgs": choices,
        "correct_index": correct_index,
        "labels_ar": labels_ar,
        "prompt": prompt_ar,
        "meta": meta
    }

# ---------------- Sidebar builder (prevents DeltaGenerator doc spam) ----------------
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

        gen_btn = st.button("Generate New Question", type="primary", use_container_width=True)

        st.subheader("Batch")
        batch_n = st.number_input("How many?", min_value=2, max_value=100, value=8, step=1)
        batch_btn = st.button("Generate Batch ZIP", use_container_width=True)

    return dict(seed=seed, bg_hex=bg_hex, paper_hex=paper_hex, outline_hex=outline_hex,
                fold_line_hex=fold_line_hex, gen_btn=gen_btn, batch_n=batch_n, batch_btn=batch_btn)

# ---------------- Streamlit UI ----------------
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
    st.session_state.fold_qpack = generate_single_fold_question(rng, style=style)

qp = st.session_state.get("fold_qpack")

if qp:
    grid = compose_2x2_grid(qp["choices_imgs"], qp["labels_ar"], pad=18, bg=style["bg"])
    composite = stack_vertical(qp["problem_img"], grid, pad=24, bg=style["bg"])

    st.subheader("Question")
    st.write(qp["prompt"])
    st.image(composite, width=DISPLAY_WIDTH)  # smaller on page

    chosen = st.radio("Pick your answer:", qp["labels_ar"], index=0, horizontal=True)
    if st.button("Check answer"):
        ans = qp["labels_ar"][qp["correct_index"]]
        st.success(f"الإجابة الصحيحة: {ans}") if chosen == ans else st.error(f"غير صحيح. الإجابة الصحيحة: {ans}")

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
                local_rng  = make_rng((ui["seed"] or 0) + k + 1)
                local_pack = generate_single_fold_question(local_rng, style=style)
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
                    "shapes_visible": local_pack["shapes_visible"],
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
