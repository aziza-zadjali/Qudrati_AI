# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py
#
# Paper Folding — Canonical Dimensions (Single Fold)
# - All paper panels (example & choices) share one canonical PAPER_W×PAPER_H.
# - Inset on the left (folded half + dotted empty half) uses the SAME paper aspect ratio.
# - Choices are wide landscape tiles; labels below; Arabic prompt outside images.
# - Single fold (L/R/U/D) with correct mirroring.
# - Exports: problem.png, grid.png, question.png, question.json (ensure_ascii=False).
# - Images auto-fit Streamlit container width.

import io, time, math, random, zipfile, json
from typing import List, Tuple, Optional, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------- Utilities ----------------
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
        if len(h) == 3: h = "".join([c*2 for c in h])
        if len(h) != 6: return (255, 255, 255)
        return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
    except Exception:
        return (255, 255, 255)

def show_image(img, caption=None):
    st.image(img, caption=caption, use_container_width=True)

# ---------------- Geometry ----------------
def reflect_point(p: Tuple[float, float], axis: str) -> Tuple[float, float]:
    x, y = p
    return (-x, y) if axis == "V" else (x, -y) if axis == "H" else (x, y)

def dir_to_axis_and_half(direction: str):
    if direction == "L": return "V", "right"
    if direction == "R": return "V", "left"
    if direction == "U": return "H", "bottom"
    return "H", "top"  # "D"

def norm_to_px(xy, rect):
    x, y = xy
    l, t, r, b = rect
    px = (x + 1) / 2.0 * (r - l) + l
    py = (1 - (y + 1) / 2.0) * (b - t) + t
    return px, py

# ---------------- Drawing primitives ----------------
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
    x1, y1 = p1; x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0: return
    steps = max(1, int(length // gap))
    for i in range(steps + 1):
        t = i / steps
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=color)

def draw_arc_arrow(draw: ImageDraw.ImageDraw, bbox, start_deg, end_deg, color, width=7, head_len=22, head_w=14):
    draw.arc(bbox, start=start_deg, end=end_deg, fill=color, width=width)
    cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
    rx = abs(bbox[2] - bbox[0]) / 2; ry = abs(bbox[3] - bbox[1]) / 2
    t = math.radians(end_deg)
    tip_x = cx + rx * math.cos(t); tip_y = cy + ry * math.sin(t)
    vx = -rx * math.sin(t); vy =  ry * math.cos(t)
    L = math.hypot(vx, vy) or 1.0; ux, uy = vx / L, vy / L; px, py = -uy, ux
    base_x = tip_x - ux * head_len; base_y = tip_y - uy * head_len
    p1 = (base_x + px * head_w, base_y + py * head_w)
    p2 = (base_x - px * head_w, base_y - py * head_w)
    draw.polygon([(tip_x, tip_y), p1, p2], fill=color)

def draw_shape_outline(draw: ImageDraw.ImageDraw, center, size, shape, color, width):
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

# --------------- Canonical layout spec ---------------
class CanonSpec:
    # ONE canonical paper size used everywhere in a question.
    PANEL_RATIO = 2.3    # width:height (landscape like your sample)
    PAPER_MARGIN = 0.09  # as fraction of canvas width
    PAPER_STROKE = 6
    CHOICE_TILE = (560, 260)   # canvas for each choice
    DPI = 2

    @classmethod
    def paper_rect_on_canvas(cls, W, H):
        # Respect PANEL_RATIO and center paper
        margin = int(cls.PAPER_MARGIN * W)
        l = margin; r = W - margin
        p_w = r - l
        p_h = int(p_w / cls.PANEL_RATIO)
        t = max(16, (H - p_h)//2); b = t + p_h
        return (l, t, r, b)

# ---------------- Example (uses canonical paper size) ----------------
def draw_example(direction: str,
                 shapes_norm: List[Tuple[str, Tuple[float,float]]],
                 out_width: int,
                 bg=(255,255,255), paper_fill=(250,250,250),
                 outline=(20,20,20), fold_line_color=(60,60,60)) -> Image.Image:

    dpi = CanonSpec.DPI
    # Target: example width == grid width; height based on canonical paper height + arrow room
    # Compute canonical paper height from choice tile
    Wc, Hc = CanonSpec.CHOICE_TILE
    tmp_img = Image.new("RGB", (Wc * dpi, Hc * dpi), bg)
    paper_rect = CanonSpec.paper_rect_on_canvas(tmp_img.width, tmp_img.height)
    paper_h = paper_rect[3] - paper_rect[1]

    out_height = int(paper_h * 1.35 / dpi)   # arrow gap
    W, H = out_width * dpi, out_height * dpi
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    axis, _half = dir_to_axis_and_half(direction)

    # Big paper placed center-right; the paper rectangle must equal canonical size
    # Build a canvas sized like one choice tile and paste into the right side area.
    big_canvas = Image.new("RGB", (Wc * dpi, Hc * dpi), bg)
    bd = ImageDraw.Draw(big_canvas)
    # Canonical paper rect on big_canvas
    pl, pt, pr, pb = CanonSpec.paper_rect_on_canvas(big_canvas.width, big_canvas.height)
    paper_shadow(big_canvas, (pl, pt, pr, pb), blur=12, offset=(8,10), opacity=75)
    rounded_rect(bd, (pl, pt, pr, pb), radius=14, fill=paper_fill, outline=outline, width=CanonSpec.PAPER_STROKE)

    # Fold line + arrow on big paper
    cx = (pl + pr)//2; cy = (pt + pb)//2
    if axis == "V":
        dotted_line_circles(bd, (cx, pt + 10), (cx, pb - 10), fold_line_color, dot_r=4, gap=16)
        arc_left, arc_right = pr - int(0.32 * (pr - pl)), pr - int(0.04 * (pr - pl))
        arc_top, arc_bottom = pt, pb
        start, end = (210, -20) if direction == "L" else (-20, 210)
        draw_arc_arrow(bd, (arc_left, arc_top, arc_right, arc_bottom), start, end, outline, width=7, head_len=22, head_w=14)
    else:
        dotted_line_circles(bd, (pl + 10, cy), (pr - 10, cy), fold_line_color, dot_r=4, gap=16)
        arc_w = int((pr - pl) * 0.45)
        axl = (pl + pr)//2 - arc_w//2; axr = axl + arc_w
        ayt = pb + int(0.06*(pb-pt)); ayb = ayt + int((pb - pt) * 0.9)
        start, end = (160, 340) if direction == "U" else (-20, 160)
        draw_arc_arrow(bd, (axl, ayt, axr, ayb), start, end, outline, width=7, head_len=22, head_w=14)

    # Left inset: two smaller panels, each maintaining SAME aspect ratio as canonical paper.
    inset_w = int(Wc * 0.86) * dpi
    inset_h = big_canvas.height
    inset = Image.new("RGB", (inset_w, inset_h), bg)
    idraw = ImageDraw.Draw(inset)

    def draw_small_paper(box, dotted=False):
        bl, bt, br, bb = box
        inset_canvas = Image.new("RGB", (br - bl, bb - bt), bg)
        icd = ImageDraw.Draw(inset_canvas)
        rpl, rpt, rpr, rpb = CanonSpec.paper_rect_on_canvas(inset_canvas.width, inset_canvas.height)
        rounded_rect(icd, (rpl, rpt, rpr, rpb), radius=12, fill=paper_fill, outline=outline if not dotted else None, width=CanonSpec.PAPER_STROKE)
        if dotted:
            # dotted rectangle outline
            step = 20
            for x in range(rpl+8, rpr-8, step):
                icd.line([(x, rpt+8), (x+8, rpt+8)], fill=fold_line_color, width=3)
                icd.line([(x, rpb-8), (x+8, rpb-8)], fill=fold_line_color, width=3)
            for y in range(rpt+8, rpb-8, step):
                icd.line([(rpl+8, y), (rpl+8, y+8)], fill=fold_line_color, width=3)
                icd.line([(rpr-8, y), (rpr-8, y+8)], fill=fold_line_color, width=3)
        inset.paste(inset_canvas, (bl, bt))

    axis, shaded_half = dir_to_axis_and_half(direction)
    gap = int(12 * dpi)
    if axis == "V":
        half_w = (inset.width - gap) // 2
        left_box  = (0, 0, half_w, inset.height)
        right_box = (half_w + gap, 0, inset.width, inset.height)
        box_with_shapes = left_box if shaded_half == "left" else right_box
        empty_box       = right_box if shaded_half == "left" else left_box
    else:
        half_h = (inset.height - gap) // 2
        top_box    = (0, 0, inset.width, half_h)
        bottom_box = (0, half_h + gap, inset.width, inset.height)
        box_with_shapes = top_box if shaded_half == "top" else bottom_box
        empty_box       = bottom_box if shaded_half == "top" else top_box

    draw_small_paper(box_with_shapes, dotted=False)
    draw_small_paper(empty_box, dotted=True)

    # Draw shapes (circle + triangle) into the "with_shapes" small paper using SAME relative size.
    # Map from normalized to the paper rect inside box_with_shapes:
    def inner_paper_rect_of(box):
        bl, bt, br, bb = box
        w, h = br - bl, bb - bt
        rpl, rpt, rpr, rpb = CanonSpec.paper_rect_on_canvas(w, h)
        return (bl + rpl, bt + rpt, bl + rpr, bt + rpb)

    ipr = inner_paper_rect_of(box_with_shapes)
    s_line = max(4, CanonSpec.PAPER_STROKE - 1)
    for shp, (sx, sy) in shapes_norm:
        px, py = norm_to_px((sx, sy), ipr)
        draw_shape_outline(idraw, (px, py), size=10 * dpi, shape=shp, color=outline, width=s_line)

    # Compose inset + big_canvas onto final band
    # Align to left/right with a consistent gap
    left_x = int(12 * dpi)
    img.paste(inset, (left_x, (H - inset.height)//2))
    right_x = W - big_canvas.width - left_x
    img.paste(big_canvas, (right_x, (H - big_canvas.height)//2))

    if CanonSpec.DPI != 1:
        img = img.resize((out_width, out_height), Image.LANCZOS)
    return img

# ---------------- Choices (canonical size) ----------------
def draw_choice(shapes_norm: List[Tuple[str, Tuple[float,float]]],
                tile_size=None, bg=(255,255,255), paper_fill=(250,250,250),
                outline=(20,20,20)) -> Image.Image:

    tile_size = tile_size or CanonSpec.CHOICE_TILE
    W, H = tile_size[0]*CanonSpec.DPI, tile_size[1]*CanonSpec.DPI
    img = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(img)

    l, t, r, b = CanonSpec.paper_rect_on_canvas(W, H)
    paper_shadow(img, (l, t, r, b), blur=10, offset=(6, 8), opacity=75)
    rounded_rect(d, (l, t, r, b), radius=14, fill=paper_fill, outline=outline, width=CanonSpec.PAPER_STROKE)

    for shp, (x, y) in shapes_norm:
        px, py = norm_to_px((x, y), (l, t, r, b))
        draw_shape_outline(d, (px, py), size=10 * CanonSpec.DPI, shape=shp, color=outline, width=max(4, CanonSpec.PAPER_STROKE-2))

    return img.resize(tile_size, Image.LANCZOS) if CanonSpec.DPI != 1 else img

# ---------------- Grid & stacking ----------------
def overlay_label_below(tile: Image.Image, label: str, text_color=(20,20,20)) -> Image.Image:
    font = _try_font(24)
    tw = ImageDraw.Draw(tile).textlength(label, font=font)
    pad = 10
    canvas = Image.new("RGB", (tile.width, tile.height + 40), (255,255,255))
    canvas.paste(tile, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text(((canvas.width - tw)/2, tile.height + pad), label, fill=text_color, font=font)
    return canvas

def compose_2x2_grid(choices, labels, pad=24, bg=(255,255,255)):
    tiles = []
    for i in range(4):
        t = choices[i].copy()
        d = ImageDraw.Draw(t)
        d.rectangle([3, 3, t.width-3, t.height-3], outline=(20,20,20), width=2)
        tiles.append(overlay_label_below(t, labels[i]))
    w, h = tiles[0].size
    W = 2*w + 3*pad; H = 2*h + 3*pad
    canvas = Image.new("RGB", (W, H), bg)
    canvas.paste(tiles[0], (pad, pad))
    canvas.paste(tiles[1], (2*pad + w, pad))
    canvas.paste(tiles[2], (pad, 2*pad + h))
    canvas.paste(tiles[3], (2*pad + w, 2*pad + h))
    return canvas

def stack_vertical(top_img, bottom_img, pad=28, bg=(255,255,255)):
    W = max(top_img.width, bottom_img.width) + 2*pad
    H = top_img.height + bottom_img.height + 3*pad
    canvas = Image.new("RGB", (W, H), bg)
    canvas.paste(top_img, ((W - top_img.width)//2, pad))
    canvas.paste(bottom_img, ((W - bottom_img.width)//2, top_img.height + 2*pad))
    return canvas

# ---------------- Question generator ----------------
def generate_single_fold_question(rng: random.Random, style: Optional[Dict]=None) -> Dict:
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    fold_line_color = (style.get("fold_line") if style else None) or (60, 60, 60)

    direction = rng.choice(["L", "R", "U", "D"])
    axis, _ = dir_to_axis_and_half(direction)

    # Sample two shapes on the folding half (circle + triangle), spaced apart
    def sample_point_on_half(shaded_half):
        x = rng.uniform(-0.85, 0.85)
        y = rng.uniform(-0.70, 0.70)
        if axis == "V":
            x = rng.uniform(-0.85, -0.12) if shaded_half == "left" else rng.uniform(0.12, 0.85)
        else:
            y = rng.uniform(-0.85, -0.12) if shaded_half == "top" else rng.uniform(0.12, 0.85)
        return (round(x, 3), round(y, 3))

    shaded_half = dir_to_axis_and_half(direction)[1]
    pts = []
    for _ in range(2):
        p = sample_point_on_half(shaded_half); tries=0
        while any(math.hypot(p[0]-q[0], p[1]-q[1]) < 0.35 for q in pts) and tries < 20:
            p = sample_point_on_half(shaded_half); tries += 1
        pts.append(p)

    shapes_half = []
    base_shapes = ["circle", "triangle"]; rng.shuffle(base_shapes)
    shapes_half.append((base_shapes[0], pts[0]))
    shapes_half.append((base_shapes[1], pts[1]))

    mirrored = [(shp, reflect_point(pt, axis)) for (shp, pt) in shapes_half]
    shapes_correct = shapes_half + mirrored

    wrong_axis = "H" if axis == "V" else "V"
    shapes_wrong1 = list(shapes_half)
    shapes_wrong2 = shapes_half + [(shp, reflect_point(pt, wrong_axis)) for (shp, pt) in shapes_half]
    swap = {"circle": "triangle", "triangle": "circle"}
    shapes_wrong3 = shapes_half + [(swap[shp], reflect_point(pt, axis)) for (shp, pt) in shapes_half]

    c0 = draw_choice(shapes_correct, CanonSpec.CHOICE_TILE, bg, paper_fill, outline)
    c1 = draw_choice(shapes_wrong1, CanonSpec.CHOICE_TILE, bg, paper_fill, outline)
    c2 = draw_choice(shapes_wrong2, CanonSpec.CHOICE_TILE, bg, paper_fill, outline)
    c3 = draw_choice(shapes_wrong3, CanonSpec.CHOICE_TILE, bg, paper_fill, outline)

    choices = [c0, c1, c2, c3]; rng.shuffle(choices)
    correct_index = choices.index(c0)

    labels_ar = ["أ", "ب", "ج", "د"]
    grid = compose_2x2_grid(choices, labels_ar, pad=24, bg=bg)
    example = draw_example(direction, shapes_half, out_width=grid.width,
                           bg=bg, paper_fill=paper_fill, outline=outline, fold_line_color=fold_line_color)

    prompt_ar = "ما رمز البديل الذي يحتوي على صورة الورقة بعد إعادة فتحها من بين البدائل الأربعة؟"
    meta = {
        "type": "folding_single_shapes",
        "direction": direction,
        "axis": axis,
        "half": shaded_half,
        "shapes_half": shapes_half
    }
    return {
        "problem_img": example,
        "choices_imgs": choices,
        "correct_index": correct_index,
        "labels_ar": labels_ar,
        "prompt": prompt_ar,
        "meta": meta
    }

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Paper Folding — Canonical Dimensions", layout="wide")
st.title("Paper Folding — Canonical Dimensions (Single Fold)")

with st.sidebar:
    st.header("Controls")
    seed_str = st.text_input("Seed (optional)", value="")
    seed = int(seed_str.strip()) if seed_str.strip().isdigit() else (abs(hash(seed_str)) % (2**31) if seed_str.strip() else None)

    st.header("Visual Style")
    bg_hex = st.text_input("Background", "#FFFFFF")
    paper_hex = st.text_input("Paper fill", "#FAFAFA")
    outline_hex = st.text_input("Outline", "#1A1A1A")
    fold_line_hex = st.text_input("Fold-line (dots)", "#3C3C3C")

    # Advanced (kept fixed internally so dimensions stay consistent)
    st.caption("Dimensions are locked per question so all panels match.")
    gen_btn = st.button("Generate New Question", type="primary", use_container_width=True)

    st.subheader("Batch")
    batch_n = st.number_input("How many?", min_value=2, max_value=100, value=8, step=1)
    batch_btn = st.button("Generate Batch ZIP", use_container_width=True)

style = {
    "bg": _hex_to_rgb(bg_hex),
    "paper_fill": _hex_to_rgb(paper_hex),
    "outline": _hex_to_rgb(outline_hex),
    "fold_line": _hex_to_rgb(fold_line_hex),
}

rng = make_rng(seed)

def make_one():
    return generate_single_fold_question(rng, style=style)

if ("fold_qpack" not in st.session_state) or gen_btn:
    st.session_state.fold_qpack = make_one()

qp = st.session_state.get("fold_qpack")

if qp:
    grid = compose_2x2_grid(qp["choices_imgs"], qp["labels_ar"], pad=24, bg=style["bg"])
    composite = stack_vertical(qp["problem_img"], grid, pad=28, bg=style["bg"])

    st.subheader("Question")
    st.write(qp["prompt"])   # Arabic outside images for proper RTL
    show_image(composite)

    chosen = st.radio("Pick your answer:", qp["labels_ar"], index=0, horizontal=True)
    if st.button("Check answer"):
        correct_label = qp["labels_ar"][qp["correct_index"]]
        st.success(f"الإجابة الصحيحة: {correct_label}") if chosen == correct_label else st.error(f"غير صحيح. الإجابة الصحيحة: {correct_label}")

    st.markdown("---"); st.subheader("Export")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG"); zf.writestr("problem.png", pb.getvalue())
        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG"); zf.writestr(f"choice_{qp['labels_ar'][i]}.png", cb.getvalue())
        gb = io.BytesIO(); grid.save(gb, format="PNG"); zf.writestr("grid.png", gb.getvalue())
        qb = io.BytesIO(); composite.save(qb, format="PNG"); zf.writestr("question.png", qb.getvalue())
        meta = {
            "type": qp["meta"]["type"], "direction": qp["meta"]["direction"],
            "axis": qp["meta"]["axis"], "half": qp["meta"]["half"],
            "shapes_half": qp["meta"]["shapes_half"],
            "prompt": qp["prompt"], "labels": qp["labels_ar"],
            "correct_label": qp["labels_ar"][qp["correct_index"]],
        }
        zf.writestr("question.json", json.dumps(meta, indent=2, ensure_ascii=False))

    st.download_button("Download ZIP", data=buf.getvalue(),
                       file_name=f"folding_canonical_{int(time.time())}.zip",
                       mime="application/zip")

    if batch_btn:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(batch_n)):
                local_rng = make_rng((seed or 0) + k + 1)
                local_pack = generate_single_fold_question(local_rng, style=style)
                qid = f"folding_canonical_{int(time.time()*1000)}_{k}"

                local_grid = compose_2x2_grid(local_pack["choices_imgs"], local_pack["labels_ar"], pad=24, bg=style["bg"])
                local_composite = stack_vertical(local_pack["problem_img"], local_grid, pad=28, bg=style["bg"])

                pb = io.BytesIO(); local_pack["problem_img"].save(pb, format="PNG"); zf.writestr(f"{qid}/problem.png", pb.getvalue())
                for i, im in enumerate(local_pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG"); zf.writestr(f"{qid}/choice_{local_pack['labels_ar'][i]}.png", cb.getvalue())
                gb = io.BytesIO(); local_grid.save(gb, format="PNG"); zf.writestr(f"{qid}/grid.png", gb.getvalue())
                qb = io.BytesIO(); local_composite.save(qb, format="PNG"); zf.writestr(f"{qid}/question.png", qb.getvalue())

                meta = {
                    "id": qid, "type": local_pack["meta"]["type"],
                    "direction": local_pack["meta"]["direction"], "axis": local_pack["meta"]["axis"],
                    "half": local_pack["meta"]["half"], "shapes_half": local_pack["meta"]["shapes_half"],
                    "prompt": local_pack["prompt"], "labels": local_pack["labels_ar"],
                    "correct_label": local_pack["labels_ar"][local_pack["correct_index"]],
                }
                zf.writestr(f"{qid}/question.json", json.dumps(meta, indent=2, ensure_ascii=False))
                index.append(meta)
            zf.writestr("index.json", json.dumps(index, indent=2, ensure_ascii=False))

        st.download_button("Download Batch ZIP", data=bbuf.getvalue(),
                           file_name=f"batch_folding_canonical_{int(time.time())}.zip",
                           mime="application/zip")
