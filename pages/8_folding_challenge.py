# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py
#
# Paper Folding — Two-Panel Example (Single Fold)
# Directions fixed to match exam references:
# - Vertical fold: Right → Left  (L)  → landscape paper
# - Horizontal fold: Bottom → Top (U) → square paper
# Left tile = folded reference (other half dotted), shapes on folding half.
# Right tile = plain paper with fold line + arrow that CROSSES the dotted line.
# Uniform tile size within each question; choices have no extra outer box.

import io, time, math, random, zipfile, json
from typing import List, Tuple, Optional, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------- utils ----------------
DPI = 2  # internal render scale for crisp lines

def make_rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        seed = int(time.time() * 1000) % 2147483647
    return random.Random(seed)

def _try_font(size: int):
    for n in ("DejaVuSans.ttf", "Arial.ttf"):
        try: return ImageFont.truetype(n, size)
        except: pass
    return ImageFont.load_default()

def _hex_to_rgb(h):
    try:
        h = h.strip().lstrip("#")
        if len(h) == 3: h = "".join([c*2 for c in h])
        if len(h) != 6: return (255,255,255)
        return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
    except: return (255,255,255)

def show_image(img, caption=None):
    st.image(img, caption=caption, use_container_width=True)

# ---------------- geometry ----------------
def reflect_point(p: Tuple[float, float], axis: str) -> Tuple[float, float]:
    x, y = p
    return (-x, y) if axis == "V" else (x, -y)

def dir_to_axis_and_half(direction: str):
    # which half folds over which
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
    x1, y1 = p1; x2, y2 = p2
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

# -------------- adaptive canvas/paper ----------------
def get_tile_size(axis: str) -> Tuple[int,int]:
    """Return (W,H) for tiles based on axis."""
    if axis == "V":     # landscape like Example (2)
        return (420, 180)
    else:               # square like Example (1)
        return (260, 260)

def paper_rect_on_canvas(W: int, H: int, ratio: float) -> Tuple[int,int,int,int]:
    """
    ratio = width / height for the paper inside a tile.
    Keeps margins and centers the paper. Adjusts if one side would overflow.
    """
    m = int(0.09 * min(W, H))
    max_w = W - 2*m
    max_h = H - 2*m
    # try to use full width first
    pw = max_w
    ph = int(round(pw / ratio))
    if ph > max_h:
        ph = max_h
        pw = int(round(ph * ratio))
    l = (W - pw) // 2
    t = (H - ph) // 2
    return (l, t, l + pw, t + ph)

# -------------- example (two tiles) ----------------
def draw_example(direction: str,
                 shapes_half: List[Tuple[str, Tuple[float,float]]],
                 tile_size: Tuple[int,int],
                 ratio: float,
                 bg=(255,255,255), paper_fill=(250,250,250),
                 outline=(20,20,20), fold_line_color=(60,60,60)) -> Image.Image:

    TW, TH = [d * DPI for d in tile_size]
    pad = int(18 * DPI)

    # Left tile: folded reference (one paper; other half dotted)
    left = Image.new("RGB", (TW, TH), bg)
    pl, pt, pr, pb = paper_rect_on_canvas(TW, TH, ratio)
    ld = ImageDraw.Draw(left)
    paper_shadow(left, (pl, pt, pr, pb))
    rounded_rect(ld, (pl, pt, pr, pb), 14, paper_fill, outline, 6)

    axis, fold_half = dir_to_axis_and_half(direction)
    other_half = {"left":"right","right":"left","top":"bottom","bottom":"top"}[fold_half]

    # dotted the NON-folding half inside the same paper
    def dotted_half(rect, axis, half):
        l,t,r,b = rect; cx=(l+r)//2; cy=(t+b)//2
        if axis=="V":
            box = (l,t,cx,b) if half=="left" else (cx,t,r,b)
        else:
            box = (l,t,r,cy) if half=="top" else (l,cy,r,b)
        x0,y0,x1,y1 = box
        step = 20
        for x in range(x0+8, x1-8, step):
            ld.line([(x,y0+8),(x+8,y0+8)], fill=fold_line_color, width=3)
            ld.line([(x,y1-8),(x+8,y1-8)], fill=fold_line_color, width=3)
        for y in range(y0+8, y1-8, step):
            ld.line([(x0+8,y),(x0+8,y+8)], fill=fold_line_color, width=3)
            ld.line([(x1-8,y),(x1-8,y+8)], fill=fold_line_color, width=3)

    dotted_half((pl,pt,pr,pb), axis, other_half)

    # shapes on the folding half only
    for shp,(sx,sy) in shapes_half:
        px,py = norm_to_px((sx,sy),(pl,pt,pr,pb))
        draw_shape_outline(ld,(px,py), size=10*DPI, shape=shp, color=outline, width=5)

    # Right tile: plain paper with fold line + ARROW that crosses center
    right = Image.new("RGB", (TW, TH), bg)
    prl, prt, prr, prb = paper_rect_on_canvas(TW, TH, ratio)
    rd = ImageDraw.Draw(right)
    paper_shadow(right, (prl, prt, prr, prb))
    rounded_rect(rd, (prl, prt, prr, prb), 14, paper_fill, outline, 6)

    cx = (prl + prr)//2
    cy = (prt + prb)//2
    Wp = prr - prl
    Hp = prb - prt

    if axis == "V":
        dotted_line_circles(rd, (cx, prt + 10), (cx, prb - 10), fold_line_color, dot_r=4, gap=16)
        # arc spans both halves horizontally
        aL = prl + int(0.06 * Wp); aR = prr - int(0.06 * Wp)
        aT = prt + int(0.18 * Hp); aB = prt + int(0.78 * Hp)
        draw_arc_arrow(rd, (aL, aT, aR, aB), 210, -20, outline, width=6, head_len=18, head_w=12)
    else:
        dotted_line_circles(rd, (prl + 10, cy), (prr - 10, cy), fold_line_color, dot_r=4, gap=16)
        # arc spans both halves vertically
        aL = prl + int(0.10 * Wp); aR = prr - int(0.10 * Wp)
        aT = prt + int(0.06 * Hp); aB = prb - int(0.06 * Hp)
        draw_arc_arrow(rd, (aL, aT, aR, aB), 160, 340, outline, width=6, head_len=18, head_w=12)

    # compose
    example = Image.new("RGB", (TW*2 + pad, TH), bg)
    example.paste(left, (0, 0))
    example.paste(right, (TW + pad, 0))

    if DPI != 1:
        example = example.resize((example.width // DPI, example.height // DPI), Image.LANCZOS)
    return example

# -------------- choices (one paper per tile) --------------
def draw_choice(shapes_norm: List[Tuple[str,Tuple[float,float]]],
                tile_size: Tuple[int,int], ratio: float,
                bg=(255,255,255), paper_fill=(250,250,250),
                outline=(20,20,20)) -> Image.Image:
    TW, TH = [d * DPI for d in tile_size]
    img = Image.new("RGB", (TW, TH), bg)
    d = ImageDraw.Draw(img)
    l,t,r,b = paper_rect_on_canvas(TW, TH, ratio)
    paper_shadow(img, (l,t,r,b))
    rounded_rect(d, (l,t,r,b), 14, paper_fill, outline, 6)
    for shp,(x,y) in shapes_norm:
        px,py = norm_to_px((x,y),(l,t,r,b))
        draw_shape_outline(d,(px,py), size=10*DPI, shape=shp, color=outline, width=4)
    return img.resize(tile_size, Image.LANCZOS) if DPI!=1 else img

# -------------- grid & compose --------------
def overlay_label_below(tile:Image.Image, label:str, color=(20,20,20))->Image.Image:
    font=_try_font(24); tw=ImageDraw.Draw(tile).textlength(label,font=font)
    canvas=Image.new("RGB",(tile.width, tile.height+40),(255,255,255))
    canvas.paste(tile,(0,0)); ImageDraw.Draw(canvas).text(((canvas.width-tw)/2, tile.height+10), label, fill=color, font=font)
    return canvas

def compose_2x2_grid(choices, labels, pad=24, bg=(255,255,255)):
    tiles=[overlay_label_below(choices[i], labels[i]) for i in range(4)]
    w,h=tiles[0].size; W=2*w+3*pad; H=2*h+3*pad
    canvas=Image.new("RGB",(W,H),bg)
    canvas.paste(tiles[0],(pad,pad));           canvas.paste(tiles[1],(2*pad+w,pad))
    canvas.paste(tiles[2],(pad,2*pad+h));       canvas.paste(tiles[3],(2*pad+w,2*pad+h))
    return canvas

def stack_vertical(top_img, bottom_img, pad=28, bg=(255,255,255)):
    W=max(top_img.width,bottom_img.width)+2*pad
    H=top_img.height+bottom_img.height+3*pad
    canvas=Image.new("RGB",(W,H),bg)
    canvas.paste(top_img,((W-top_img.width)//2, pad))
    canvas.paste(bottom_img,((W-bottom_img.width)//2, top_img.height+2*pad))
    return canvas

# -------------- generator --------------
def generate_single_fold_question(rng: random.Random, style: Optional[Dict]=None) -> Dict:
    bg          = (style.get("bg") if style else None) or (255,255,255)
    paper_fill  = (style.get("paper_fill") if style else None) or (250,250,250)
    outline     = (style.get("outline") if style else None) or (20,20,20)
    fold_color  = (style.get("fold_line") if style else None) or (60,60,60)

    # Choose axis randomly but clamp directions you requested
    axis_choice = rng.choice(["V","H"])
    direction   = "L" if axis_choice=="V" else "U"  # R→L or Bottom→Top
    axis, shaded_half = dir_to_axis_and_half(direction)

    # Tile size and paper ratio depend on axis
    tile_size = get_tile_size(axis)      # (W,H)
    ratio     = 2.3 if axis=="V" else 1.0

    # Sample two shapes on the FOLDING half (fix: correct sign for y!)
    def sample_point_on_half():
        x = rng.uniform(-0.85, 0.85)
        y = rng.uniform(-0.70, 0.70)
        if axis == "V":
            # x<0 is left, x>0 is right
            x = rng.uniform(-0.85, -0.12) if shaded_half == "left" else rng.uniform(0.12, 0.85)
        else:  # axis == "H"  (y>0 is TOP, y<0 is BOTTOM)
            y = rng.uniform(0.12, 0.85) if shaded_half == "top" else rng.uniform(-0.85, -0.12)
        return (round(x,3), round(y,3))

    pts=[]
    for _ in range(2):
        p=sample_point_on_half(); tries=0
        while any(math.hypot(p[0]-q[0], p[1]-q[1])<0.35 for q in pts) and tries<20:
            p=sample_point_on_half(); tries+=1
        pts.append(p)

    shapes = ["circle","triangle"]; rng.shuffle(shapes)
    shapes_half = [(shapes[0], pts[0]), (shapes[1], pts[1])]

    # Correct answer = shapes mirrored about the fold axis (true reflection)
    mirrored = [(s, reflect_point(p, axis)) for s,p in shapes_half]
    shapes_correct = shapes_half + mirrored

    # Distractors
    wrong_axis = "H" if axis=="V" else "V"
    shapes_wrong1 = list(shapes_half)  # no reflection
    shapes_wrong2 = shapes_half + [(s, reflect_point(p, wrong_axis)) for s,p in shapes_half]  # wrong axis
    swap = {"circle":"triangle","triangle":"circle"}
    shapes_wrong3 = shapes_half + [(swap[s], reflect_point(p, axis)) for s,p in shapes_half]   # wrong shape + right axis

    # Render choices with the right tile size/ratio for this question
    c0 = draw_choice(shapes_correct, tile_size, ratio, bg, paper_fill, outline)
    c1 = draw_choice(shapes_wrong1,  tile_size, ratio, bg, paper_fill, outline)
    c2 = draw_choice(shapes_wrong2,  tile_size, ratio, bg, paper_fill, outline)
    c3 = draw_choice(shapes_wrong3,  tile_size, ratio, bg, paper_fill, outline)
    choices = [c0, c1, c2, c3]; rng.shuffle(choices); correct_index = choices.index(c0)

    labels_ar = ["أ","ب","ج","د"]
    grid = compose_2x2_grid(choices, labels_ar, pad=24, bg=bg)
    example = draw_example(direction, shapes_half, tile_size, ratio, bg, paper_fill, outline, fold_color)

    prompt_ar = "ما رمز البديل الذي يحتوي على صورة الورقة بعد إعادة فتحها من بين البدائل الأربعة؟"
    meta = {
        "type":"folding_single_shapes",
        "direction":direction,
        "axis":axis,
        "half":shaded_half,
        "ratio":ratio,
        "tile_size":tile_size,
        "shapes_half":shapes_half
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
st.set_page_config(page_title="Paper Folding — Two-Panel Example", layout="wide")
st.title("Paper Folding — Two-Panel Example (Single Fold)")

with st.sidebar:
    st.header("Controls")
    seed_str = st.text_input("Seed (optional)", value="")
    seed = int(seed_str.strip()) if seed_str.strip().isdigit() else (
        abs(hash(seed_str)) % (2**31) if seed_str.strip() else None
    )
    st.header("Visual Style")
    bg_hex        = st.text_input("Background", "#FFFFFF")
    paper_hex     = st.text_input("Paper fill", "#FAFAFA")
    outline_hex   = st.text_input("Outline", "#1A1A1A")
    fold_line_hex = st.text_input("Fold-line (dots)", "#3C3C3C")
    gen_btn = st.button("Generate New Question", type="primary", use_container_width=True)

    st.subheader("Batch")
    batch_n  = st.number_input("How many?", min_value=2, max_value=100, value=8, step=1)
    batch_btn= st.button("Generate Batch ZIP", use_container_width=True)

style = {
    "bg": _hex_to_rgb(bg_hex),
    "paper_fill": _hex_to_rgb(paper_hex),
    "outline": _hex_to_rgb(outline_hex),
    "fold_line": _hex_to_rgb(fold_line_hex),
}

rng = make_rng(seed)
if ("fold_qpack" not in st.session_state) or gen_btn:
    st.session_state.fold_qpack = generate_single_fold_question(rng, style=style)

qp = st.session_state.get("fold_qpack")

if qp:
    grid = compose_2x2_grid(qp["choices_imgs"], qp["labels_ar"], pad=24, bg=style["bg"])
    composite = stack_vertical(qp["problem_img"], grid, pad=28, bg=style["bg"])

    st.subheader("Question")
    st.write(qp["prompt"])
    show_image(composite)

    chosen = st.radio("Pick your answer:", qp["labels_ar"], index=0, horizontal=True)
    if st.button("Check answer"):
        ans = qp["labels_ar"][qp["correct_index"]]
        st.success(f"الإجابة الصحيحة: {ans}") if chosen == ans else st.error(f"غير صحيح. الإجابة الصحيحة: {ans}")

    st.markdown("---"); st.subheader("Export")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG"); zf.writestr("problem.png", pb.getvalue())
        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG"); zf.writestr(f"choice_{qp['labels_ar'][i]}.png", cb.getvalue())
        gb = io.BytesIO(); grid.save(gb, format="PNG"); zf.writestr("grid.png", gb.getvalue())
        qb = io.BytesIO(); composite.save(qb, format="PNG"); zf.writestr("question.png", qb.getvalue())
        meta = {
            "type": qp["meta"]["type"],
            "direction": qp["meta"]["direction"],
            "axis": qp["meta"]["axis"],
            "half": qp["meta"]["half"],
            "ratio": qp["meta"]["ratio"],
            "tile_size": qp["meta"]["tile_size"],
            "shapes_half": qp["meta"]["shapes_half"],
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

    if batch_btn:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(batch_n)):
                local_rng  = make_rng((seed or 0) + k + 1)
                local_pack = generate_single_fold_question(local_rng, style=style)
                qid = f"folding_two_panel_{int(time.time()*1000)}_{k}"

                local_grid = compose_2x2_grid(local_pack["choices_imgs"], local_pack["labels_ar"], pad=24, bg=style["bg"])
                local_comp = stack_vertical(local_pack["problem_img"], local_grid, pad=28, bg=style["bg"])

                pb = io.BytesIO(); local_pack["problem_img"].save(pb, format="PNG"); zf.writestr(f"{qid}/problem.png", pb.getvalue())
                for i, im in enumerate(local_pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG"); zf.writestr(f"{qid}/choice_{local_pack['labels_ar'][i]}.png", cb.getvalue())
                gb = io.BytesIO(); local_grid.save(gb, format="PNG"); zf.writestr(f"{qid}/grid.png", gb.getvalue())
                qb = io.BytesIO(); local_comp.save(qb, format="PNG"); zf.writestr(f"{qid}/question.png", qb.getvalue())

                meta = {
                    "id": qid,
                    "type": local_pack["meta"]["type"],
                    "direction": local_pack["meta"]["direction"],
                    "axis": local_pack["meta"]["axis"],
                    "half": local_pack["meta"]["half"],
                    "ratio": local_pack["meta"]["ratio"],
                    "tile_size": local_pack["meta"]["tile_size"],
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
            file_name=f"batch_folding_two_panel_{int(time.time())}.zip",
            mime="application/zip"
        )
