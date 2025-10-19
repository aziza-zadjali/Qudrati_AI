# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py

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

def resize_max_height(img, max_h: int):
    """Scale image down to max height (keep aspect)."""
    if img.height <= max_h or max_h <= 0:
        return img
    scale = max_h / img.height
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)

# ----------------------------
# Folding math
# ----------------------------
def reflect_point(p: Tuple[float, float], axis: str) -> Tuple[float, float]:
    x, y = p
    if axis == "V":
        return (-x, y)
    if axis == "H":
        return (x, -y)
    if axis == "D1":
        return (y, x)
    if axis == "D2":
        return (-y, -x)
    return (x, y)

def unfold_points(base_point: Tuple[float, float], folds_axes: List[str]) -> List[Tuple[float, float]]:
    pts = [base_point]
    for axis in reversed(folds_axes):
        mirrored = [reflect_point(p, axis) for p in pts]
        pts = pts + mirrored
    out, seen = [], set()
    for x, y in pts:
        key = (round(x, 4), round(y, 4))
        if key not in seen:
            seen.add(key)
            out.append((x, y))
    return out

# ----------------------------
# Drawing helpers
# ----------------------------
def _arrow(d: ImageDraw.ImageDraw, start, end, color, width):
    d.line([start, end], fill=color, width=width)
    vx, vy = end[0] - start[0], end[1] - start[1]
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    head_len = max(12, width * 3)
    head_w = max(8, width * 2)
    tip = (end[0], end[1])
    base = (end[0] - ux * head_len, end[1] - uy * head_len)
    p1 = (base[0] + px * head_w, base[1] + py * head_w)
    p2 = (base[0] - px * head_w, base[1] - py * head_w)
    d.polygon([tip, p1, p2], fill=color)

def draw_fold_icon(direction: str, size=(180, 180),
                   bg=(255, 255, 255), paper_fill=(250, 250, 250),
                   outline=(20, 20, 20), stroke=4) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    margin = max(10, int(min(W, H) * 0.1))
    left, top, right, bottom = margin, margin, W - margin, H - margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    cx, cy = W // 2, H // 2
    Llen = max(18, min(W, H) // 3)
    if direction == "L":
        start, end = (cx + Llen // 2, cy), (cx - Llen, cy)
    elif direction == "R":
        start, end = (cx - Llen // 2, cy), (cx + Llen, cy)
    elif direction == "U":
        start, end = (cx, cy + Llen // 2), (cx, cy - Llen)
    elif direction == "D":
        start, end = (cx, cy - Llen // 2), (cx, cy + Llen)
    elif direction == "UL":
        start, end = (cx + Llen // 2, cy + Llen // 2), (cx - Llen, cy - Llen)
    elif direction == "UR":
        start, end = (cx - Llen // 2, cy + Llen // 2), (cx + Llen, cy - Llen)
    elif direction == "DL":
        start, end = (cx + Llen // 2, cy - Llen // 2), (cx - Llen, cy + Llen)
    else:  # "DR"
        start, end = (cx - Llen // 2, cy - Llen // 2), (cx + Llen, cy + Llen)
    _arrow(d, start, end, outline, max(3, stroke))
    return img

def draw_paper_with_holes(size=(420, 420), holes=None,
                          paper_margin=40, hole_radius=12,
                          bg=(255, 255, 255), paper_fill=(250, 250, 250),
                          outline=(20, 20, 20), stroke=5) -> Image.Image:
    holes = holes or []
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    left, top, right, bottom = paper_margin, paper_margin, W - paper_margin, H - paper_margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    for (x, y) in holes:
        px = (x + 1) / 2.0 * (right - left) + left
        py = (1 - (y + 1) / 2.0) * (bottom - top) + top
        r = max(5, hole_radius)
        d.ellipse([px - r, py - r, px + r, py + r], fill=outline, outline=outline, width=1)
    return img

def draw_folded_with_punch(point_folded: Tuple[float, float], size=(220, 220),
                           bg=(255, 255, 255), paper_fill=(250, 250, 250),
                           outline=(20, 20, 20), stroke=4, show_punch_label=False) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    margin = max(10, int(min(W, H) * 0.08))
    left, top, right, bottom = margin, margin, W - margin, H - margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    x, y = point_folded
    px = (x + 1) / 2.0 * (right - left) + left
    py = (1 - (y + 1) / 2.0) * (bottom - top) + top
    r = max(6, int(min(W, H)*0.05))
    d.ellipse([px - r, py - r, px + r, py + r], fill=outline, outline=outline)
    if show_punch_label:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        d.text((left, max(2, top - 20)), "Punch", fill=(20,20,20), font=font)
    return img

def overlay_label(img: Image.Image, label: str, spot="tr",
                  circle_fill=(255,255,255), circle_outline=(20,20,20),
                  text_color=(20,20,20), r: Optional[int]=None) -> Image.Image:
    """Draw a small labeled circle (أ/ب/ج/د) inside the image."""
    im = img.copy()
    d = ImageDraw.Draw(im)
    if r is None:
        r = max(10, int(min(im.size) * 0.08))  # scale with image size
    margin = max(6, int(min(im.size) * 0.06))
    if spot == "tl":
        cx, cy = margin + r, margin + r
    elif spot == "tr":
        cx, cy = im.width - margin - r, margin + r
    elif spot == "bl":
        cx, cy = margin + r, im.height - margin - r
    else:  # br
        cx, cy = im.width - margin - r, im.height - margin - r
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=circle_fill, outline=circle_outline, width=2)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, int(r*1.1)))
    except Exception:
        font = ImageFont.load_default()
    # approximate centering
    try:
        tw = d.textlength(label, font=font)
    except Exception:
        tw = r
    th = r  # approx
    d.text((cx - tw/2, cy - th/2), label, fill=text_color, font=font)
    return im

def compose_2x2_grid(choices: List[Image.Image], labels: List[str],
                     pad=14, tile_border=1, bg=(255,255,255),
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
        r, c = idx // 2, idx % 2
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(im, (x, y))
        d.rectangle([x, y, x + w, y + h], outline=(20,20,20), width=tile_border)
    return canvas

def compose_row(images, pad=10, bg=(255, 255, 255)) -> Image.Image:
    if not images:
        return Image.new("RGB", (360, 120), bg)
    w, h = images[0].size
    W = len(images) * w + (len(images) + 1) * pad
    H = h + 2 * pad
    canvas = Image.new("RGB", (W, H), bg)
    x = pad
    for im in images:
        canvas.paste(im, (x, pad))
        x += w + pad
    return canvas

def stack_vertical(top_img: Image.Image, bottom_img: Image.Image, pad=18, bg=(255,255,255)) -> Image.Image:
    W = max(top_img.width, bottom_img.width) + 2*pad
    H = top_img.height + bottom_img.height + 3*pad
    canvas = Image.new("RGB", (W, H), bg)
    x_top = (W - top_img.width)//2
    canvas.paste(top_img, (x_top, pad))
    x_bot = (W - bottom_img.width)//2
    canvas.paste(bottom_img, (x_bot, top_img.height + 2*pad))
    return canvas

# ----------------------------
# Generator
# ----------------------------
def generate_folding_challenge(rng: random.Random, difficulty="Medium",
                               allow_diagonal=False, style: Optional[Dict] = None,
                               show_punch_label=False) -> Dict:
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    stroke = (style.get("stroke_width") if style else None) or 4

    # Compact sizes (from style)
    icon_size = style.get("icon_size", (180, 180))
    punch_size = style.get("punch_size", (220, 220))
    choice_size = style.get("choice_size", (420, 420))
    paper_margin = style.get("paper_margin", 40)
    hole_radius = style.get("hole_radius", 12)
    row_pad = style.get("row_pad", 16)

    if difficulty == "Easy":
        n_folds = rng.choice([1, 2])
    elif difficulty == "Hard":
        n_folds = 3
    else:
        n_folds = rng.choice([2, 3])

    dirs_card = ["L", "R", "U", "D"]
    dirs_diag = ["UL", "UR", "DL", "DR"]
    dirs_all = dirs_card + (dirs_diag if allow_diagonal else [])
    folds = []
    for i in range(n_folds):
        cand = rng.choice(dirs_all) if i == 0 else rng.choice([d for d in dirs_all if d != folds[-1]])
        folds.append(cand)

    def dir_to_axis(d):
        if d in ("L", "R"): return "V"
        if d in ("U", "D"): return "H"
        if d in ("UL", "DR"): return "D1"
        return "D2"

    axes = [dir_to_axis(d) for d in folds]

    px = rng.uniform(-0.65, 0.65)
    py = rng.uniform(-0.65, 0.65)
    point_folded = (px, py)

    holes = unfold_points(point_folded, axes)

    # Visuals (use compact sizes)
    icons = [draw_fold_icon(d, size=icon_size, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke) for d in folds]
    row = compose_row(icons, pad=row_pad, bg=bg)
    punch = draw_folded_with_punch(point_folded, size=punch_size, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke, show_punch_label=show_punch_label)

    W = max(row.size[0], punch.size[0]) + 24
    H = row.size[1] + punch.size[1] + 40
    problem = Image.new("RGB", (W, H), bg)
    x1 = (W - row.size[0]) // 2
    problem.paste(row, (x1, 12))
    x2 = (W - punch.size[0]) // 2
    problem.paste(punch, (x2, row.size[1] + 24))

    # Choices
    def mk_choice(holes_pts):
        return draw_paper_with_holes(size=choice_size, holes=holes_pts, bg=bg, paper_fill=paper_fill,
                                     outline=outline, stroke=stroke, paper_margin=paper_margin, hole_radius=hole_radius)

    correct = mk_choice(holes)
    holes_d1 = unfold_points(point_folded, axes[:-1]) if len(axes) > 0 else [(px, py)]
    d1 = mk_choice(holes_d1)

    if len(axes) > 0:
        wrong_axes = axes.copy()
        idx = rng.randrange(len(wrong_axes))
        wrong_axes[idx] = "H" if wrong_axes[idx] == "V" else ("V" if wrong_axes[idx] == "H" else ("D2" if wrong_axes[idx] == "D1" else "D1"))
        holes_d2 = unfold_points(point_folded, wrong_axes)
    else:
        holes_d2 = [(-px, py)]
    d2 = mk_choice(holes_d2)

    d3 = correct.rotate(90, expand=False)

    choices = [correct, d1, d2, d3]
    rng.shuffle(choices)
    answer_index = choices.index(correct)

    prompt_ar = "ما رمز البديل الذي يحتوي على صورة الورقة بعد إعادة فتحها من بين البدائل الأربعة؟"
    rule_desc_ar = "عند فتح كل طية، تنعكس مواضع الثقوب عبر خط الطي."

    return {
        "problem_img": problem,
        "choices_imgs": choices,
        "correct_index": answer_index,
        "prompt": prompt_ar,
        "rule_desc": rule_desc_ar,
        "meta": {"type": "folding", "folds": folds, "axes": axes, "punch": (round(px,3), round(py,3))}
    }

# ----------------------------
# Streamlit UI (Arabic)
# ----------------------------
st.set_page_config(page_title="تحدي طي الورقة", layout="wide")
st.title("تحدي طي الورقة")

with st.sidebar:
    st.header("التحكم")
    diff_map = {"سهل": "Easy", "متوسط": "Medium", "صعب": "Hard"}
    diff_choice = st.select_slider("مستوى الصعوبة", options=list(diff_map.keys()), value="متوسط")
    difficulty = diff_map[diff_choice]

    seed_str = st.text_input("البذرة (اختياري)", value="")
    seed = None
    if seed_str.strip():
        try:
            seed = int(seed_str.strip())
        except Exception:
            seed = abs(hash(seed_str)) % (2**31)

    allow_diag = st.checkbox("السماح بالطيات القطرية", value=False)

    st.header("التنسيق البصري")
    bg_hex = st.text_input("لون الخلفية", "#FFFFFF")
    paper_hex = st.text_input("لون الورقة", "#FAFAFA")
    outline_hex = st.text_input("لون الحدود", "#141414")
    stroke_w = st.number_input("سماكة إطار الورقة (px)", min_value=2, max_value=12, value=4, step=1,
                               help="السُمك بالبيكسل للحدود المرسومة حول الورقة/العناصر")

    # 🔽 جديد: وضع مدمج + ارتفاع أقصى للصورة المركبة
    compact_mode = st.toggle("وضع مدمج (تصغير تلقائي بلا تمرير)", value=True)
    max_height_px = st.slider("الارتفاع الأقصى للصورة المركبة (px)", min_value=400, max_value=900, value=620, step=10)

    # نمط مرجعي (شبكة 2×2 + حروف داخل الصور + نقطة فقط للثقب)
    reference_mode = st.toggle("مطابقة النمط المرجعي (شبكة 2×2 + حروف داخل الصور + نقطة فقط للثقب)", value=True)

    # تعيين أحجام مدمجة
    if compact_mode:
        icon_size   = (96, 96)
        punch_size  = (160, 160)
        choice_size = (160, 160)
        paper_margin = 26
        hole_radius  = 8
        row_pad      = 10
        grid_pad     = 14
        tile_border  = 1
    else:
        icon_size   = (180, 180)
        punch_size  = (220, 220)
        choice_size = (420, 420)
        paper_margin = 40
        hole_radius  = 12
        row_pad      = 16
        grid_pad     = 18
        tile_border  = 2

    style = {
        "bg": _hex_to_rgb(bg_hex),
        "paper_fill": _hex_to_rgb(paper_hex),
        "outline": _hex_to_rgb(outline_hex),
        "stroke_width": int(stroke_w),
        # Sizes
        "icon_size": icon_size,
        "punch_size": punch_size,
        "choice_size": choice_size,
        "paper_margin": paper_margin,
        "hole_radius": hole_radius,
        "row_pad": row_pad,
        "grid_pad": grid_pad,
        "tile_border": tile_border,
        "max_height": max_height_px,
    }

    st.markdown("---")
    gen_btn = st.button("إنشاء سؤال جديد", type="primary", use_container_width=True)

    st.subheader("إنشاء مجموعة (Batch)")
    batch_n = st.number_input("عدد الأسئلة في المجموعة", min_value=2, max_value=100, value=10, step=1)
    batch_btn = st.button("إنشاء ملف ZIP للمجموعة", use_container_width=True)

rng = make_rng(seed)
labels_ar = ["أ", "ب", "ج", "د"]

def make_one():
    return generate_folding_challenge(
        rng, difficulty=difficulty, allow_diagonal=allow_diag, style=style,
        show_punch_label=not reference_mode  # في النمط المرجعي نخفي النص ونضع نقطة فقط
    )

if ("fold_qpack" not in st.session_state) or gen_btn:
    st.session_state.fold_qpack = make_one()

qp = st.session_state.get("fold_qpack")

if qp:
    if reference_mode:
        grid = compose_2x2_grid(qp["choices_imgs"], labels_ar,
                                pad=style["grid_pad"], tile_border=style["tile_border"],
                                bg=style["bg"], label_spot="tr")
        composite = stack_vertical(qp["problem_img"], grid, pad=18 if style["row_pad"] <= 10 else 22, bg=style["bg"])
        # 🔽 جديد: تصغير تلقائي لارتفاع أقصى
        composite = resize_max_height(composite, style.get("max_height", 620))
        st.subheader("السؤال")
        st.write(qp["prompt"])
        show_image(composite)
    else:
        colQ, colA = st.columns([2.1, 1.4])
        with colQ:
            st.subheader("السؤال")
            show_image(resize_max_height(qp["problem_img"], style.get("max_height", 620)))
            st.write(qp["prompt"])
        with colA:
            st.subheader("البدائل")
            cols2 = st.columns(2)
            for i, im in enumerate(qp["choices_imgs"]):
                with cols2[i % 2]:
                    im_small = resize_max_height(im, 300 if style["choice_size"][0] > 200 else 200)
                    show_image(overlay_label(im_small, labels_ar[i]), caption=f"{labels_ar[i]}")

    # اختيار الإجابة
    chosen = st.radio("اختر الإجابة:", labels_ar, index=0, horizontal=True)
    if st.button("تحقق من الإجابة"):
        if chosen == labels_ar[qp["correct_index"]]:
            st.success(f"صحيح. الإجابة: {labels_ar[qp['correct_index']]}")
        else:
            st.error(f"غير صحيح. الإجابة الصحيحة: {labels_ar[qp['correct_index']]}")
        st.markdown("**القاعدة:** " + qp["rule_desc"])

    st.markdown("---")
    st.subheader("تصدير")

    # تحضير التصدير: صورة مركبة + ملفات منفصلة + JSON
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # صور منفصلة
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG")
        zf.writestr("problem.png", pb.getvalue())
        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG")
            zf.writestr(f"choice_{labels_ar[i]}.png", cb.getvalue())

        # صورة مركبة مُصغّرة للارتفاع المحدد
        grid = compose_2x2_grid(qp["choices_imgs"], labels_ar,
                                pad=style["grid_pad"], tile_border=style["tile_border"],
                                bg=style["bg"], label_spot="tr")
        composite = stack_vertical(qp["problem_img"], grid, pad=18 if style["row_pad"] <= 10 else 22, bg=style["bg"])
        composite = resize_max_height(composite, style.get("max_height", 620))
        qb = io.BytesIO(); composite.save(qb, format="PNG")
        zf.writestr("question.png", qb.getvalue())

        # ميتاداتا
        meta = {
            "type": qp["meta"]["type"],
            "folds": qp["meta"]["folds"],
            "axes": qp["meta"]["axes"],
            "punch": qp["meta"]["punch"],
            "prompt": qp["prompt"],
            "labels": labels_ar,
            "correct_label": labels_ar[qp["correct_index"]],
            "rule": qp["rule_desc"],
            "compact_mode": True,
            "max_height_px": style.get("max_height", 620),
        }
        zf.writestr("question.json", json.dumps(meta, indent=2, ensure_ascii=False))

    st.download_button(
        "تنزيل ملف ZIP",
        data=buf.getvalue(),
        file_name=f"folding_{int(time.time())}.zip",
        mime="application/zip"
    )

    if batch_btn:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(batch_n)):
                local_rng = make_rng((seed or 0) + k + 1)
                local_pack = generate_folding_challenge(
                    local_rng, difficulty=difficulty, allow_diagonal=allow_diag, style=style,
                    show_punch_label=not reference_mode)
                qid = f"folding_{int(time.time()*1000)}_{k}"

                # separate images
                pb = io.BytesIO(); local_pack["problem_img"].save(pb, format="PNG")
                zf.writestr(f"{qid}/problem.png", pb.getvalue())
                for i, im in enumerate(local_pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG")
                    zf.writestr(f"{qid}/choice_{labels_ar[i]}.png", cb.getvalue())

                # composite question.png (scaled)
                grid = compose_2x2_grid(local_pack["choices_imgs"], labels_ar,
                                        pad=style["grid_pad"], tile_border=style["tile_border"],
                                        bg=style["bg"], label_spot="tr")
                composite = stack_vertical(local_pack["problem_img"], grid, pad=18 if style["row_pad"] <= 10 else 22, bg=style["bg"])
                composite = resize_max_height(composite, style.get("max_height", 620))
                qb = io.BytesIO(); composite.save(qb, format="PNG")
                zf.writestr(f"{qid}/question.png", qb.getvalue())

                meta = {
                    "id": qid,
                    "type": local_pack["meta"]["type"],
                    "folds": local_pack["meta"]["folds"],
                    "axes": local_pack["meta"]["axes"],
                    "punch": local_pack["meta"]["punch"],
                    "prompt": local_pack["prompt"],
                    "labels": labels_ar,
                    "correct_label": labels_ar[local_pack["correct_index"]],
                    "rule": local_pack["rule_desc"],
                    "compact_mode": True,
                    "max_height_px": style.get("max_height", 620),
                }
                zf.writestr(f"{qid}/question.json", json.dumps(meta, indent=2, ensure_ascii=False))
                index.append(meta)

            zf.writestr("index.json", json.dumps(index, indent=2, ensure_ascii=False))

        st.download_button(
            "تنزيل ملف ZIP للمجموعة",
            data=bbuf.getvalue(),
            file_name=f"batch_folding_{int(time.time())}.zip",
            mime="application/zip"
        )
