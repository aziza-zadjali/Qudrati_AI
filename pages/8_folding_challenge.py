# -*- coding: utf-8 -*-
# pages/8_folding_challenge.py
# Folding Challenge generator (with optional diagonal folds), ASCII-only drawing (Pillow 10+ safe).

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
# Utilities (Pillow-safe text, RNG, colors)
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
        # Fallback using textlength and approx height
        try:
            w = int(d.textlength(text, font=font))
        except Exception:
            w = int(len(text) * font_size * 0.6)
        h = int(font_size * 1.2)
        x = (size[0] - w) / 2
        y = (size[1] - h) / 2
    d.text((x, y), text, fill=color, font=font)
    return img

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
    """Compat across Streamlit versions."""
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)

# ----------------------------
# Folding math: reflect and unfold (supports diagonals)
# ----------------------------
def reflect_point(p: Tuple[float, float], axis: str) -> Tuple[float, float]:
    x, y = p
    if axis == "V":  # reflect across x=0
        return (-x, y)
    if axis == "H":  # reflect across y=0
        return (x, -y)
    if axis == "D1":  # reflect across y = x
        return (y, x)
    if axis == "D2":  # reflect across y = -x
        return (-y, -x)
    return (x, y)

def unfold_points(base_point: Tuple[float, float], folds_axes: List[str]) -> List[Tuple[float, float]]:
    pts = [base_point]
    for axis in reversed(folds_axes):
        mirrored = [reflect_point(p, axis) for p in pts]
        pts = pts + mirrored
    # Deduplicate (rounding)
    out, seen = [], set()
    for x, y in pts:
        key = (round(x, 4), round(y, 4))
        if key not in seen:
            seen.add(key)
            out.append((x, y))
    return out

# ----------------------------
# Drawing helpers for folding visuals
# ----------------------------
def _arrow(d: ImageDraw.ImageDraw, start, end, color, width):
    d.line([start, end], fill=color, width=width)
    vx, vy = end[0] - start[0], end[1] - start[1]
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    head_len = max(14, width * 3)
    head_w = max(10, width * 2)
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
    margin = 18
    left, top, right, bottom = margin, margin, W - margin, H - margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    cx, cy = W // 2, H // 2
    Llen = min(W, H) // 3
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
    _arrow(d, start, end, outline, stroke)
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
        d.ellipse([px - hole_radius, py - hole_radius, px + hole_radius, py + hole_radius],
                  fill=outline, outline=outline, width=1)
    return img

def draw_folded_with_punch(point_folded: Tuple[float, float], size=(220, 220),
                           bg=(255, 255, 255), paper_fill=(250, 250, 250),
                           outline=(20, 20, 20), stroke=4, text_color=(20, 20, 20)) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    margin = 16
    left, top, right, bottom = margin, margin, W - margin, H - margin
    d.rectangle([left, top, right, bottom], outline=outline, width=stroke, fill=paper_fill)
    x, y = point_folded
    px = (x + 1) / 2.0 * (right - left) + left
    py = (1 - (y + 1) / 2.0) * (bottom - top) + top
    r = 10
    d.ellipse([px - r, py - r, px + r, py + r], fill=outline, outline=outline)
    # Keep English here to avoid Arabic shaping issues on Pillow
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    d.text((left, top - 22), "Punch", fill=text_color, font=font)
    return img

def compose_row(images, pad=16, bg=(255, 255, 255)) -> Image.Image:
    if not images:
        return text_image("No folds", color=(20,20,20), bg=bg)
    w, h = images[0].size
    W = len(images) * w + (len(images) + 1) * pad
    H = h + 2 * pad
    canvas = Image.new("RGB", (W, H), bg)
    x = pad
    for im in images:
        canvas.paste(im, (x, pad))
        x += w + pad
    return canvas

# ----------------------------
# Generator: Folding Challenge
# ----------------------------
def generate_folding_challenge(rng: random.Random, difficulty="Medium",
                               allow_diagonal=False, style: Optional[Dict] = None) -> Dict:
    bg = (style.get("bg") if style else None) or (255, 255, 255)
    paper_fill = (style.get("paper_fill") if style else None) or (250, 250, 250)
    outline = (style.get("outline") if style else None) or (20, 20, 20)
    stroke = (style.get("stroke_width") if style else None) or 5
    text_color = (style.get("text_color") if style else None) or (20, 20, 20)

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

    # Build problem: fold icons row + folded paper with punch
    icons = [draw_fold_icon(d, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke) for d in folds]
    row = compose_row(icons, pad=16, bg=bg)
    punch = draw_folded_with_punch(point_folded, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke, text_color=text_color)

    W = max(row.size[0], punch.size[0]) + 32
    H = row.size[1] + punch.size[1] + 48
    problem = Image.new("RGB", (W, H), bg)
    x1 = (W - row.size[0]) // 2
    problem.paste(row, (x1, 16))
    x2 = (W - punch.size[0]) // 2
    problem.paste(punch, (x2, row.size[1] + 32))

    # Choices (correct + 3 distractors)
    correct = draw_paper_with_holes(holes=holes, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    if len(axes) > 0:
        holes_d1 = unfold_points(point_folded, axes[:-1])
    else:
        holes_d1 = [(px, py)]
    d1 = draw_paper_with_holes(holes=holes_d1, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    if len(axes) > 0:
        wrong_axes = axes.copy()
        idx = rng.randrange(len(wrong_axes))
        wrong_axes[idx] = "H" if wrong_axes[idx] == "V" else ("V" if wrong_axes[idx] == "H" else ("D2" if wrong_axes[idx] == "D1" else "D1"))
        holes_d2 = unfold_points(point_folded, wrong_axes)
    else:
        holes_d2 = [(-px, py)]
    d2 = draw_paper_with_holes(holes=holes_d2, bg=bg, paper_fill=paper_fill, outline=outline, stroke=stroke)

    d3 = correct.rotate(90, expand=False)

    choices = [correct, d1, d2, d3]
    rng.shuffle(choices)
    answer_index = choices.index(correct)

    # Arabic prompt (exact phrase requested)
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
    # Difficulty (Arabic UI -> English internal)
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
    text_hex = st.text_input("لون النص", "#141414")
    stroke_w = st.number_input("سماكة الحدود", min_value=2, max_value=12, value=5, step=1)
    style = {
        "bg": _hex_to_rgb(bg_hex),
        "paper_fill": _hex_to_rgb(paper_hex),
        "outline": _hex_to_rgb(outline_hex),
        "text_color": _hex_to_rgb(text_hex),
        "stroke_width": int(stroke_w),
    }

    st.markdown("---")
    gen_btn = st.button("إنشاء سؤال جديد", type="primary", use_container_width=True)

    st.subheader("إنشاء مجموعة (Batch)")
    batch_n = st.number_input("عدد الأسئلة في المجموعة", min_value=2, max_value=100, value=10, step=1)
    batch_btn = st.button("إنشاء ملف ZIP للمجموعة", use_container_width=True)

rng = make_rng(seed)

def make_one():
    return generate_folding_challenge(rng, difficulty=difficulty, allow_diagonal=allow_diag, style=style)

if ("fold_qpack" not in st.session_state) or gen_btn:
    st.session_state.fold_qpack = make_one()

qp = st.session_state.get("fold_qpack")

if qp:
    colQ, colA = st.columns([2.1, 1.4])

    labels_ar = ["أ", "ب", "ج", "د"]  # Arabic option symbols

    with colQ:
        st.subheader("السؤال")
        show_image(qp["problem_img"])
        # Exact Arabic question text
        st.write(qp["prompt"])

    with colA:
        st.subheader("البدائل")
        chosen = st.radio("اختر الإجابة:", labels_ar, index=0, horizontal=True, label_visibility="collapsed")

        cols2 = st.columns(2)
        for i, im in enumerate(qp["choices_imgs"]):
            with cols2[i % 2]:
                show_image(im, caption=f"البديل {labels_ar[i]}")

        if st.button("تحقق من الإجابة"):
            if chosen == labels_ar[qp["correct_index"]]:
                st.success(f"صحيح. الإجابة: البديل {labels_ar[qp['correct_index']]}")
            else:
                st.error(f"غير صحيح. الإجابة الصحيحة: البديل {labels_ar[qp['correct_index']]}")
            st.markdown("**القاعدة:** " + qp["rule_desc"])

    st.markdown("---")
    st.subheader("تصدير السؤال")

    # Single export ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Images
        pb = io.BytesIO(); qp["problem_img"].save(pb, format="PNG")
        zf.writestr("problem.png", pb.getvalue())
        for i, im in enumerate(qp["choices_imgs"]):
            cb = io.BytesIO(); im.save(cb, format="PNG")
            zf.writestr(f"choice_{labels_ar[i]}.png", cb.getvalue())

        # Metadata (Arabic preserved)
        meta = {
            "type": qp["meta"]["type"],
            "folds": qp["meta"]["folds"],
            "axes": qp["meta"]["axes"],
            "punch": qp["meta"]["punch"],
            "prompt": qp["prompt"],
            "labels": labels_ar,
            "correct_label": labels_ar[qp["correct_index"]],
            "rule": qp["rule_desc"],
        }
        zf.writestr("question.json", json.dumps(meta, indent=2, ensure_ascii=False))

    st.download_button(
        "تنزيل ملف ZIP",
        data=buf.getvalue(),
        file_name=f"folding_{int(time.time())}.zip",
        mime="application/zip"
    )

    # Batch export
    if batch_btn:
        bbuf = io.BytesIO()
        with zipfile.ZipFile(bbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for k in range(int(batch_n)):
                # new RNG per item for reproducibility across batch
                local_rng = make_rng((seed or 0) + k + 1)
                local_pack = generate_folding_challenge(local_rng, difficulty=difficulty, allow_diagonal=allow_diag, style=style)
                qid = f"folding_{int(time.time()*1000)}_{k}"

                pb = io.BytesIO(); local_pack["problem_img"].save(pb, format="PNG")
                zf.writestr(f"{qid}/problem.png", pb.getvalue())

                for i, im in enumerate(local_pack["choices_imgs"]):
                    cb = io.BytesIO(); im.save(cb, format="PNG")
                    zf.writestr(f"{qid}/choice_{labels_ar[i]}.png", cb.getvalue())

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
