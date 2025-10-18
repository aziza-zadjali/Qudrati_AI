# -*- coding: utf-8 -*-
"""
Full-Featured Arabic Visual/Spatial IQ Generator for Streamlit
Covers: paper folding, 2D quadrant rotation, 2D diagonal rotation, 3D cube rotation, shape assembly
"""

import io, math, random, zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional
import enum
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import streamlit as st

# ==== Enums/Constants ====
class Difficulty(enum.Enum):
    EASY = "سهل"
    MEDIUM = "متوسط"
    HARD = "صعب"

class PuzzleType(enum.Enum):
    PAPER_FOLD = "طيّ الورق"
    QUADRANT_ROTATE = "تدوير رباعي"
    DIAGONAL_ROTATE = "تدوير قطري"
    CUBE_ROTATE = "تدوير مكعبات"
    SHAPE_ASSEMBLY = "تركيب شكل"

LANGS = {
    "ar": {
        "title": "مولّد أسئلة ذكاء مرئية (Spatial IQ)",
        "settings": "الإعدادات",
        "num_questions": "عدد الأسئلة",
        "difficulty": "مستوى الصعوبة",
        "variable_symbols": "وضع رموز متغيّرة",
        "select_types": "أنواع الأسئلة",
        "generate": "🚀 إنشاء الأسئلة",
        "instructions": "اختر الإجابة الصحيحة بعد تأمّل الصور وحسب التعليمات.",
        "download": "⬇️ تنزيل كل الأسئلة (ZIP)",
        "progress": "تقدم الأسئلة",
        "show_solution": "إظهار الحل/الشرح",
        "correct_answer": "الإجابة الصحيحة:",
        "start_generation": "اضغط إنشاء الأسئلة لبدء التوليد.",
        "seed_label": "البذرة (Seed)",
        "random_seed": "بذرة عشوائية",
        "ollama_model": "نموذج Ollama (اختياري)",
        "use_llm": "استخدم LLM لكتابة التعليمات/الشرح",
        "question": "سؤال",
        "option": "خيار",
        "alt_image_question": "صورة السؤال رقم",
        "alt_image_option": "خيار رقم",
    }
}
def _t(key: str) -> str:  # All-AR
    return LANGS["ar"].get(key, key)

RNG = random.Random()
AR_LETTERS = ["أ", "ب", "ج", "د"]

# ==== Drawing & Utility ====
def new_canvas(w, h, bg=(255,255,255)):
    return Image.new("RGB", (w, h), bg)

def _finalize_for_display(img: Image.Image, size):
    return img.resize(size, Image.LANCZOS)

def draw_paper_arrow(width=72, height=32):
    arr = Image.new("RGBA", (width, height), (255,255,255,0))
    dr = ImageDraw.Draw(arr)
    dr.rectangle([8, height//3+6, width-32, 2*height//3+2], outline="black", width=2)
    dr.polygon([(width-32, height//2), (width-8, height//3-8), (width-8, 2*height//3+8)],
                fill="white", outline="black")
    return arr

def faint_hint_box(side=60, text="?"):
    img = new_canvas(side,side)
    d = ImageDraw.Draw(img)
    d.rectangle([5,5,side-5,side-5], outline=(180,180,180), width=3)
    fnt = ImageFont.load_default()
    w, h = d.textsize(text, font=fnt)
    d.text(((side-w)//2, (side-h)//2), text, fill=(170,170,170), font=fnt)
    return img

@dataclass
class Question:
    title: str
    stem_image: Image.Image
    options: List[Image.Image]
    correct_index: int
    explanation: str

# ==== Type 1: Paper Folding ====
def paper_fold_question(seed=0, difficulty="سهل", use_llm=True) -> Question:
    RNG.seed(seed)
    # Tall rectangle pro aspect
    W, H = 90, 130
    MARGIN = 12
    HOLE_RADIUS = 9

    stem_img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(stem_img)
    d.rectangle([MARGIN, MARGIN, W-MARGIN, H-MARGIN], outline="black", width=3)
    fold = (H + MARGIN) // 2
    d.line([MARGIN, fold, W-MARGIN, fold], fill="black", width=2)
    for x in range(MARGIN+6, W-MARGIN, 9):
        d.line([(x, fold-20), (x + 5, fold-20)], fill="gray", width=1)
    hole_x, hole_y = W-MARGIN-14, fold-17
    d.ellipse([hole_x-HOLE_RADIUS, hole_y-HOLE_RADIUS, hole_x+HOLE_RADIUS, hole_y+HOLE_RADIUS], outline="black", width=2)
    stem_img = _finalize_for_display(stem_img, (110, 170))
    arrow = draw_paper_arrow()
    qbox = faint_hint_box(46, "?")
    stem = Image.new("RGB", (330, 180), "white")
    stem.paste(stem_img, (0,5))
    stem.paste(arrow, (120,65), arrow)
    stem.paste(qbox, (250,15))

    # Options
    reflected_y = 2*fold - hole_y
    correct = [ (hole_x, hole_y), (hole_x, reflected_y) ]
    wrong1 = [ (hole_x, hole_y) ]
    wrong2 = [ (hole_x, reflected_y) ]
    wrong3 = [ (hole_x - 22, hole_y) ]

    def opt_img(holes):
        ans = Image.new("RGB", (W, H), "white")
        dd = ImageDraw.Draw(ans)
        dd.rectangle([MARGIN, MARGIN, W-MARGIN, H-MARGIN], outline="black", width=3)
        for x, y in holes:
            dd.ellipse([x-HOLE_RADIUS, y-HOLE_RADIUS, x+HOLE_RADIUS, y+HOLE_RADIUS], outline="black", width=2)
        return _finalize_for_display(ans, (110, 170))

    opts = [opt_img(correct), opt_img(wrong1), opt_img(wrong2), opt_img(wrong3)]
    idxs = list(range(4)); RNG.shuffle(idxs)
    options = [opts[i] for i in idxs]; correct_index = idxs.index(0)

    return Question(
        title="ما رمز البديل الصحيح بعد إعادة فتح الورقة؟",
        stem_image=stem, options=options, correct_index=correct_index,
        explanation="الإجابة الصحيحة تُظهر الثقب الأصلي مع المعكوس."
    )

# ==== Type 2: Quadrant Rotation ====
def quadrant_rotation_question(seed=0, difficulty="سهل", use_llm=True) -> Question:
    RNG.seed(seed)
    S = 150
    base = new_canvas(S,S)
    d = ImageDraw.Draw(base)
    d.rectangle([15,15,S-15,S-15], outline="black", width=2)
    d.line([S//2,15,S//2,S-15], fill="black", width=2)
    d.line([15,S//2,S-15,S//2], fill="black", width=2)
    # Simple glyphs
    d.ellipse([95,40,115,60], outline="black", width=3)
    d.ellipse([95,105,115,125], outline="black", width=3)
    d.rectangle([30,40,50,60], outline="black", width=3)
    d.rectangle([30,105,50,125], outline="black", width=3)
    stem = base
    angle = RNG.choice([90,180,270]); arr_img = base.rotate(-angle)
    opts = [arr_img, base.rotate(-90), base.rotate(-180), base.rotate(-270)]
    idxs = list(range(4)); RNG.shuffle(idxs)
    options = [opts[i] for i in idxs]; correct_index = idxs.index(0)
    return Question(
        title=f"أي بديل يطابق الشكل بعد تدويره {angle}°؟",
        stem_image=_finalize_for_display(stem, (180,180)),
        options=[_finalize_for_display(opt,(180,180)) for opt in options],
        correct_index=correct_index,
        explanation=f"الصحيح هو التدوير {angle}°."
    )

# ==== Type 3: Diagonal Rotation ====
def diagonal_rotation_question(seed=0, difficulty="سهل", use_llm=True) -> Question:
    RNG.seed(seed)
    S = 150
    base = new_canvas(S,S)
    d = ImageDraw.Draw(base)
    d.line([15,15,S-15,S-15], fill="black", width=2)
    d.line([15,S-15,S-15,15], fill="black", width=2)
    # Distinct glyphs diagonal
    d.ellipse([30,30,50,50], outline="black", width=3)
    d.rectangle([100,100,120,120], outline="black", width=3)
    d.rectangle([30,100,50,120], outline="black", width=3)
    d.ellipse([100,30,120,50], outline="black", width=3)
    angle = RNG.choice([90,180,270]); arr_img = base.rotate(-angle)
    opts = [arr_img, base.rotate(-90), base.rotate(-180), base.rotate(-270)]
    idxs = list(range(4)); RNG.shuffle(idxs)
    options = [opts[i] for i in idxs]; correct_index = idxs.index(0)
    return Question(
        title=f"أي بديل يطابق الشكل بعد تدويره {angle}°؟",
        stem_image=_finalize_for_display(base, (180,180)),
        options=[_finalize_for_display(opt,(180,180)) for opt in options],
        correct_index=correct_index,
        explanation=f"الصحيح هو التدوير {angle}°."
    )

# ==== Type 4: Cube rotation (simple, not full isometric polycube) ====
def cubes_rotation_question(seed=0, difficulty="سهل", use_llm=True) -> Question:
    # Placeholder demo: show four cubes, rotate via PIL, mark correct
    S = 110
    face = new_canvas(S,S)
    d = ImageDraw.Draw(face)
    d.rectangle([10,10,100,100], outline="black", width=2)
    d.rectangle([50,10,100,60], outline="black", width=2)
    d.rectangle([10,60,60,110], outline="black", width=2)
    face_rot90 = face.rotate(90); face_rot180 = face.rotate(180); face_rot270 = face.rotate(270)
    opts = [face, face_rot90, face_rot180, face_rot270]
    idxs = list(range(4)); RNG.shuffle(idxs)
    options = [opts[i] for i in idxs]; correct_index = idxs.index(0)
    return Question(
        title="أي مجسم يطابق بعد التدوير؟",
        stem_image=_finalize_for_display(face, (140,140)),
        options=[_finalize_for_display(opt,(140,140)) for opt in options],
        correct_index=correct_index,
        explanation="الصحيح هو صورة المكعب بعد التدوير."
    )

# ==== Type 5: Shape Assembly (Tangram-style) ====
def shape_assembly_question(seed=0, use_llm=True) -> Question:
    S = 160
    base = new_canvas(S,S)
    d = ImageDraw.Draw(base)
    # Draw square; options will have different shapes
    d.rectangle([30,30,130,130], outline="black", width=3)
    option1 = new_canvas(S,S); d1=ImageDraw.Draw(option1)
    d1.polygon([ (30,130), (80,30), (130,130) ], outline="black", width=3)   # Triangle
    d1.rectangle([60,60,100,100], outline="black", width=3)
    option2 = new_canvas(S,S); d2=ImageDraw.Draw(option2)
    d2.rectangle([30,30,85,85], outline="black", width=3)
    d2.rectangle([85,85,130,130], outline="black", width=3)
    option3 = new_canvas(S,S); d3=ImageDraw.Draw(option3)
    d3.ellipse([40,40,100,100], outline="black", width=3)
    d3.rectangle([100,100,130,130], outline="black", width=3)
    option4 = new_canvas(S,S); d4=ImageDraw.Draw(option4)
    d4.polygon([ (40,40), (90,30), (130,80), (30,130) ], outline="black", width=3)
    opts = [option1, option2, option3, option4]
    idxs = list(range(4)); RNG.shuffle(idxs)
    options = [opts[i] for i in idxs]; correct_index = idxs.index(0)
    return Question(
        title="أي مجموعة قطع تكون الشكل الأعلى؟",
        stem_image=_finalize_for_display(base, (180,180)),
        options=[_finalize_for_display(opt,(180,180)) for opt in options],
        correct_index=correct_index,
        explanation="الصحيح هو الخيار المناسب لبناء الشكل الأصلي."
    )

# ==== Build Factory ====
def build_by_type(kind:str, seed:int) -> Question:
    if kind == PuzzleType.PAPER_FOLD.value:
        return paper_fold_question(seed=seed)
    if kind == PuzzleType.QUADRANT_ROTATE.value:
        return quadrant_rotation_question(seed=seed)
    if kind == PuzzleType.DIAGONAL_ROTATE.value:
        return diagonal_rotation_question(seed=seed)
    if kind == PuzzleType.CUBE_ROTATE.value:
        return cubes_rotation_question(seed=seed)
    if kind == PuzzleType.SHAPE_ASSEMBLY.value:
        return shape_assembly_question(seed=seed)
    raise ValueError("نوع غير معروف!")

# ==== Streamlit UI ====
st.sidebar.markdown(f"### {_t('settings')}")
col_left, col_right = st.sidebar.columns(2)
with col_left:
    n_q = st.number_input(_t("num_questions"), 1, 10, 4)
with col_right:
    difficulty_str = st.selectbox(_t("difficulty"), [d.value for d in Difficulty])
seed_base = st.sidebar.number_input(_t("seed_label"), 0, 10_000_000, 12345)
if st.sidebar.button(_t("random_seed")):
    seed_base = RNG.randint(0, 10_000_000)
st.session_state["llm_model"] = st.sidebar.text_input(_t("ollama_model"), value="qwen2.5:3b")
st.session_state["use_llm"] = st.sidebar.checkbox(_t("use_llm"), value=True)
puzzle_types = st.sidebar.multiselect(
    _t("select_types"),
    [p.value for p in PuzzleType],
    default=[p.value for p in PuzzleType]
)
col1, col2 = st.columns(2)
with col1:
    gen = st.button(_t("generate"), use_container_width=True)
with col2:
    st.caption(_t("instructions"))
st.title(_t("title"))

if gen:
    RNG.seed(seed_base)
    order = []
    while len(order) < n_q:
        order.extend(puzzle_types)
    RNG.shuffle(order); order = order[:n_q]
    answers_csv = io.StringIO(); zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for idx, kind in enumerate(order,1):
            qseed = seed_base ^ (RNG.randint(1, 100000007) + idx*9973)
            q = build_by_type(kind, qseed)
            st.markdown(f"#### سؤال {idx}: {q.title}")
            st.image(q.stem_image, use_container_width=False, caption=f"{_t('alt_image_question')} {idx}")
            cols = st.columns(4, gap="large")
            for i, (c, col) in enumerate(zip(q.options, cols)):
                col.image(c, use_container_width=False)
                col.markdown("<div style='text-align:center;font-size:20px'>{}</div>".format(AR_LETTERS[i]), unsafe_allow_html=True)
            # Export to ZIP
            buf = io.BytesIO(); q.stem_image.save(buf, format="PNG"); zf.writestr(f"q{idx}_stem.png", buf.getvalue())
            for i, opt in enumerate(q.options, start=1):
                buf2 = io.BytesIO(); opt.save(buf2, format="PNG"); zf.writestr(f"q{idx}_opt_{i}.png", buf2.getvalue())
            answers_csv.write(f"{idx},{kind},{AR_LETTERS[q.correct_index]},{qseed}\n")
            with st.expander(_t("show_solution")):
                st.markdown(f"**{_t('correct_answer')}** {AR_LETTERS[q.correct_index]}")
                st.write(q.explanation)
            st.progress(idx / n_q, text=_t("progress")); st.divider()
        zf.writestr("answers.csv", answers_csv.getvalue().encode("utf-8"))
    st.download_button(_t("download"), data=zip_buf.getvalue(),
        file_name="arabic_visual_iq_questions.zip",
        mime="application/zip",
        use_container_width=True)
else:
    st.info(_t("start_generation"))
