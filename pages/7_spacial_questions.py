# -*- coding: utf-8 -*-
"""
Arabic Visual IQ Generator — Pro version for Streamlit
"""

import io, math, random, zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional
import enum
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import streamlit as st

# =================== CONFIGURATION & ENUMS ===================
class Difficulty(enum.Enum):
    EASY = "سهل"
    MEDIUM = "متوسط"
    HARD = "صعب"

class PuzzleType(enum.Enum):
    PAPER_FOLD = "طيّ الورق"
    # Add other types...

LANGS = {
    "ar": {
        "title": "مولّد أسئلة ذكاء مرئية (Spatial IQ)",
        "settings": "الإعدادات",
        "num_questions": "عدد الأسئلة",
        "difficulty": "مستوى الصعوبة",
        "variable_symbols": "وضع رموز متغيّرة",
        "select_types": "أنواع الأسئلة",
        "generate": "🚀 إنشاء الأسئلة",
        "instructions": "اللوحة العلوية تعرض المرجع وسهمًا عربيًا. اختر من البدائل (أ/ب/ج/د).",
        "download": "⬇️ تنزيل كل الأسئلة (ZIP)",
        "progress": "تقدّم الأسئلة",
        "show_solution": "إظهار الحل/الشرح",
        "correct_answer": "الإجابة الصحيحة:",
        "start_generation": "اضغط **إنشاء الأسئلة** لبدء التوليد.",
        "seed_label": "البذرة (Seed)",
        "random_seed": "🎲 بذرة عشوائية",
        "ollama_model": "نموذج Ollama (اختياري)",
        "use_llm": "استخدم LLM لكتابة التعليمات/الشرح",
        "question": "سؤال",
        "option": "خيار",
        "alt_image_question": "صورة السؤال رقم",
        "alt_image_option": "خيار رقم",
    }
}
def _t(key: str) -> str:
    return LANGS["ar"].get(key, key)

RNG = random.Random()
AR_LETTERS = ["أ", "ب", "ج", "د"]

# =================== DRAWING, SHAPE, and ARROW UTILS ===================
CANVAS_SCALE = 1.7

def new_canvas(w, h, bg=(255,255,255)):
    return Image.new("RGB", (w, h), bg)

def _finalize_for_display(img: Image.Image, target_size: Optional[Tuple[int, int]] = None):
    if target_size:
        return img.resize(target_size, Image.LANCZOS)
    return img

def draw_paper_arrow(width=72, height=32):
    arr = Image.new("RGBA", (width, height), (255,255,255,0))
    dr = ImageDraw.Draw(arr)
    dr.rectangle([8, height//3+6, width-32, 2*height//3+2], outline="black", width=2)
    dr.polygon([(width-32, height//2), (width-8, height//3-8), (width-8, 2*height//3+8)],
                fill="white", outline="black")
    return arr

def faint_hint_box(side:int=60, text="?"):
    img = new_canvas(side,side)
    d = ImageDraw.Draw(img)
    d.rectangle([5,5,side-5,side-5], outline=(180,180,180), width=3)
    fnt = None
    for fnt_name in ["NotoNaskhArabic-Regular.ttf","arial.ttf","DejaVuSans-Bold.ttf"]:
        try:
            fnt = ImageFont.truetype(fnt_name, int(side*0.8))
            break
        except Exception: continue
    if not fnt:
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

def paper_fold_question(seed: int = 0, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    RNG.seed(seed)
    # === Strict tall rectangle, good padding
    W, H = 90, 130
    MARGIN = 12
    HOLE_RADIUS = 9
    LINE_WIDTH = 3

    # Step 1 - Stem (folded rectangle)
    stem_img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(stem_img)
    d.rectangle([MARGIN, MARGIN, W-MARGIN, H-MARGIN], outline="black", width=LINE_WIDTH)
    fold = (H + MARGIN) // 2
    d.line([MARGIN, fold, W-MARGIN, fold], fill="black", width=2)
    for x in range(MARGIN+6, W-MARGIN, 9):
        d.line([(x, fold-22), (x + 5, fold-22)], fill="gray", width=1)
    hole_x, hole_y = W-MARGIN-14, fold-17
    d.ellipse([hole_x-HOLE_RADIUS, hole_y-HOLE_RADIUS, hole_x+HOLE_RADIUS, hole_y+HOLE_RADIUS], outline="black", width=2)

    # Arrow and faint question mark - boxed outside
    stem_img = _finalize_for_display(stem_img, (110, 170))
    arrow = draw_paper_arrow()
    qbox = faint_hint_box(46, "?")
    stem = Image.new("RGB", (330, 180), "white")
    stem.paste(stem_img, (0,5))
    stem.paste(arrow, (120,65), arrow)
    stem.paste(qbox, (250,15))

    # Step 2 - Options (all answers in 110x170)
    reflected_y = 2*fold - hole_y
    correct = [ (hole_x, hole_y), (hole_x, reflected_y) ]
    wrong1 = [ (hole_x, hole_y) ]
    wrong2 = [ (hole_x, reflected_y) ]
    wrong3 = [ (hole_x - 22, hole_y) ]

    def opt_img(holes):
        ans = Image.new("RGB", (W, H), "white")
        dd = ImageDraw.Draw(ans)
        dd.rectangle([MARGIN, MARGIN, W-MARGIN, H-MARGIN], outline="black", width=LINE_WIDTH)
        for x, y in holes:
            dd.ellipse([x-HOLE_RADIUS, y-HOLE_RADIUS, x+HOLE_RADIUS, y+HOLE_RADIUS], outline="black", width=2)
        return _finalize_for_display(ans, (110, 170))

    opts = [opt_img(correct), opt_img(wrong1), opt_img(wrong2), opt_img(wrong3)]
    idxs = list(range(4))
    RNG.shuffle(idxs)
    options = [opts[i] for i in idxs]
    correct_index = idxs.index(0)

    sys = "أنت مساعد تعليمي بالعربية."
    title = "انظر إلى المثال وحدد الإجابة الصحيحة كما في التعليمات. ما رمز البديل المطابق بعد فتح الورقة؟"
    expl = "الإجابة الصحيحة تُظهر الثقب الأصلي مع الثقب المعكوس أسفل خط الطي."

    return Question(
        title=title,
        stem_image=stem,
        options=options,
        correct_index=correct_index,
        explanation=expl
    )

# ---- Streamlit UI ----

st.sidebar.markdown(f"### {_t('settings')}")
col_left, col_right = st.sidebar.columns(2)
with col_left:
    n_q = st.number_input(_t("num_questions"), 1, 12, 4)
with col_right:
    difficulty_str = st.selectbox(_t("difficulty"), [d.value for d in Difficulty])

seed_base = st.sidebar.number_input(_t("seed_label"), 0, 10_000_000, 12345)
if st.sidebar.button(_t("random_seed")):
    seed_base = RNG.randint(0, 10_000_000)

st.session_state["llm_model"] = st.sidebar.text_input(_t("ollama_model"), value="qwen2.5:3b")
st.session_state["use_llm"] = st.sidebar.checkbox(_t("use_llm"), value=True)

# Only paper fold for this minimal example; add more as needed
puzzle_types = st.sidebar.multiselect(
    _t("select_types"),
    [PuzzleType.PAPER_FOLD.value],
    default=[PuzzleType.PAPER_FOLD.value],
)

col1, col2 = st.columns(2)
with col1:
    gen = st.button(_t("generate"), use_container_width=True)
with col2:
    st.caption(_t("instructions"))

st.title(_t("title"))

def build_by_type(kind: str, seed: int) -> Question:
    # Only paper fold for this demo
    return paper_fold_question(seed=seed, difficulty=difficulty_str, use_llm=st.session_state.get("use_llm", True))

if gen:
    RNG.seed(seed_base)
    order: List[str] = []
    while len(order) < n_q:
        order.extend(puzzle_types)
    RNG.shuffle(order)
    order = order[:n_q]

    answers_csv = io.StringIO()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, kind in enumerate(order, 1):
            qseed = seed_base ^ (RNG.randint(1, 1_000_000_007) + idx * 9973)
            q = build_by_type(kind, qseed)

            st.markdown(f"#### {_t('question')} {idx}: {q.title}")
            st.image(q.stem_image, use_container_width=False, caption=f"{_t('alt_image_question')} {idx}")
            cols = st.columns(4, gap="large")
            for i, (c, col) in enumerate(zip(q.options, cols)):
                col.image(c, use_container_width=False)
                col.markdown(
                    f"<div style='text-align:center;font-size:20px'>{AR_LETTERS[i]}</div>", unsafe_allow_html=True
                )

            zf.writestr(f"q{idx}_stem.png", c.tobytes())
            for i, opt in enumerate(q.options, start=1):
                buf = io.BytesIO()
                opt.save(buf, format="PNG")
                zf.writestr(f"q{idx}_opt_{i}.png", buf.getvalue())
            answers_csv.write(f"{idx},{kind},{AR_LETTERS[q.correct_index]},{qseed}\n")

            with st.expander(_t("show_solution")):
                st.markdown(f"**{_t('correct_answer')}** {AR_LETTERS[q.correct_index]}")
                st.write(q.explanation if st.session_state.get("use_llm", True) else "الصحيح الوحيد يوافق قاعدة السؤال (انعكاس/تدوير/تجميع).")

            st.progress(idx / n_q, text=_t("progress"))
            st.divider()
        zf.writestr("answers.csv", answers_csv.getvalue().encode("utf-8"))
    st.download_button(
        _t("download"),
        data=zip_buf.getvalue(),
        file_name="arabic_visual_iq_questions.zip",
        mime="application/zip",
        use_container_width=True,
    )
else:
    st.info(_t("start_generation"))
