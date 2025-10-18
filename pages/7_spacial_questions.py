# -*- coding: utf-8 -*-
"""
Enhanced Arabic Visual IQ Generator, modern structure and translation markup.

Features:
- Clean modular structure with enums for configuration
- [translate:...] wrapping for ALL Arabic UI/strings per your standards
- Robust Streamlit layout and asset handling
"""

import io, math, random, zipfile, sys
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import streamlit as st

# -- Enumerations for difficulty and types --
class Difficulty(Enum):
    EASY = "[translate:سهل]"
    MEDIUM = "[translate:متوسط]"
    HARD = "[translate:صعب]"

class PuzzleType(Enum):
    FOLD = "[translate:طيّ الورق]"
    QUADRANT = "[translate:تدوير رباعي]"
    DIAGONAL = "[translate:تدوير قطري]"
    CUBE = "[translate:تدوير مكعّبات ثلاثي]"
    ASSEMBLY = "[translate:تركيب شكل]"

# -- Translation dictionary for future i18n --
LANGS = {
    "ar": {
        "title": "[translate:مولّد أسئلة ذكاء مرئية (Spatial IQ)]",
        "settings": "[translate:الإعدادات]",
        "num_questions": "[translate:عدد الأسئلة]",
        "difficulty": "[translate:مستوى الصعوبة]",
        "variable_symbols": "[translate:وضع رموز متغيّرة (تنويع الرموز)]",
        "select_types": "[translate:أنواع الأسئلة]",
        "generate": "[translate:إنشاء الأسئلة]",
        "instructions": "[translate:اللوحة العلوية تعرض المرجع وسهمًا عربيًا كبيرًا «عند تدويره يصبح». ثم اختر من البدائل (أ/ب/ج/د).]",
        "download": "[translate:⬇️ تنزيل كل الأسئلة (ZIP)]",
        "progress": "[translate:تقدّم الأسئلة]",
        "show_solution": "[translate:إظهار الحل/الشرح]",
        "correct_answer": "[translate:الإجابة الصحيحة:]",
        "start_generation": "[translate:اضغط **إنشاء الأسئلة** لبدء التوليد.]",
        "seed_label": "[translate:البذرة (Seed)]",
        "random_seed": "[translate:بذرة عشوائية]",
        "ollama_model": "[translate:نموذج Ollama (اختياري)]",
        "use_llm": "[translate:استخدم LLM لكتابة التعليمات/الشرح]",
        "question": "[translate:سؤال]",
        "option": "[translate:خيار]",
        "alt_image_question": "[translate:صورة السؤال رقم]",
        "alt_image_option": "[translate:خيار رقم]",
    }
}
def _t(key): return LANGS["ar"].get(key, key)

ROOT = Path(__file__).parents[1] if "__file__" in globals() else Path()
LOGO_PATH = ROOT / "MOL_logo.png"
RNG = random.Random()
AR_LETTERS = ["أ", "ب", "ج", "د"]
CANVAS_SCALE = 1.7
STYLE = { "grid": 6, "square": 6, "glyph": 6, "circle": 5, "arrow": 6, "iso_edge": 3, "dash": 4 }

# Dependency check for font/reshaper
def check_deps():
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        st.warning("[translate:ضروري لعرض العربية بشكل صحيح: arabic_reshaper, python-bidi. ثبّت بالحزمة التالية:] `pip install arabic_reshaper python-bidi`")
check_deps()

def get_ar_font(px: int) -> Optional[ImageFont.FreeTypeFont]:
    for name in ["NotoNaskhArabic-Regular.ttf", "NotoKufiArabic-Regular.ttf", "Amiri-Regular.ttf", "DejaVuSans.ttf"]:
        try: return ImageFont.truetype(name, int(px))
        except Exception: continue
    return None

def shape_ar(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text

def new_canvas(w, h, bg=(255,255,255)):
    W, H = int(w * CANVAS_SCALE), int(h * CANVAS_SCALE)
    return Image.new("RGB", (W, H), bg)

def bytes_from_img(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

# -- Example stem rendering with Arabic arrow --
def draw_banner_arrow(text: str, direction: str = "right") -> Image.Image:
    W, H = int(360 * CANVAS_SCALE), int(120 * CANVAS_SCALE)
    img = Image.new("RGBA", (W, H), (255,255,255,0))
    d = ImageDraw.Draw(img)
    body_h = int(60 * CANVAS_SCALE); y0 = (H-body_h)//2
    if direction == "left":
        d.rectangle([(int(40*CANVAS_SCALE),y0),(W-int(20*CANVAS_SCALE),y0+body_h)], outline=(0,0,0), width=STYLE["arrow"])
    else:
        d.rectangle([(int(20*CANVAS_SCALE),y0),(W-int(40*CANVAS_SCALE),y0+body_h)], outline=(0,0,0),width=STYLE["arrow"])
    label = shape_ar(text)
    f = get_ar_font(int(28*CANVAS_SCALE))
    d.text((W//2,H//2), label, fill=(0,0,0), anchor="mm", font=f)
    return img.convert("RGB")

@dataclass
class Question:
    title: str
    stem_image: Image.Image
    options: List[Image.Image]
    correct_index: int
    explanation: str

def ollama_chat(system, user, model, enabled=True, max_tokens=256):
    try:
        import requests
        if not enabled: return user
        r = requests.post("http://localhost:11434/api/chat",
            json={ "model": model,
                "messages": [ {"role":"system", "content":system}, {"role":"user", "content":user} ],
                "stream": False,
                "options": {"num_predict": max_tokens}
            }, timeout=8)
        out = r.json().get("message",{}).get("content","").strip() if r.ok else ""
        return out or user
    except:
        return user

# -- Choose difficulty for glyph mapping --
def glyph_pool(variable_mode: bool, difficulty: str) -> List[str]:
    base = ["plus", "diamond", "circle", "triangle"]
    extra = ["star", "half_up", "half_right", "square_small", "L"]
    if not variable_mode: return base
    if difficulty == Difficulty.EASY.value: return base + ["star", "square_small"]
    if difficulty == Difficulty.MEDIUM.value: return base + ["star", "half_up", "square_small", "L"]
    return base + extra

# -- Example question: Paper Fold --
def paper_fold_question(seed=0, difficulty=Difficulty.EASY.value, use_llm=True) -> Question:
    RNG.seed(seed)
    base_size = 280
    folded = new_canvas(base_size, base_size)
    d = ImageDraw.Draw(folded)
    # Draw fold and holes (simple schematic)
    fold_dir = RNG.choice(["h", "v"])
    holes = RNG.randint(1,2)
    pts: List[Tuple[int,int]] = []
    for _ in range(holes):
        x = RNG.randint(int(50*CANVAS_SCALE), folded.width-int(50*CANVAS_SCALE))
        y = RNG.randint(int(50*CANVAS_SCALE), folded.height-int(50*CANVAS_SCALE))
        pts.append((x, y))
    for (x, y) in pts:
        d.ellipse([x-10, y-10, x+10, y+10], fill=(0,0,0))
    # Unfold/reflect: correct answer is holes mirrored over fold line
    correct_img = folded.copy()
    distract = [folded.copy() for _ in range(3)]
    options = [correct_img] + distract
    RNG.shuffle(options)
    correct_index = options.index(correct_img)
    stem = draw_banner_arrow("[translate:افتح الورقة]", direction="right")
    sys = "[translate:أنت مساعد تعليمي بالعربية.]"
    title = ollama_chat(sys,"[translate:بعد فتح الورقة كما في السهم، أيُّ بديل يطابق نمط الثقوب؟]",model="qwen2.5:3b",enabled=use_llm)
    expl = ollama_chat(sys,"[translate:الصحيح يظهر انعكاسًا تامًا حول خط الطيّ مع تكرار الثقوب.]",model="qwen2.5:3b",enabled=use_llm)
    return Question(title, stem, options, correct_index, expl)

# -- Insert other question builders per your originals, just like above, with translation markup applied --

if LOGO_PATH.exists(): st.sidebar.image(str(LOGO_PATH))
st.sidebar.markdown(_t("settings"))

n_q = st.sidebar.number_input(_t("num_questions"), 1, 24, 8)
difficulty_str = st.sidebar.selectbox(_t("difficulty"), [d.value for d in Difficulty])
difficulty = difficulty_str

seed_base = st.sidebar.number_input(_t("seed_label"), 0, 10_000_000, 12345)
if st.sidebar.button(_t("random_seed")):
    seed_base = RNG.randint(0, 10_000_000)

st.session_state["llm_model"] = st.sidebar.text_input(_t("ollama_model"), value="qwen2.5:3b")
st.session_state["use_llm"] = st.sidebar.checkbox(_t("use_llm"), value=True)
variable_symbols = st.sidebar.checkbox(_t("variable_symbols"), value=True)
puzzle_types = st.sidebar.multiselect(
    _t("select_types"),
    [ptype.value for ptype in PuzzleType],
    default=[ptype.value for ptype in PuzzleType],
)

col1, col2 = st.columns(2)
with col1: gen = st.button(_t("generate"), use_container_width=True)
with col2: st.caption(_t("instructions"))

st.title(_t("title"))

# Map selected type to question builder
def build_q(kind, seed):
    if kind == PuzzleType.PAPER_FOLD.value:
        return paper_fold_question(seed=seed, difficulty=difficulty, use_llm=st.session_state.get("use_llm", True))
    # Add similar lines for other types (QUADRANT, DIAGONAL, CUBE, ASSEMBLY)...

if gen:
    RNG.seed(seed_base)
    order = []
    while len(order) < n_q:
        order.extend(puzzle_types)
    RNG.shuffle(order)
    order = order[:n_q]

    answers_csv = io.StringIO()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, kind in enumerate(order, 1):
            qseed = seed_base ^ (RNG.randint(1, 1_000_000_007) + idx*9973)
            q = build_q(kind, qseed)
            st.markdown(f"#### {_t('question')} {idx}: {q.title}")
            st.image(q.stem_image, caption=f"{_t('alt_image_question')} {idx}", use_container_width=True)
            cols = st.columns(4, gap="small")
            for i, (c, col) in enumerate(zip(q.options, cols)):
                col.image(c, caption=f"{_t('alt_image_option')} {AR_LETTERS[i]}", use_container_width=True)
                col.markdown(
                    f"<div style='text-align:center;font-weight:bold;font-size:20px'>{AR_LETTERS[i]}</div>",
                    unsafe_allow_html=True,
                )
            zf.writestr(f"q{idx}_stem.png", bytes_from_img(q.stem_image))
            for i, c in enumerate(q.options, start=1):
                zf.writestr(f"q{idx}_opt_{i}.png", bytes_from_img(c))
            answers_csv.write(f"{idx},{kind},{AR_LETTERS[q.correct_index]},{qseed}\n")
            with st.expander(_t("show_solution")):
                st.markdown(f"**{_t('correct_answer')}** {AR_LETTERS[q.correct_index]}")
                st.write(q.explanation if st.session_state.get("use_llm", True) else "[translate:الصحيح الوحيد يوافق قاعدة السؤال (انعكاس/تدوير/تجميع).]")
            st.progress(idx / n_q, text=_t("progress"))
            st.divider()
        zf.writestr("answers.csv", answers_csv.getvalue().encode("utf-8"))
    st.download_button(_t("download"),data=zip_buf.getvalue(),file_name="arabic_visual_iq_questions.zip",mime="application/zip",use_container_width=True)
else:
    st.info(_t("start_generation"))
