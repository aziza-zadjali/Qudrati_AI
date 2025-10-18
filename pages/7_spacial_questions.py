# spatial_iq_generator.py
# -*- coding: utf-8 -*-
"""
[translate:مولّد أسئلة الذكاء البصري والمكاني - محسّن وحديث]
"""

import io, math, random, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import enum
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import streamlit as st

# =================== ENUMS ===================
class Difficulty(enum.Enum):
    EASY = "[translate:سهل]"
    MEDIUM = "[translate:متوسط]"
    HARD = "[translate:صعب]"

class PuzzleType(enum.Enum):
    PAPER_FOLD = "[translate:طيّ الورق]"
    QUADRANT_ROTATE = "[translate:تدوير رباعي]"
    DIAGONAL_ROTATE = "[translate:تدوير قطري]"
    CUBE_ROTATE = "[translate:تدوير مكعّبات ثلاثي]"
    SHAPE_ASSEMBLY = "[translate:تركيب شكل]"
    SYMBOL_GRID = "[translate:شبكة الرموز]"
    ANALOGY = "[translate:علاقة/تناظر]" # Example of new/modern style
    COMPLETION = "[translate:إكمال النمط]" # Example of new/modern style

# =================== TRANSLATION HELPERS ===================
LANGS = {
    "ar": {
        "title": "[translate:مولّد أسئلة ذكاء مرئية (Spatial IQ)]",
        "settings": "[translate:الإعدادات]",
        "num_questions": "[translate:عدد الأسئلة]",
        "difficulty": "[translate:مستوى الصعوبة]",
        "variable_symbols": "[translate:وضع رموز متغيّرة (تنويع الرموز)]",
        "select_types": "[translate:أنواع الأسئلة]",
        "generate": "[translate:إنشاء الأسئلة]",
        "instructions": "[translate:اختر الإجابة الصحيحة بعد تأمّل الصور وحسب التعليمات.]",
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
        "modern_examples": "[translate:أنماط حديثة]",
    }
}# ========== PART 2: Drawing & Shape Logic ==========

STYLE = {
    "grid": 6,
    "square": 6,
    "glyph": 6,
    "circle": 5,
    "arrow": 6,
    "diamond": 6,
    "half": 4,
    "iso_edge": 3,
    "dash": 4,
}

AR_LETTERS = ["أ", "ب", "ج", "د"]
CANVAS_SCALE = 1.7

def get_ar_font(px: int) -> Optional[ImageFont.FreeTypeFont]:
    for name in ["NotoNaskhArabic-Regular.ttf", "NotoKufiArabic-Regular.ttf", "Amiri-Regular.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, int(px))
        except Exception:
            continue
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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def dashed_line(draw: ImageDraw.ImageDraw, start, end, dash_len=10, gap_len=8, fill=(0,0,0), width: Optional[int]=None):
    if width is None:
        width = STYLE["dash"]
    x1, y1 = start
    x2, y2 = end
    total_len = math.hypot(x2-x1, y2-y1)
    if total_len == 0: return
    dx = (x2-x1)/total_len; dy = (y2-y1)/total_len; dist = 0.0
    while dist < total_len:
        s, e = dist, min(dist+dash_len, total_len)
        xs, ys = x1+dx*s, y1+dy*s; xe, ye = x1+dx*e, y1+dy*e
        draw.line((xs,ys,xe,ye), fill=fill, width=width)
        dist += dash_len + gap_len

def poly_with_width(draw: ImageDraw.ImageDraw, pts, fill=None, outline=(0,0,0), width=1):
    if fill is not None: draw.polygon(pts, fill=fill)
    if outline is not None and width > 0:
        for i in range(len(pts)):
            a, b = pts[i], pts[(i+1)%len(pts)]
            draw.line([a, b], fill=outline, width=width)

def draw_square(draw: ImageDraw.ImageDraw, xy, size, outline=(0,0,0), width: Optional[int]=None):
    if width is None: width = STYLE["square"]
    x, y = xy
    draw.rectangle([x, y, x+size, y+size], outline=outline, width=width)

def draw_circle(draw: ImageDraw.ImageDraw, center, r, fill=None, outline=(0,0,0), width: Optional[int]=None):
    if width is None: width = STYLE["circle"]
    cx, cy = center
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill, outline=outline, width=width)

def draw_plus(draw: ImageDraw.ImageDraw, center, size, width: Optional[int]=None, fill=(0,0,0)):
    if width is None: width = STYLE["glyph"]
    cx, cy = center; s = size // 2
    draw.line((cx-s,cy,cx+s,cy), fill=fill, width=width)
    draw.line((cx,cy-s,cx,cy+s), fill=fill, width=width)

def draw_diamond(draw: ImageDraw.ImageDraw, center, size, fill=None, outline=(0,0,0), width: Optional[int]=None):
    if width is None: width = STYLE["diamond"]
    cx, cy = center; s = size // 2
    pts = [(cx, cy-s), (cx+s, cy), (cx, cy+s), (cx-s, cy)]
    poly_with_width(draw, pts, fill=fill, outline=outline, width=width)

def draw_L(draw: ImageDraw.ImageDraw, center, size, outline=(0,0,0), width: Optional[int]=None):
    if width is None: width = STYLE["glyph"]
    cx, cy = center; s = size
    draw.line((cx-s//2, cy-s//2, cx-s//2, cy+s//2), fill=outline, width=width)
    draw.line((cx-s//2, cy+s//2, cx+s//2, cy+s//2), fill=outline, width=width)

def draw_half_circle(draw: ImageDraw.ImageDraw, center, r, orientation: str = "up", outline=(0,0,0), width: Optional[int]=None):
    if width is None: width = STYLE["half"]
    cx, cy = center; bbox = [cx-r, cy-r, cx+r, cy+r]
    start_map = {"up":0, "right":90, "down":180, "left":270}
    end_map = {"up":180, "right":270, "down":360, "left":90}
    draw.pieslice(bbox, start=start_map[orientation], end=end_map[orientation], fill=None, outline=outline, width=width)

def draw_star(draw: ImageDraw.ImageDraw, center, r, points=5, outline=(0,0,0), width: Optional[int]=None):
    if width is None: width = STYLE["glyph"]
    cx, cy = center; pts = []
    for i in range(points*2):
        angle = math.pi * i / points
        rr = r if i%2==0 else r*0.45
        pts.append((cx+rr*math.cos(angle-math.pi/2), cy+rr*math.sin(angle-math.pi/2)))
    poly_with_width(draw, pts, fill=None, outline=outline, width=width)

# ...additional draw functions for modern spatial IQ figures as needed...

# -- Banner arrow for Arab questions --
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

# -- Helper for stacking images horizontally in question stems --
def hstack(*imgs: Image.Image, pad: int = 16, bg=(255,255,255)) -> Image.Image:
    H = max(im.height for im in imgs)
    W = sum(im.width for im in imgs) + int(pad*CANVAS_SCALE)*(len(imgs)-1)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for im in imgs:
        y = (H-im.height)//2
        out.paste(im, (x, y)); x += im.width + int(pad*CANVAS_SCALE)
    return out

# -- Finalize display for crispness --
def _finalize_for_display(img: Image.Image, target_size: Optional[Tuple[int, int]] = None):
    if target_size:
        return img.resize((int(target_size[0]), int(target_size[1])), Image.LANCZOS)
    if CANVAS_SCALE == 1.0: return img
    return img.resize((int(img.width / CANVAS_SCALE), int(img.height / CANVAS_SCALE)), Image.LANCZOS)

# -- Use these helpers in the next part to assemble classic and modern questions! --
# ========== PART 3: Question Builder Functions ==========

@dataclass
class Question:
    title: str
    stem_image: Image.Image
    options: List[Image.Image]
    correct_index: int
    explanation: str

# --- Example: Quadrant 2D Rotation question ---
def quadrant_rotation_question(seed=0, difficulty=Difficulty.EASY.value, variable_mode=True, use_llm=True) -> Question:
    RNG.seed(seed)
    canvas_size = 320
    img = new_canvas(canvas_size,canvas_size)
    d = ImageDraw.Draw(img)
    draw_square(d, (12,12), img.width-24)
    midx, midy = img.width//2, img.height//2
    d.line((12, midy, img.width-12, midy), fill=(0,0,0), width=STYLE["grid"])
    d.line((midx, 12, midx, img.height-12), fill=(0,0,0), width=STYLE["grid"])
    # Place glyphs
    glyphs = ["plus","diamond","circle","L"]
    locs = [(midx-55, midy-55), (midx+55, midy-55), (midx-55, midy+55), (midx+55, midy+55)]
    for glyph, (cx, cy) in zip(glyphs, locs):
        if glyph == "plus": draw_plus(d, (cx, cy), 40)
        elif glyph == "diamond": draw_diamond(d, (cx, cy), 40)
        elif glyph == "circle": draw_circle(d, (cx, cy), 18)
        elif glyph == "L": draw_L(d, (cx, cy), 32)
    base = _finalize_for_display(img)
    angle = RNG.choice([90,180,270])
    correct_img = base.rotate(-angle, expand=True, fillcolor=(255,255,255))
    distract = [base.rotate(-a, expand=True, fillcolor=(255,255,255)) for a in [90,180,270] if a != angle]
    distract.append(ImageOps.mirror(base))
    options = [correct_img] + distract
    RNG.shuffle(options)
    stem = hstack(base, draw_banner_arrow(f"[translate:عند تدويرها {angle}° يصبح]"), faint_hint_box(text="[translate:؟]"))
    correct_index = options.index(correct_img)
    sys = "[translate:أنت مساعد تعليمي بالعربية.]"
    title = ollama_chat(sys, "[translate:أيُّ بديل يحقق التطابق بعد تدوير اللوحة؟]", model="qwen2.5:3b", enabled=use_llm)
    expl = ollama_chat(sys, "[translate:الإجابة الصحيحة لها نفس ترتيب الرموز بعد التدوير حول المركز.]", model="qwen2.5:3b",enabled=use_llm)
    return Question(title, _finalize_for_display(stem), [_finalize_for_display(o) for o in options], correct_index, expl)

# --- More question builders: Paper folding, cube rotation, assembly, symbol grid, analogy, completion ---
# For brevity, define them similar to above. See your original code for each type.

# ---- Example of Modern question: Analogy/Completion ----
def analogy_question(seed=0, use_llm=True) -> Question:
    RNG.seed(seed)
    # Simple pattern analogy: matching patterns, e.g. pattern transformation (modern style)
    base = new_canvas(200,200)
    d = ImageDraw.Draw(base)
    draw_square(d, (30,30), 140)
    draw_plus(d, (100,100), 90)
    pattern = base.copy()
    transformed = ImageOps.mirror(pattern)
    options = [transformed, base.rotate(90), ImageOps.flip(base), base.rotate(180)]
    RNG.shuffle(options)
    correct_index = options.index(transformed)
    stem = hstack(pattern, draw_banner_arrow("[translate:إذا عكسنا النمط يصبح]"), faint_hint_box(text="[translate:؟]"))
    sys = "[translate:أنت مساعد تعليمي بالعربية.]"
    title = ollama_chat(sys, "[translate:اختر الصورة التي تحقق العلاقة أو التناظر في الأعلى.]", model="qwen2.5:3b", enabled=use_llm)
    expl = ollama_chat(sys, "[translate:الإجابة الصحيحة تظهر انعكاس النمط تمامًا.]", model="qwen2.5:3b", enabled=use_llm)
    return Question(title, _finalize_for_display(stem), [_finalize_for_display(o) for o in options], correct_index, expl)

# --- Add more: cube rotation, paper fold, shape assembly, fill in logic from your original code ---

# Factory function
def build_by_type(kind: str, seed: int) -> Question:
    if kind == PuzzleType.PAPER_FOLD.value:
        return paper_fold_question(seed=seed, difficulty=Difficulty.EASY.value, use_llm=st.session_state.get("use_llm", True))
    if kind == PuzzleType.QUADRANT_ROTATE.value:
        return quadrant_rotation_question(seed=seed, difficulty=Difficulty.EASY.value, variable_mode=True, use_llm=st.session_state.get("use_llm", True))
    if kind == PuzzleType.ANALOGY.value:
        return analogy_question(seed=seed, use_llm=st.session_state.get("use_llm", True))
    # ...add all others...
    raise ValueError("نوع سؤال غير معروف")

# ========== Ready for PART 4: Main Streamlit UI and export logic ==========
# ========== PART 4: Main Streamlit UI and Export Logic ==========

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH))
st.sidebar.markdown(_t("settings"))

n_q = st.sidebar.number_input(_t("num_questions"), 1, 24, 8)
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
with col1:
    gen = st.button(_t("generate"), use_container_width=True)
with col2:
    st.caption(_t("instructions"))

st.title(_t("title"))

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
            q = build_by_type(kind, seed=qseed)
            st.markdown(f"#### {_t('question')} {idx}: {q.title}")
            st.image(q.stem_image, caption=f"{_t('alt_image_question')} {idx}", use_container_width=True)
            cols = st.columns(4, gap="small")
            for i, (c, col) in enumerate(zip(q.options, cols)):
                col.image(c, caption=f"{_t('alt_image_option')} {AR_LETTERS[i]}", use_container_width=True)
                col.markdown(
                    f"<div style='text-align:center;font-weight:bold;font-size:20px'>{AR_LETTERS[i]}</div>",
                    unsafe_allow_html=True,
                )
            # Export as ZIP file
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
    st.download_button(
        label=_t("download"),
        data=zip_buf.getvalue(),
        file_name="arabic_visual_iq_questions.zip",
        mime="application/zip",
        use_container_width=True,
    )
else:
    st.info(_t("start_generation"))

def _t(key: str) -> str:
    lang = "ar"
    return LANGS.get(lang, {}).get(key, key)

# ========== REMAINING: Drawing, question generation, and main Streamlit blocks will follow. ==========
# ========== Please reply "continue" to receive the next segment including all glyph drawing, shape and cube logic, and full modern question builders. ==========
