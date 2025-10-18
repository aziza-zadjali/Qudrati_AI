# -*- coding: utf-8 -*-
"""
Arabic Visual IQ Generator - Full Production Code
Generates spatial/visual IQ questions with modern enhancements
"""

import io
import math
import random
import zipfile
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
    QUADRANT_ROTATE = "تدوير رباعي"
    DIAGONAL_ROTATE = "تدوير قطري"
    CUBE_ROTATE = "تدوير مكعّبات ثلاثي"
    SHAPE_ASSEMBLY = "تركيب شكل"

# Translation dictionary
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

# Constants
RNG = random.Random()
AR_LETTERS = ["أ", "ب", "ج", "د"]
CANVAS_SCALE = 1.7
STYLE = {
    "grid": 6, "square": 6, "glyph": 6, "circle": 5,
    "arrow": 6, "diamond": 6, "half": 4, "iso_edge": 3, "dash": 4,
}

# =================== FONT & ARABIC HELPERS ===================

def get_ar_font(px: int) -> Optional[ImageFont.FreeTypeFont]:
    for name in ["NotoNaskhArabic-Regular.ttf", "NotoKufiArabic-Regular.ttf", 
                 "Amiri-Regular.ttf", "DejaVuSans.ttf"]:
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

# =================== DRAWING UTILITIES ===================

def new_canvas(w, h, bg=(255,255,255)):
    W, H = int(w * CANVAS_SCALE), int(h * CANVAS_SCALE)
    return Image.new("RGB", (W, H), bg)

def bytes_from_img(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _finalize_for_display(img: Image.Image, target_size: Optional[Tuple[int, int]] = None):
    if target_size:
        return img.resize((int(target_size[0]), int(target_size[1])), Image.LANCZOS)
    if CANVAS_SCALE == 1.0:
        return img
    return img.resize((int(img.width / CANVAS_SCALE), int(img.height / CANVAS_SCALE)), Image.LANCZOS)

def dashed_line(draw: ImageDraw.ImageDraw, start, end, dash_len=10, gap_len=8, 
                fill=(0,0,0), width: Optional[int]=None):
    if width is None:
        width = STYLE["dash"]
    x1, y1 = start
    x2, y2 = end
    total_len = math.hypot(x2-x1, y2-y1)
    if total_len == 0:
        return
    dx = (x2-x1)/total_len
    dy = (y2-y1)/total_len
    dist = 0.0
    while dist < total_len:
        s, e = dist, min(dist+dash_len, total_len)
        xs, ys = x1+dx*s, y1+dy*s
        xe, ye = x1+dx*e, y1+dy*e
        draw.line((xs,ys,xe,ye), fill=fill, width=width)
        dist += dash_len + gap_len

def poly_with_width(draw: ImageDraw.ImageDraw, pts, fill=None, outline=(0,0,0), width=1):
    if fill is not None:
        draw.polygon(pts, fill=fill)
    if outline is not None and width > 0:
        for i in range(len(pts)):
            a, b = pts[i], pts[(i+1)%len(pts)]
            draw.line([a, b], fill=outline, width=width)

def draw_square(draw: ImageDraw.ImageDraw, xy, size, outline=(0,0,0), width: Optional[int]=None):
    if width is None:
        width = STYLE["square"]
    x, y = xy
    draw.rectangle([x, y, x+size, y+size], outline=outline, width=width)

def draw_circle(draw: ImageDraw.ImageDraw, center, r, fill=None, outline=(0,0,0), width: Optional[int]=None):
    if width is None:
        width = STYLE["circle"]
    cx, cy = center
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill, outline=outline, width=width)

def draw_plus(draw: ImageDraw.ImageDraw, center, size, width: Optional[int]=None, fill=(0,0,0)):
    if width is None:
        width = STYLE["glyph"]
    cx, cy = center
    s = size // 2
    draw.line((cx-s,cy,cx+s,cy), fill=fill, width=width)
    draw.line((cx,cy-s,cx,cy+s), fill=fill, width=width)

def draw_diamond(draw: ImageDraw.ImageDraw, center, size, fill=None, outline=(0,0,0), width: Optional[int]=None):
    if width is None:
        width = STYLE["diamond"]
    cx, cy = center
    s = size // 2
    pts = [(cx, cy-s), (cx+s, cy), (cx, cy+s), (cx-s, cy)]
    poly_with_width(draw, pts, fill=fill, outline=outline, width=width)

def draw_L(draw: ImageDraw.ImageDraw, center, size, outline=(0,0,0), width: Optional[int]=None):
    if width is None:
        width = STYLE["glyph"]
    cx, cy = center
    s = size
    draw.line((cx-s//2, cy-s//2, cx-s//2, cy+s//2), fill=outline, width=width)
    draw.line((cx-s//2, cy+s//2, cx+s//2, cy+s//2), fill=outline, width=width)

def draw_triangle(draw: ImageDraw.ImageDraw, pts, fill=None, outline=(0,0,0), width: Optional[int]=None):
    if width is None:
        width = STYLE["glyph"]
    poly_with_width(draw, pts, fill=fill, outline=outline, width=width)

def draw_star(draw: ImageDraw.ImageDraw, center, r, points=5, outline=(0,0,0), width: Optional[int]=None):
    if width is None:
        width = STYLE["glyph"]
    cx, cy = center
    pts = []
    for i in range(points*2):
        angle = math.pi * i / points
        rr = r if i%2==0 else r*0.45
        pts.append((cx+rr*math.cos(angle-math.pi/2), cy+rr*math.sin(angle-math.pi/2)))
    poly_with_width(draw, pts, fill=None, outline=outline, width=width)

def draw_half_circle(draw: ImageDraw.ImageDraw, center, r, orientation: str = "up", 
                     outline=(0,0,0), width: Optional[int]=None):
    if width is None:
        width = STYLE["half"]
    cx, cy = center
    bbox = [cx-r, cy-r, cx+r, cy+r]
    start_map = {"up":0, "right":90, "down":180, "left":270}
    end_map = {"up":180, "right":270, "down":360, "left":90}
    draw.pieslice(bbox, start=start_map[orientation], end=end_map[orientation], 
                  fill=None, outline=outline, width=width)

def hstack(*imgs: Image.Image, pad: int = 16, bg=(255,255,255)) -> Image.Image:
    H = max(im.height for im in imgs)
    W = sum(im.width for im in imgs) + int(pad*CANVAS_SCALE)*(len(imgs)-1)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for im in imgs:
        y = (H-im.height)//2
        out.paste(im, (x, y))
        x += im.width + int(pad*CANVAS_SCALE)
    return out

def draw_banner_arrow(text: str, direction: str = "right") -> Image.Image:
    W, H = int(360 * CANVAS_SCALE), int(120 * CANVAS_SCALE)
    img = Image.new("RGBA", (W, H), (255,255,255,0))
    d = ImageDraw.Draw(img)
    body_h = int(60 * CANVAS_SCALE)
    y0 = (H-body_h)//2
    if direction == "left":
        d.rectangle([(int(40*CANVAS_SCALE),y0),(W-int(20*CANVAS_SCALE),y0+body_h)], 
                    outline=(0,0,0), width=STYLE["arrow"])
        tri = [(int(40*CANVAS_SCALE),H//2),
               (int(75*CANVAS_SCALE),y0-int(6*CANVAS_SCALE)),
               (int(75*CANVAS_SCALE),y0+body_h+int(6*CANVAS_SCALE))]
        d.polygon(tri, fill=(255,255,255), outline=(0,0,0))
    else:
        d.rectangle([(int(20*CANVAS_SCALE),y0),(W-int(40*CANVAS_SCALE),y0+body_h)], 
                    outline=(0,0,0), width=STYLE["arrow"])
        tri = [(W-int(40*CANVAS_SCALE),H//2),
               (W-int(75*CANVAS_SCALE),y0-int(6*CANVAS_SCALE)),
               (W-int(75*CANVAS_SCALE),y0+body_h+int(6*CANVAS_SCALE))]
        d.polygon(tri, fill=(255,255,255), outline=(0,0,0))
    label = shape_ar(text)
    f = get_ar_font(int(28*CANVAS_SCALE))
    d.text((W//2,H//2), label, fill=(0,0,0), anchor="mm", font=f)
    return img.convert("RGB")

def faint_hint_box(side: int = 220, text: str = "؟") -> Image.Image:
    img = new_canvas(side, side)
    d = ImageDraw.Draw(img)
    draw_square(d, (10,10), img.width-20, outline=(150,150,150), width=max(2,STYLE["square"]-2))
    f = get_ar_font(int(72*CANVAS_SCALE))
    d.text((img.width//2, img.height//2), text, fill=(170,170,170), anchor="mm", font=f)
    return img

# =================== GLYPH LOGIC ===================

def glyph_pool(variable_mode: bool, difficulty: str) -> List[str]:
    base = ["plus", "diamond", "circle", "triangle"]
    extra = ["star", "half_up", "half_right", "square_small", "L"]
    if not variable_mode:
        return base
    if difficulty == Difficulty.EASY.value:
        return base + ["star", "square_small"]
    if difficulty == Difficulty.MEDIUM.value:
        return base + ["star", "half_up", "square_small", "L"]
    return base + extra

def draw_glyph(draw: ImageDraw.ImageDraw, glyph: str, center: Tuple[int,int], size: int):
    if glyph == "plus":
        draw_plus(draw, center, size)
    elif glyph == "diamond":
        draw_diamond(draw, center, size-int(4*CANVAS_SCALE))
    elif glyph == "circle":
        draw_circle(draw, center, (size//2)-int(2*CANVAS_SCALE))
    elif glyph == "triangle":
        cx, cy = center
        half = (size//2)
        pts = [(cx-half, cy+half), (cx+half, cy+half), (cx, cy-half)]
        draw_triangle(draw, pts)
    elif glyph == "star":
        draw_star(draw, center, (size//2)-int(2*CANVAS_SCALE))
    elif glyph == "half_up":
        draw_half_circle(draw, center, (size//2)-int(2*CANVAS_SCALE), orientation="up")
    elif glyph == "half_right":
        draw_half_circle(draw, center, (size//2)-int(2*CANVAS_SCALE), orientation="right")
    elif glyph == "square_small":
        cx, cy = center
        s = size - int(8*CANVAS_SCALE)
        draw_square(draw, (cx-s//2, cy-s//2), s)
    elif glyph == "L":
        draw_L(draw, center, size)

# =================== BOARD LAYOUTS ===================

def board_cross(canvas_size=300, seed=0, variable_mode: bool=True, difficulty: str="سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas_size, canvas_size)
    d = ImageDraw.Draw(img)
    draw_square(d, (10,10), img.width-20)
    midx, midy = img.width//2, img.height//2
    d.line((10,midy,img.width-10,midy), fill=(0,0,0), width=STYLE["grid"])
    d.line((midx,10,midx,img.height-10), fill=(0,0,0), width=STYLE["grid"])
    pool = glyph_pool(variable_mode, difficulty)
    RNG.shuffle(pool)
    chosen = pool[:4]
    jitter_map = {"سهل": 6, "متوسط": 10, "صعب": 16}
    jitter = int(jitter_map.get(difficulty, 6) * CANVAS_SCALE)
    offset = int((70 if difficulty != "صعب" else 64) * CANVAS_SCALE)
    locs = [(midx-offset,midy-offset), (midx+offset,midy-offset),
            (midx-offset,midy+offset), (midx+offset,midy+offset)]
    for glyph, (cx,cy) in zip(chosen, locs):
        draw_glyph(d, glyph, (cx+RNG.randint(-jitter,jitter), cy+RNG.randint(-jitter,jitter)), 
                   int(48*CANVAS_SCALE))
    return img

def board_diag(canvas_size=300, seed=0, variable_mode: bool=True, difficulty: str="سهل") -> Image.Image:
    RNG.seed(seed)
    img = new_canvas(canvas_size, canvas_size)
    d = ImageDraw.Draw(img)
    draw_square(d, (10,10), img.width-20)
    d.line((10,10,img.width-10,img.height-10), fill=(0,0,0), width=STYLE["grid"])
    d.line((10,img.height-10,img.width-10,10), fill=(0,0,0), width=STYLE["grid"])
    pool = glyph_pool(variable_mode, difficulty)
    RNG.shuffle(pool)
    chosen = pool[:3]
    midx, midy = img.width//2, img.height//2
    base = int((76 if difficulty != "صعب" else 70) * CANVAS_SCALE)
    jitter_map = {"سهل": 6, "متوسط": 10, "صعب": 14}
    jitter = int(jitter_map.get(difficulty, 6) * CANVAS_SCALE)
    slots = [(midx-base,midy-base), (midx+base,midy-base), (midx-base,midy+base)]
    for glyph, (cx,cy) in zip(chosen, slots):
        draw_glyph(d, glyph, (cx+RNG.randint(-jitter,jitter), cy+RNG.randint(-jitter,jitter)), 
                   int(48*CANVAS_SCALE))
    return img

def rotate_image(img: Image.Image, angle_deg: int, allow_mirror=False) -> Image.Image:
    out = img.rotate(angle_deg, expand=True, fillcolor=(255,255,255))
    if allow_mirror and RNG.random() < 0.25:
        out = ImageOps.mirror(out)
    return out

def compose_stem(reference: Image.Image, banner_text: str) -> Image.Image:
    ref = _finalize_for_display(reference)
    arrow = draw_banner_arrow(banner_text, direction="right")
    qbox = faint_hint_box(text="؟")
    return hstack(ref, arrow, qbox)

# =================== LLM HELPERS ===================

def ollama_chat(system: str, user: str, model: str, enabled: bool = True, max_tokens: int = 256) -> str:
    if not enabled:
        return user
    try:
        import requests
        r = requests.post("http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role":"system", "content":system},
                    {"role":"user", "content":user}
                ],
                "stream": False,
                "options": {"num_predict": max_tokens}
            }, timeout=8)
        if r.ok:
            out = r.json().get("message",{}).get("content","").strip()
            if out:
                return out
    except Exception:
        pass
    return user

# =================== QUESTION DATACLASS ===================

@dataclass
class Question:
    title: str
    stem_image: Image.Image
    options: List[Image.Image]
    correct_index: int
    explanation: str

# =================== QUESTION BUILDERS ===================

def quadrant_rotation_question(seed: int = 0, difficulty: str = "سهل", 
                               variable_mode: bool = True, use_llm: bool = True) -> Question:
    base = board_cross(canvas_size=300, seed=seed, variable_mode=variable_mode, difficulty=difficulty)
    angle = RNG.choice([90, 180, 270])
    stem = compose_stem(base, banner_text=f"عند تدويرها {angle}° يصبح")
    
    correct = rotate_image(base, -angle, allow_mirror=False)
    distract = [rotate_image(base, -a) for a in [90,180,270] if a != angle][:2]
    distract.append(ImageOps.mirror(base))
    
    options = [correct] + distract
    RNG.shuffle(options)
    correct_index = options.index(correct)
    
    stem = _finalize_for_display(stem)
    options = [_finalize_for_display(o) for o in options]
    
    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat(sys, "انظر إلى اللوحة على اليسار والسهم. أيُّ بديل يطابق نتيجة التدوير؟", 
                       model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    expl = ollama_chat(sys, "الصحيح يحافظ على ترتيب الرموز بعد تدويرها حول المركز دون انعكاس.", 
                      model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    return Question(title=title, stem_image=stem, options=options, 
                   correct_index=correct_index, explanation=expl)

def diagonal_rotation_question(seed: int = 0, variable_mode: bool = True, 
                               difficulty: str = "سهل", use_llm: bool = True) -> Question:
    base = board_diag(canvas_size=300, seed=seed, variable_mode=variable_mode, difficulty=difficulty)
    angle = RNG.choice([90, 180, 270])
    stem = compose_stem(base, banner_text=f"عند تدويرها {angle}° يصبح")
    
    correct = rotate_image(base, -angle)
    distract = [rotate_image(base, -a) for a in [90,180,270] if a != angle][:2]
    distract.append(ImageOps.mirror(base))
    
    options = [correct] + distract
    RNG.shuffle(options)
    correct_index = options.index(correct)
    
    stem = _finalize_for_display(stem)
    options = [_finalize_for_display(o) for o in options]
    
    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat(sys, "أيُّ بديل يطابق نتيجة التدوير الموضّحة؟", 
                       model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    expl = ollama_chat(sys, "راجع مواضع الرموز على القطرين بعد الدوران؛ لا يوجد انعكاس.", 
                      model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    return Question(title=title, stem_image=stem, options=options, 
                   correct_index=correct_index, explanation=expl)

def paper_fold_question(seed: int = 0, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    RNG.seed(seed)
    # Setup dimensions
    W, H = 160, 200      # Rectangle and grid size
    MARGIN = 16
    HOLE_RADIUS = 13
    # -- Step 1. Draw folded rectangle, fold line, and hole location --
    folded = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(folded)
    # Main rectangle
    d.rectangle([MARGIN, MARGIN, W - MARGIN, H - MARGIN], outline="black", width=4)
    # Fold line center (horizontal)
    yfold = (H + MARGIN) // 2
    # Dashed line
    dashlen, gap = 7, 7
    for x in range(MARGIN+7, W - MARGIN, dashlen + gap):
        d.line([(x, yfold), (min(x + dashlen, W - MARGIN), yfold)], fill="black", width=2)
    # Draw one hole on upper right panel (user can randomize location)
    hole_x = W - MARGIN - 23
    hole_y = yfold - 36
    d.ellipse((hole_x - HOLE_RADIUS, hole_y - HOLE_RADIUS, hole_x + HOLE_RADIUS, hole_y + HOLE_RADIUS), outline="black", width=3)
    # -- Step 2. Stem: rectangle + pro arrow outside (not on image) --
    stem_img = folded

    # -- Step 3. Answer options images; always 4, with various placements --
    def ans_img(holes: List[Tuple[int, int]]) -> Image.Image:
        ans = Image.new("RGB", (W, H), "white")
        d2 = ImageDraw.Draw(ans)
        d2.rectangle([MARGIN, MARGIN, W - MARGIN, H - MARGIN], outline="black", width=4)
        for (hx, hy) in holes:
            d2.ellipse((hx-HOLE_RADIUS, hy-HOLE_RADIUS, hx+HOLE_RADIUS, hy+HOLE_RADIUS), outline="black", width=3)
        return ans

    # Hole reflection logic after fold: symmetry over fold line (horizontal)
    hole2_x, hole2_y = hole_x, 2*yfold - hole_y
    correct = ans_img([(hole_x, hole_y), (hole2_x, hole2_y)])
    wrong1 = ans_img([(hole_x, hole_y)])                             # Only top hole
    wrong2 = ans_img([(hole2_x, hole2_y)])                           # Only bottom hole
    wrong3 = ans_img([(hole_x-30, hole_y), (hole2_x-30, hole2_y)])   # Shifted holes

    opts = [correct, wrong1, wrong2, wrong3]
    labels = list(range(len(opts)))
    RNG.shuffle(labels)
    final_opts = [opts[i] for i in labels]
    correct_index = labels.index(0) # 'correct' is always at position 0 in opts

    # -- Step 4. Professional answer panel images (with _finalize_for_display) --
    ans_imgs = [_finalize_for_display(o, (120, 150)) for o in final_opts]

    # -- Step 5. Stem: concat with big arrow (outside image), faint q mark
    arrow_img = draw_banner_arrow("إعادة فتح الورقة", direction="right")
    qmark_img = faint_hint_box(70, text="؟")
    stem = hstack(_finalize_for_display(stem_img, (120, 150)), arrow_img, qmark_img, pad=32)

    sys = "أنت مساعد تعليمي بالعربية."
    title = ollama_chat(sys, "انظر إلى المثال وحدد الإجابة حسب التعليمات. ما رمز البديل الذي يحتوي على صورة المطابقة بعد إعادة فتح الورقة؟",
                        model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)
    expl = ollama_chat(sys, "الصحيح يُظهر كل الثقوب بعد الطي (أصلي + منعكس للأسفل).", model=st.session_state.get("llm_model", "qwen2.5:3b"), enabled=use_llm)

    return Question(
        title=title,
        stem_image=_finalize_for_display(stem, (410, 160)),
        options=ans_imgs,
        correct_index=correct_index,
        explanation=expl
    )



# Placeholder for cube and assembly questions (add your full implementations here)
def cubes_rotation_question(seed: int = 0, difficulty: str = "سهل", use_llm: bool = True) -> Question:
    # Simplified placeholder - add your full 3D cube logic here
    return paper_fold_question(seed, difficulty, use_llm)

def shape_assembly_question(seed: int = 0, use_llm: bool = True) -> Question:
    # Simplified placeholder - add your full assembly logic here
    return paper_fold_question(seed, "سهل", use_llm)

# =================== FACTORY FUNCTION ===================

def build_by_type(kind: str, seed: int, difficulty: str, variable_mode: bool) -> Question:
    use_llm = st.session_state.get("use_llm", True)
    if kind == PuzzleType.PAPER_FOLD.value:
        return paper_fold_question(seed=seed, difficulty=difficulty, use_llm=use_llm)
    if kind == PuzzleType.QUADRANT_ROTATE.value:
        return quadrant_rotation_question(seed=seed, difficulty=difficulty, 
                                         variable_mode=variable_mode, use_llm=use_llm)
    if kind == PuzzleType.DIAGONAL_ROTATE.value:
        return diagonal_rotation_question(seed=seed, difficulty=difficulty, 
                                         variable_mode=variable_mode, use_llm=use_llm)
    if kind == PuzzleType.CUBE_ROTATE.value:
        return cubes_rotation_question(seed=seed, difficulty=difficulty, use_llm=use_llm)
    if kind == PuzzleType.SHAPE_ASSEMBLY.value:
        return shape_assembly_question(seed=seed, use_llm=use_llm)
    raise ValueError(f"Unknown puzzle type: {kind}")

# =================== STREAMLIT UI ===================

st.sidebar.markdown(f"### {_t('settings')}")

col_left, col_right = st.sidebar.columns(2)
with col_left:
    n_q = st.number_input(_t("num_questions"), 1, 24, 8)
with col_right:
    difficulty_str = st.selectbox(_t("difficulty"), [d.value for d in Difficulty])

seed_base = st.sidebar.number_input(_t("seed_label"), 0, 10_000_000, 12345, 
                                    help="لإعادة إنتاج نفس الأسئلة.")
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

col1, col2 = st.columns([1, 1])
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
            q = build_by_type(kind, seed=qseed, difficulty=difficulty_str, 
                            variable_mode=variable_symbols)

            st.markdown(f"#### {_t('question')} {idx}: {q.title}")
            st.image(q.stem_image, use_container_width=True, output_format="PNG", 
                    caption=f"{_t('alt_image_question')} {idx}")

            cols = st.columns(4, gap="small")
            for i, (c, col) in enumerate(zip(q.options, cols)):
                col.image(c, use_container_width=True, output_format="PNG", 
                         caption=f"{_t('alt_image_option')} {AR_LETTERS[i]}")
                col.markdown(
                    f"<div style='text-align:center;font-weight:bold;font-size:20px'>{AR_LETTERS[i]}</div>",
                    unsafe_allow_html=True,
                )

            # Export to ZIP
            zf.writestr(f"q{idx}_stem.png", bytes_from_img(q.stem_image))
            for i, c in enumerate(q.options, start=1):
                zf.writestr(f"q{idx}_opt_{i}.png", bytes_from_img(c))
            answers_csv.write(f"{idx},{kind},{AR_LETTERS[q.correct_index]},{qseed}\n")

            with st.expander(_t("show_solution")):
                st.markdown(f"**{_t('correct_answer')}** {AR_LETTERS[q.correct_index]}")
                st.write(q.explanation if st.session_state.get("use_llm", True) 
                        else "الصحيح الوحيد يوافق قاعدة السؤال (انعكاس/تدوير/تجميع).")

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


