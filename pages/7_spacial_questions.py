import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import random
import io

def draw_folded_question(shape_type="circle"):
    img = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0,0,100,100], outline="black", width=3)
    draw.rectangle([100,0,200,100], outline="black", width=3)
    
    if shape_type == "circle":
        draw.ellipse([135, 35, 165, 65], outline="black", width=3)  # circle in top right
    elif shape_type == "triangle":
        draw.polygon([130,60,170,60,150,30], outline="black", width=3)  # triangle in top right

    # Draw arrow
    draw.arc([90, 20, 170, 80], start=0, end=180, fill="black", width=2)
    draw.polygon([170,50,160,45,160,55], fill="black")

    return img

def draw_answer_option(shape_type="circle", config=0):
    img = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0,0,100,100], outline="black", width=3)
    if shape_type == "circle":
        # Four configurations
        positions = [
            [(70, 30), (30,70)],        # Correct (diagonal)
            [(70, 30), (70,70)],        # right side
            [(30,70), (70,70)],         # bottom row
            [(30, 30), (70, 30)]        # top row
        ]
        for pos in positions[config]:
            draw.ellipse([pos[0]-15,pos[1]-15,pos[0]+15,pos[1]+15], outline="black", width=3)
    elif shape_type == "triangle":
        positions = [
            [(30,70,70,70,50,30),(80,50,60,80,100,80)], # original & mirrored
            [(70,70,30,70,50,100),(20,50,60,20,100,20)],
            [(30,30,70,30,50,70),(60,100,90,60,100,30)],
            [(20,60,60,20,80,80),(90,10,100,50,80,100)],
        ]
        pts = positions[config]
        draw.polygon(pts[0], outline="black", width=3)
        draw.polygon(pts[1], outline="black", width=3)
        # Also add some circles for style
        draw.ellipse([40, 50, 60, 70], outline="black", width=2)
        draw.ellipse([70, 60, 90, 80], outline="black", width=2)
    return img

st.title("أسئلة ذكاء مكاني بنمط الصور الأصلية")

question_type = st.selectbox("اختر نوع السؤال", ["دائرة", "مثلث"])
shape_type = "circle" if question_type == "دائرة" else "triangle"
st.write("الصورة بعد الطي (قبل الفتح):")
st.image(draw_folded_question(shape_type))

st.write("اختر الإجابة الصحيحة من الخيارات التالية:")

answer_configs = range(4)
cols = st.columns(4)
selected = None
for idx in answer_configs:
    image_buf = draw_answer_option(shape_type, idx)
    with cols[idx]:
        st.image(image_buf)
        if st.button(f"اختر {chr(65+idx)}"):
            selected = idx

correct_idx = 0 if shape_type == "circle" else 0  # Adjust logic to match your keys
if selected is not None:
    if selected == correct_idx:
        st.success("إجابة صحيحة!")
    else:
        st.error("خطأ! جرب مرة أخرى.")
