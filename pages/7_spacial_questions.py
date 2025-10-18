import streamlit as st
from PIL import Image, ImageDraw
import math

def draw_folded_vertical_question():
    # Create blank image: (width, height)
    img = Image.new("RGB", (280, 140), "white")
    draw = ImageDraw.Draw(img)

    # Outer rectangle
    draw.rounded_rectangle([10, 20, 270, 120], radius=15, outline="black", width=4)

    # Vertical fold axis as a thick line
    draw.line([(140, 20), (140, 120)], fill="black", width=4)

    # Triangle on the right half
    draw.polygon([(200, 95), (230, 75), (220, 40)], outline="black", width=4)

    # Folding arrow (curved from right to left OVER vertical axis)
    # Draw a quarter ellipse to mimic your sample
    bbox = [110, 55, 170, 115]
    draw.arc(bbox, start=270, end=360, fill="black", width=3)
    # Arrow head (simple triangle)
    arrow_tip = (140, 85)
    left = (148, 88)
    right = (143, 95)
    draw.polygon([arrow_tip, left, right], fill="black")

    return img

# Streamlit display page
st.title("أسئلة ذكاء مكاني بنمط عينة المطابقة")

st.write("الصورة بعد الطي (قبل الفتح):")
st.image(draw_folded_vertical_question())

st.write("اختر الإجابة الصحيحة من الخيارات التالية:")

# ---- Answer option generation here: draw your four answer options ----
# Use the same style as for the question, but images should show *both sides*.
# You can expand on this by coding the four option images as needed.

# Example placeholder with blank rectangles:
cols = st.columns(4)
for i in range(4):
    opt = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(opt)
    draw.rectangle([0,0,99,99], outline="black", width=3)
    cols[i].image(opt)
    cols[i].button(f"اختر {chr(65+i)}", key=f"btn_{i}")

