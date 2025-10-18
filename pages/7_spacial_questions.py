import streamlit as st
import random

st.title("مولد أسئلة ذكاء مكاني (بدون مكتبات إضافية)")

def get_circle_grid(pos_list, size=2):
    grid = [["⬜" for _ in range(size)] for _ in range(size)]
    for x, y in pos_list:
        grid[y][x] = "⚪"
    return "\n".join("".join(row) for row in grid)

def get_triangle_grid(pos_type, size=2):
    # Just flip positions for demonstration; can make logic more complex as needed
    if pos_type == "original":
        grid = [
            ["🔺", "⚪"],
            ["⚪", "🔺"]
        ]
    elif pos_type == "mirror":
        grid = [
            ["⚪", "🔺"],
            ["🔺", "⚪"]
        ]
    return "\n".join("".join(row) for row in grid)

def question_circle():
    st.write("أي صورة تظهر بعد فتح الورقة كما هو موضح؟")
    folded = [(1,0)] # Top right
    st.write("الصورة المطوية:")
    st.code(get_circle_grid(folded))
    
    options = [
        [(1,0), (1,1)],   # two right
        [(0,0), (1,0)],   # top row
        [(1,0), (0,1)],   # diagonal (correct)
        [(0,1), (1,1)],   # two bottom
    ]
    correct = 2
    cols = st.columns(4)
    btns = []
    for i, opt in enumerate(options):
        with cols[i]:
            st.code(get_circle_grid(opt))
            btns.append(st.button(f"اختر {chr(65+i)}", key=f"btn-circ-{i}"))
    return btns, correct

def question_triangle():
    st.write("أي صورة تظهر بعد فتح الورقة كما هو موضح؟")
    st.write("الصورة المطوية:")
    st.code(get_triangle_grid("original"))
    options = [
        "original",
        "mirror",   # correct
        "original",
        "original"
    ]
    correct = 1
    cols = st.columns(4)
    btns = []
    for i, opt in enumerate(options):
        with cols[i]:
            st.code(get_triangle_grid(opt))
            btns.append(st.button(f"اختر {chr(65+i)}", key=f"btn-tri-{i}"))
    return btns, correct

qtype = random.choice(["circle", "triangle"])
if qtype == "circle":
    btns, correct = question_circle()
else:
    btns, correct = question_triangle()

for idx, pressed in enumerate(btns):
    if pressed:
        if idx == correct:
            st.success("إجابة صحيحة!")
        else:
            st.error("إجابة خاطئة. حاول مرة أخرى!")
