import streamlit as st
import random
import matplotlib.pyplot as plt
import io

# Helper function to draw shapes
def draw_shape(question_type, answer=None):
    fig, ax = plt.subplots(figsize=(2,2))
    ax.axis("off")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    if question_type == "circle":
        # Positions for the circles
        positions = [(0.5, 1.5), (1.5, 0.5)]
        if answer:
            positions = answer
        for (x, y) in positions:
            circle = plt.Circle((x, y), 0.3, color='white', ec='black')
            ax.add_patch(circle)
        plt.gca().set_aspect('equal')
    elif question_type == "triangle":
        positions = [((0.4,1.7),(0.8,1.7),(0.6,1.4)), ((1.3,0.7),(1.7,0.7),(1.5,1.0))]
        if answer:
            positions = answer
        for pts in positions:
            triangle = plt.Polygon(pts, color='white', ec='black')
            ax.add_patch(triangle)
        # Draw some circles for answer options
        ax.add_patch(plt.Circle((0.9,0.7), 0.15, color='white', ec='black'))
        ax.add_patch(plt.Circle((1.5,1.4), 0.15, color='white', ec='black'))
        plt.gca().set_aspect('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_question():
    # Randomly select question type
    q_type = random.choice(["circle", "triangle"])
    question_text = ""
    answer_options = []
    correct_idx = 0

    if q_type == "circle":
        question_text = "أي صورة تظهر بعد فتح الورقة كما هو موضح؟"
        # Main positions
        main_positions = [(0.5, 1.5), (1.5, 0.5)]
        # Now generate 4 options: one correct (mirrored), three distractors
        options = [
            [(1.5, 1.5), (0.5, 0.5)],       # mirrored
            [(0.5, 1.5), (1.5, 0.5)],       # same as original
            [(0.5, 0.5), (1.5, 1.5)],       # both moved down
            [(1, 1.5), (1, 0.5)]            # both moved to middle x
        ]
        correct_idx = 0
        answer_options = options
    elif q_type == "triangle":
        question_text = "أي صورة تظهر بعد فتح الورقة كما هو موضح؟"
        main_positions = [((0.4,1.7),(0.8,1.7),(0.6,1.4)), ((1.3,0.7),(1.7,0.7),(1.5,1.0))]
        options = [
            [((0.4,1.7),(0.8,1.7),(0.6,1.4)), ((1.3,0.7),(1.7,0.7),(1.5,1.0))],         # same as original
            [((1.3,1.7),(1.7,1.7),(1.5,1.4)), ((0.4,0.7),(0.8,0.7),(0.6,1.0))],         # mirrored
            [((0.4,1.4),(0.8,1.4),(0.6,1.1)), ((1.3,1.0),(1.7,1.0),(1.5,1.3))],         # both moved down
            [((1.0,1.7),(1.4,1.7),(1.2,1.4)), ((0.7,0.7),(1.1,0.7),(0.9,1.0))],         # both moved right
        ]
        correct_idx = 1
        answer_options = options

    return question_text, q_type, main_positions, answer_options, correct_idx

st.title("مولد أسئلة الذكاء المكاني")
st.write("صفحة لتوليد أسئلة مشابهة للأمثلة المرفقة للذكاء المكاني.\nاختر الإجابة الصحيحة من بين الخيارات.")

question_text, q_type, main_positions, answer_options, correct_idx = generate_question()

st.header("السؤال")
st.write(question_text)

st.subheader("الصورة بعد الطي (قبل الفتح)")
image_buf = draw_shape(q_type, main_positions)
st.image(image_buf)

st.subheader("اختر الإجابة الصحيحة")

cols = st.columns(4)
selected = None

for i, option in enumerate(answer_options):
    with cols[i]:
        buf = draw_shape(q_type, option)
        st.image(buf)
        if st.button(f"اختيار الخيار {chr(65+i)}"):
            selected = i

if selected is not None:
    if selected == correct_idx:
        st.success("إجابة صحيحة!")
    else:
        st.error("إجابة خاطئة. حاول مرة أخرى.")

