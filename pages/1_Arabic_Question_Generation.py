import streamlit as st
import json
import uuid
import re
import random
import os
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(page_title="توليد الأسئلة العربية", page_icon="🤖", layout="wide")

# Arabic RTL styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700&display=swap');

.main .block-container {
    direction: rtl;
    text-align: right;
    font-family: 'Noto Sans Arabic', 'Cairo', sans-serif;
}

.question-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 20px 0;
    border-right: 5px solid #1e3a8a;
}

.correct-choice {
    background: #dcfce7;
    padding: 12px;
    border-radius: 8px;
    border-right: 3px solid #16a34a;
    margin: 5px 0;
    font-weight: 500;
}

.choice-option {
    background: #f1f5f9;
    padding: 12px;
    border-radius: 8px;
    border-right: 3px solid #64748b;
    margin: 5px 0;
}

.bulk-progress {
    background: #f0f9ff;
    padding: 20px;
    border-radius: 12px;
    border-right: 3px solid #0891b2;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

# Database helpers
DB_PATH = Path(__file__).parents[1] / "demo_db_arabic.json"

def load_db():
    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        empty_db = {"questions": [], "exams": [], "submissions": []}
        save_db(empty_db)
        return empty_db

def save_db(data):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Initialize session state
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []
if 'bulk_generation' not in st.session_state:
    st.session_state.bulk_generation = False

# OpenAI integration
try:
    import openai
    openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not openai.api_key or openai.api_key == "your_openai_api_key_here":
        openai = None
except ImportError:
    openai = None


def call_openai(prompt, max_tokens=200):
    """Call OpenAI API if available"""
    if not openai:
        return ""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"OpenAI API Error: {e}")
        return ""

def randomize_choices(correct_choice, distractors):
    """Randomize the order of choices and return shuffled choices with correct answer info"""
    all_choices = [correct_choice] + distractors[:3]  # Ensure we have exactly 4
    
    # Shuffle the choices
    random.shuffle(all_choices)
    
    # Find the new position of the correct answer
    correct_index = all_choices.index(correct_choice)
    
    return all_choices, correct_choice

def get_random_arabic_word(difficulty="متوسط"):
    """Get a random Arabic word based on difficulty level"""
    
    word_banks = {
        "سهل": [
            "الحب", "البيت", "الماء", "النور", "السلام", "الأمل", "الخير", "الصدق",
            "الكرم", "الشجاعة", "الحكمة", "الصبر", "العدل", "الرحمة", "الود", "الفرح"
        ],
        "متوسط": [
            "التعاون", "الإبداع", "العدالة", "المسؤولية", "الإنجاز", "التميز", 
            "الاحترام", "التسامح", "المثابرة", "الأمانة", "الوفاء", "التفاني",
            "الإخلاص", "المبادرة", "التطوير", "الإصلاح", "التقدم", "النهضة"
        ],
        "صعب": [
            "الديمقراطية", "الاستراتيجية", "الفلسفة", "المنهجية", "الأيديولوجية",
            "البيروقراطية", "الدبلوماسية", "التكنولوجيا", "الاستدامة", "الشفافية",
            "المصداقية", "الموضوعية", "التحليل", "الاستنباط", "التجريد", "التركيب"
        ]
    }
    
    words = word_banks.get(difficulty, word_banks["متوسط"])
    return random.choice(words)

# Arabic question generators
def generate_word_meaning_question(main_word=None, difficulty="متوسط"):
    """Generate Arabic word meaning MCQ with randomized choices"""
    
    # Auto-select word if not provided
    if not main_word:
        main_word = get_random_arabic_word(difficulty)
    
    prompt = f"""
أنشئ سؤال اختيار من متعدد لاختبار معنى الكلمة العربية: "{main_word}"

المطلوب:
- إجابة صحيحة واحدة (مرادف دقيق)
- 3 إجابات خاطئة من نفس المجال الدلالي لكن ليست مرادفات
- تجنب الكلمات من نفس الجذر
- مستوى الصعوبة: {difficulty}

أعط الإجابة الصحيحة أولاً، ثم 3 إجابات خاطئة، كل كلمة في سطر منفصل:
"""
    
    gpt_response = call_openai(prompt)
    
    if gpt_response:
        lines = [line.strip() for line in gpt_response.split('\n') if line.strip()]
        if len(lines) >= 4:
            correct_choice = lines[0]
            distractors = lines[1:4]
        else:
            words = [w.strip() for w in gpt_response.split(',')][:4]
            if len(words) >= 4:
                correct_choice = words[0]
                distractors = words[1:4]
            else:
                # Fallback
                correct_choice = f"مرادف_{main_word}"
                distractors = ["خيار1", "خيار2", "خيار3"]
    else:
        # Enhanced fallback dictionary
        fallbacks = {
            "التعاون": ["التضامن", "التنافس", "الفردية", "الصراع"],
            "الإبداع": ["الابتكار", "التقليد", "المحاكاة", "النسخ"],
            "العدالة": ["الإنصاف", "الظلم", "التحيز", "المحاباة"],
            "الصبر": ["التحمل", "العجلة", "الغضب", "الاندفاع"],
            "الحكمة": ["الرشد", "الجهل", "الغباء", "الطيش"],
            "الشجاعة": ["البسالة", "الجبن", "الخوف", "التردد"],
            "الكرم": ["السخاء", "البخل", "الشح", "الطمع"],
            "الصدق": ["الأمانة", "الكذب", "النفاق", "الخداع"],
            "الحب": ["المودة", "الكراهية", "البغض", "النفور"],
            "السلام": ["الأمان", "الحرب", "النزاع", "الصراع"],
            "الأمل": ["التفاؤل", "اليأس", "القنوط", "الإحباط"],
            "النور": ["الضياء", "الظلام", "العتمة", "الكآبة"],
            "الرحمة": ["الشفقة", "القسوة", "الغلظة", "الجفاء"],
            "الود": ["المحبة", "الجفاء", "القطيعة", "النفور"],
            "المثابرة": ["الاستمرار", "التوقف", "الانقطاع", "التراجع"],
            "الأمانة": ["الثقة", "الخيانة", "الغدر", "المكر"],
            "الوفاء": ["الإخلاص", "الغدر", "الخيانة", "الجحود"],
            "المبادرة": ["الاستباق", "التأخير", "التواني", "التكاسل"],
            "التطوير": ["التحسين", "الإهمال", "التدهور", "التراجع"],
            "التقدم": ["النمو", "التأخر", "الانتكاس", "الجمود"]
        }
        
        if main_word in fallbacks:
            correct_choice = fallbacks[main_word][0]
            distractors = fallbacks[main_word][1:4]
        else:
            correct_choice = f"مرادف_{main_word}"
            distractors = ["خيار1", "خيار2", "خيار3"]
    
    # Ensure we have enough distractors
    while len(distractors) < 3:
        distractors.append(f"خيار_{len(distractors)+1}")
    
    # Randomize choices
    shuffled_choices, correct_answer = randomize_choices(correct_choice, distractors)
    
    return f"ما معنى كلمة '{main_word}'؟", shuffled_choices, correct_answer, main_word

def generate_quantitative_comparison_question(difficulty="متوسط"):
    """Generate quantitative comparison question"""
    
    if difficulty == "سهل":
        a, b = random.randint(1, 10), random.randint(1, 10)
        expr_a = f"{a} + {b}"
        result_a = a + b
        c = random.randint(1, 15)
        expr_b = str(c)
        result_b = c
    else:
        a, b = random.randint(5, 15), random.randint(3, 12)
        expr_a = f"{a} × {b}"
        result_a = a * b
        c, d = random.randint(2, 8), random.randint(15, 25)
        expr_b = f"{c}² + {d}"
        result_b = c**2 + d
    
    # Determine relationship
    if result_a > result_b:
        correct = "العمود الأول أكبر"
    elif result_a < result_b:
        correct = "العمود الثاني أكبر"
    else:
        correct = "القيمتان متساويتان"
    
    # Convert to Arabic numerals
    def to_arabic_numerals(text):
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        english_digits = "0123456789"
        for eng, ar in zip(english_digits, arabic_digits):
            text = text.replace(eng, ar)
        return text
    
    expr_a = to_arabic_numerals(expr_a)
    expr_b = to_arabic_numerals(expr_b)
    
    question = f"قارن بين:\nالعمود الأول: {expr_a}\nالعمود الثاني: {expr_b}"
    
    distractors = [
        "العمود الأول أكبر" if correct != "العمود الأول أكبر" else "العمود الثاني أكبر",
        "العمود الثاني أكبر" if correct != "العمود الثاني أكبر" else "القيمتان متساويتان",
        "لا يمكن تحديد العلاقة"
    ]
    
    # Randomize choices
    shuffled_choices, correct_answer = randomize_choices(correct, distractors)
    
    return question, shuffled_choices, correct_answer

def generate_number_sequence_question(difficulty="متوسط"):
    """Generate number sequence question"""
    
    def to_arabic_numerals(num):
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        english_digits = "0123456789"
        text = str(num)
        for eng, ar in zip(english_digits, arabic_digits):
            text = text.replace(eng, ar)
        return text
    
    if difficulty == "سهل":
        start = random.randint(2, 10)
        diff = random.randint(2, 5)
        sequence = [start + i * diff for i in range(4)]
        next_num = sequence[-1] + diff
    else:
        start = random.randint(2, 4)
        ratio = random.randint(2, 3)
        sequence = [start * (ratio ** i) for i in range(4)]
        next_num = sequence[-1] * ratio
    
    sequence_ar = [to_arabic_numerals(x) for x in sequence]
    correct_next = to_arabic_numerals(next_num)
    
    distractors = [
        to_arabic_numerals(next_num + random.randint(1, 10)),
        to_arabic_numerals(next_num - random.randint(1, 8)),
        to_arabic_numerals(next_num + random.randint(15, 25))
    ]
    
    question = f"ما العدد التالي في المتتالية: {', '.join(sequence_ar)}، ؟"
    
    # Randomize choices
    shuffled_choices, correct_answer = randomize_choices(correct_next, distractors)
    
    return question, shuffled_choices, correct_answer

# Enhanced question generators with randomization
question_generators = {
    "معاني الكلمات": lambda difficulty="متوسط": generate_word_meaning_question(None, difficulty),
    "معنى الكلمة حسب السياق": lambda: randomize_choices("تحسن وتجدد", ["تهدم", "تتجاهل", "تؤخر"]),
    "إكمال الجمل": lambda: randomize_choices("التعاون", ["التنافس", "التجاهل", "التأجيل"]),
    "المقارنات الكمية": lambda difficulty="متوسط": generate_quantitative_comparison_question(difficulty),
    "سلاسل الأعداد": lambda difficulty="متوسط": generate_number_sequence_question(difficulty),
    "التشفير": lambda: randomize_choices("٢٨", ["٣٢", "٤٠", "٢٥"]),
    "الاستدلال العددي": lambda: randomize_choices("٣٠%", ["٢٥%", "٣٥%", "٢٠%"]),
    "الاستدلال المجرد": lambda: randomize_choices("دائرة", ["مستطيل", "معين", "خماسي"]),
    "الاستدلال الميكانيكي": lambda: randomize_choices("يميناً", ["يساراً", "لا تدور", "متغيرة"]),
    "القدرة المكانية": lambda: randomize_choices("١٩", ["٢١", "١٧", "٢٣"]),
    "تسلسل الأرقام والحروف عكسي": lambda: randomize_choices("٧، ب، ٣، أ", ["ب، ٧، أ، ٣", "٧، أ، ب، ٣", "٣، ب، ٧، أ"]),
    "مواقع الأرقام": lambda: randomize_choices("٦", ["٥", "٨", "٩"])
}

# Main UI
st.title("🤖 مولد الأسئلة العربية المتقدم")

st.markdown("""
<div style="background: #ecfdf5; padding: 20px; border-radius: 12px; border-right: 4px solid #10b981; margin: 20px 0;">
<strong>🎯 المرحلة الأولى:</strong> توليد وتطوير المحتوى التعليمي باستخدام الذكاء الاصطناعي المتقدم مع ضمانات الجودة
</div>
""", unsafe_allow_html=True)

# OpenAI status
if openai and openai.api_key:
    st.success("🟢 OpenAI API متصل ويعمل")
else:
    st.warning("🟡 OpenAI API غير متصل - سيتم استخدام البيانات التجريبية")

# Generation interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("⚙️ إعدادات التوليد")
    
    # Generation mode selection
    generation_mode = st.radio(
        "نمط التوليد:",
        ["سؤال واحد", "توليد متعدد"],
        horizontal=True
    )
    
    category = st.selectbox(
        "نوع السؤال",
        options=list(question_generators.keys()),
        help="اختر نوع السؤال المراد توليده"
    )
    
    difficulty = st.selectbox(
        "مستوى الصعوبة",
        ["سهل", "متوسط", "صعب"],
        index=1
    )
    
    # Number of questions input for bulk generation
    if generation_mode == "توليد متعدد":
        num_questions = st.number_input(
            "عدد الأسئلة المطلوبة:",
            min_value=1,
            max_value=50,
            value=5,
            help="حدد عدد الأسئلة التي تريد توليدها دفعة واحدة"
        )
    else:
        num_questions = 1
    
    # Topic input for non-word-meaning questions
    if category != "معاني الكلمات":
        topic = st.text_input(
            "الموضوع (اختياري)",
            value="العمل الجماعي",
            help="موضوع أو مفهوم محدد"
        )
    
    # Information about word meaning questions
    if category == "معاني الكلمات":
        st.info("""
        🤖 **التوليد الآلي للكلمات**
        
        سيقوم النظام تلقائياً باختيار كلمات مناسبة حسب مستوى الصعوبة:
        - **سهل**: كلمات أساسية ومألوفة
        - **متوسط**: مفردات متنوعة ومهنية  
        - **صعب**: مصطلحات متقدمة وأكاديمية
        """)

with col2:
    st.subheader("📊 إحصائيات سريعة")
    
    db = load_db()
    pending_count = len([q for q in db["questions"] if q.get("status") == "pending"])
    approved_count = len([q for q in db["questions"] if q.get("status") == "approved"])
    
    st.metric("في قائمة الانتظار", pending_count)
    st.metric("معتمد", approved_count)
    
    if st.session_state.generated_questions:
        st.metric("أسئلة جاهزة للحفظ", len(st.session_state.generated_questions))

# Generate button
button_text = f"🚀 توليد {'الأسئلة' if generation_mode == 'توليد متعدد' else 'السؤال'}"
if st.button(button_text, type="primary", use_container_width=True):
    
    # Clear previous questions
    st.session_state.generated_questions = []
    
    # Set up progress tracking for bulk generation
    if num_questions > 1:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    with st.spinner("🔄 جاري التوليد..."):
        try:
            success_count = 0
            
            for i in range(num_questions):
                # Update progress for bulk generation
                if num_questions > 1:
                    progress = (i + 1) / num_questions
                    progress_bar.progress(progress)
                    status_text.text(f"توليد السؤال {i + 1} من {num_questions}...")
                
                # Generate question based on category
                if category == "معاني الكلمات":
                    question, choices, correct_answer, selected_word = generate_word_meaning_question(None, difficulty)
                    topic_for_storage = selected_word
                    
                elif category in ["المقارنات الكمية", "سلاسل الأعداد"]:
                    if category == "المقارنات الكمية":
                        question, choices, correct_answer = generate_quantitative_comparison_question(difficulty)
                    else:
                        question, choices, correct_answer = generate_number_sequence_question(difficulty)
                    topic_for_storage = topic if 'topic' in locals() else "عام"
                
                else:
                    generator = question_generators[category]
                    result = generator()
                    
                    if len(result) == 2:  # Simple generator returns (choices, correct)
                        choices, correct_answer = result
                        # Create appropriate question text based on category
                        question_templates = {
                            "معنى الكلمة حسب السياق": "في الجملة: 'الشركة تطور منتجات جديدة' - ما معنى 'تطور'؟",
                            "إكمال الجمل": "لضمان نجاح المشروع، يجب على الفريق _____ بكفاءة",
                            "التشفير": "إذا كان كتاب = ٣٥، فما قيمة قلم؟",
                            "الاستدلال العددي": "زادت أرباح شركة من ٥٠٠ ريال إلى ٦٥٠ ريال. كم نسبة الزيادة؟",
                            "الاستدلال المجرد": "ما الشكل التالي: دائرة، مربع، مثلث، ؟",
                            "الاستدلال الميكانيكي": "إذا دارت العجلة أ يميناً، كيف تدور العجلة ج؟",
                            "القدرة المكانية": "كم مكعباً لإكمال مكعب ٣×٣×٣ إذا نقص ٨؟",
                            "تسلسل الأرقام والحروف عكسي": "عكس: أ، ٣، ب، ٧؟",
                            "مواقع الأرقام": "في شبكة ٣×٣، ما الرقم في الموضع (٢،٣)؟"
                        }
                        question = question_templates.get(category, f"سؤال {category}")
                    else:  # Advanced generator returns (question, choices, correct)
                        question, choices, correct_answer = result
                    
                    topic_for_storage = topic if 'topic' in locals() else "عام"
                
                # Ensure we have 4 choices
                while len(choices) < 4:
                    choices.append(f"خيار إضافي {len(choices)}")
                
                # Store generated question
                question_data = {
                    "question": question,
                    "choices": choices,
                    "correct_answer": correct_answer,
                    "category": category,
                    "difficulty": difficulty,
                    "topic": topic_for_storage,
                    "question_number": i + 1
                }
                
                st.session_state.generated_questions.append(question_data)
                success_count += 1
            
            # Clear progress indicators
            if num_questions > 1:
                progress_bar.empty()
                status_text.empty()
            
            st.success(f"✅ تم توليد {success_count} سؤال بنجاح!")
            
        except Exception as e:
            st.error(f"خطأ في التوليد: {e}")

# Display generated questions
if st.session_state.generated_questions:
    st.subheader(f"📝 الأسئلة المولدة ({len(st.session_state.generated_questions)})")
    
    # Display each generated question
    for i, question_data in enumerate(st.session_state.generated_questions):
        with st.expander(f"السؤال {i+1}: {question_data['category']}", expanded=(i == 0)):
            st.markdown(f"""
            <div class="question-card">
                <h4>📝 السؤال {question_data.get('question_number', i+1)}:</h4>
                <p style="font-size: 18px; font-weight: 500; margin: 15px 0;">{question_data["question"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 الخيارات:")
            
            for j, choice in enumerate(question_data["choices"]):
                if choice == question_data["correct_answer"]:
                    st.markdown(f"""
                    <div class="correct-choice">
                        ✅ <strong>{choice}</strong> (الإجابة الصحيحة)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="choice-option">
                        {choice}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Question metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**الفئة:** {question_data['category']}")
            with col2:
                st.caption(f"**الصعوبة:** {question_data['difficulty']}")
            with col3:
                if question_data['category'] == "معاني الكلمات":
                    st.caption(f"**الكلمة المختارة:** {question_data['topic']}")
                else:
                    st.caption(f"**الموضوع:** {question_data['topic']}")
    
    # Bulk save button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button(f"💾 حفظ جميع الأسئلة في قاعدة البيانات ({len(st.session_state.generated_questions)})", 
                     type="primary", use_container_width=True):
            
            db = load_db()
            saved_count = 0
            
            with st.spinner("🔄 جاري حفظ الأسئلة..."):
                for question_data in st.session_state.generated_questions:
                    new_question = {
                        "id": str(uuid.uuid4())[:8],
                        "category": question_data["category"],
                        "text": question_data["question"],
                        "options": question_data["choices"],
                        "answer": question_data["correct_answer"],
                        "status": "pending",
                        "difficulty": question_data["difficulty"],
                        "topic": question_data["topic"],
                        "generated_at": datetime.now().isoformat(),
                        "generator": "ai_advanced_bulk"
                    }
                    
                    db["questions"].append(new_question)
                    saved_count += 1
                
                save_db(db)
            
            # Clear the generated questions from session
            st.session_state.generated_questions = []
            
            st.success(f"🎉 تم حفظ {saved_count} سؤال وإرسالها لقائمة مراجعة الخبراء!")
            st.experimental_rerun()

# Recent questions queue
st.subheader("📋 الأسئلة المولدة حديثاً")

db = load_db()
recent_questions = [q for q in db["questions"]][-10:]  # Last 10

if recent_questions:
    for q in reversed(recent_questions):
        status_colors = {
            "pending": "#fef3c7", 
            "approved": "#dcfce7", 
            "rejected": "#fee2e2"
        }
        status_text = {
            "pending": "في الانتظار ⏳", 
            "approved": "معتمد ✅", 
            "rejected": "مرفوض ❌"
        }
        
        with st.expander(f"{q['category']} - {q['id']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**السؤال:** {q['text']}")
                st.markdown(f"**الخيارات:** {', '.join(q['options'][:2])}...")
                st.markdown(f"**الإجابة الصحيحة:** {q['answer']}")
                if q['category'] == "معاني الكلمات":
                    st.markdown(f"**الكلمة:** {q.get('topic', 'غير محدد')}")
            
            with col2:
                st.markdown(f"""
                <div style="background: {status_colors.get(q['status'], '#f3f4f6')}; 
                           padding: 10px; border-radius: 8px; text-align: center;">
                    <strong>{status_text.get(q['status'], 'غير معروف')}</strong>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("💡 لم يتم توليد أي أسئلة بعد. ابدأ بتوليد سؤالك الأول!")

# Instructions
with st.expander("📖 دليل الاستخدام المتقدم"):
    st.markdown("""
    ### 🎯 ميزات التوليد المحدثة:
    
    **🤖 الاختيار الآلي للكلمات:**
    - لا حاجة لإدخال كلمات يدوياً لأسئلة المعاني
    - النظام يختار كلمات مناسبة حسب مستوى الصعوبة
    - تنوع واسع في الكلمات المختارة
    - مراعاة المستوى التعليمي للطلاب
    
    **📊 بنك الكلمات حسب الصعوبة:**
    - **سهل**: كلمات أساسية (الحب، البيت، الماء...)
    - **متوسط**: مفردات مهنية (التعاون، الإبداع، العدالة...)
    - **صعب**: مصطلحات متقدمة (الديمقراطية، الاستراتيجية...)
    
    **🔄 ترتيب عشوائي للخيارات:**
    - الإجابة الصحيحة لا تظهر دائماً في المركز الأول
    - يقلل من إمكانية التخمين أو الحفظ الآلي
    - يحسن من جودة التقييم ودقة النتائج
    
    **📊 التوليد المتعدد:**
    - إمكانية توليد عدة أسئلة دفعة واحدة (حتى 50 سؤال)
    - شريط تقدم لمتابعة عملية التوليد
    - حفظ جماعي لجميع الأسئلة المولدة
    - مثالي لإنشاء بنك أسئلة كبير بسرعة
    
    ### 🎯 أنواع الأسئلة المدعومة (12 نوع):
    
    **🔤 اللغوية:**
    - **معاني الكلمات**: اختبار فهم معنى المفردات العربية (اختيار آلي للكلمات)
    - **معنى الكلمة حسب السياق**: فهم المعنى في سياق محدد
    - **إكمال الجمل**: إكمال الجمل بالخيار المناسب
    
    **🔢 الرياضية:**
    - **المقارنات الكمية**: مقارنة القيم الرياضية
    - **سلاسل الأعداد**: إيجاد النمط في سلاسل الأرقام
    - **التشفير**: حل رموز الأحرف والأرقام
    - **الاستدلال العددي**: حل المسائل الرياضية التطبيقية
    
    **🧠 الذهنية:**
    - **الاستدلال المجرد**: التعرف على الأنماط البصرية
    - **الاستدلال الميكانيكي**: فهم العمليات الميكانيكية
    - **القدرة المكانية**: التصور المكاني والهندسي
    - **تسلسل الأرقام والحروف عكسي**: عكس الترتيبات
    - **مواقع الأرقام**: تحديد المواضع في الشبكات
    
    ### ✨ نصائح للاستخدام الأمثل:
    - 🤖 **للأسئلة المفردة**: مناسب للاختبار والتجريب السريع
    - 📚 **للتوليد المتعدد**: مثالي لبناء بنك أسئلة شامل
    - 🎯 **أسئلة معاني الكلمات**: تلقائية بالكامل ومتنوعة
    - 🔄 **الخيارات العشوائية**: تحسن من جودة التقييم
    - 💾 **الحفظ الجماعي**: يوفر الوقت والجهد
    """)
