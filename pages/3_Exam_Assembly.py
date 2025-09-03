import streamlit as st
import json
import uuid
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(page_title="تجميع الاختبارات", page_icon="📋", layout="wide")

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

.exam-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 20px 0;
    border-right: 5px solid #3b82f6;
}

.question-item {
    background: #f8fafc;
    padding: 15px;
    border-radius: 8px;
    margin: 8px 0;
    border-right: 3px solid #e2e8f0;
}

.question-selected {
    background: #e0f2fe;
    border-right-color: #0891b2;
}

.create-btn {
    background: linear-gradient(45deg, #1e3a8a, #3b82f6) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
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
        return {"questions": [], "exams": [], "submissions": []}

def save_db(data):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Initialize session state
if 'selected_questions' not in st.session_state:
    st.session_state.selected_questions = []
if 'exam_preview' not in st.session_state:
    st.session_state.exam_preview = None
if 'exam_config_data' not in st.session_state:
    st.session_state.exam_config_data = {}
if 'auto_build_done' not in st.session_state:
    st.session_state.auto_build_done = False

# Main UI
st.title("📋 مجمع الاختبارات المتقدم")

st.markdown("""
<div style="background: #fef3c7; padding: 25px; border-radius: 15px; border-right: 4px solid #f59e0b; margin: 25px 0;">
<h3 style="color: #92400e; margin-top: 0;">🎯 المرحلة الأولى: بناء الاختبارات</h3>
<p style="font-size: 16px; color: #b45309;">
قم بإنشاء اختبارات مخصصة من الأسئلة المعتمدة مع إمكانيات متقدمة للتحكم في المحتوى والصعوبة والتوقيت
</p>
</div>
""", unsafe_allow_html=True)

# Load data
db = load_db()
approved_questions = [q for q in db.get('questions', []) if q.get('status') == 'approved']
existing_exams = db.get('exams', [])

# Statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("الأسئلة المعتمدة", len(approved_questions))
with col2:
    st.metric("الاختبارات الموجودة", len(existing_exams))
with col3:
    st.metric("أسئلة مختارة", len(st.session_state.selected_questions))
with col4:
    categories = len(set(q['category'] for q in approved_questions)) if approved_questions else 0
    st.metric("فئات متاحة", categories)

if not approved_questions:
    st.warning("""
    ⚠️ لا توجد أسئلة معتمدة لإنشاء اختبار
    
    **لإنشاء اختبار:**
    1. انتقل لصفحة "توليد الأسئلة" وأنشئ أسئلة جديدة
    2. انتقل لصفحة "مراجعة الخبراء" واعتمد الأسئلة
    3. عُد هنا لإنشاء اختبارات من الأسئلة المعتمدة
    """)
    st.stop()

# Exam builder interface
st.subheader("🔧 منشئ الاختبارات")

# Exam configuration
st.markdown("### ⚙️ إعدادات الاختبار")

# Use regular inputs instead of form to avoid session state conflicts
col1, col2 = st.columns(2)

with col1:
    exam_title = st.text_input(
        "عنوان الاختبار *",
        value=st.session_state.exam_config_data.get('title', 'اختبار القدرات المعرفية الأساسي'),
        key="exam_title_input",
        help="عنوان واضح ومحدد للاختبار"
    )
    
    exam_description = st.text_area(
        "وصف الاختبار",
        value=st.session_state.exam_config_data.get('description', 'اختبار شامل لتقييم القدرات المعرفية الأساسية للمتقدمين'),
        height=100,
        key="exam_description_input",
        help="وصف مختصر لمحتوى وهدف الاختبار"
    )
    
    exam_difficulty = st.selectbox(
        "مستوى الصعوبة العام",
        ["سهل", "متوسط", "صعب", "مختلط"],
        index=1,
        key="exam_difficulty_input"
    )

with col2:
    time_limit = st.number_input(
        "المدة الزمنية (بالدقائق)",
        min_value=5,
        max_value=180,
        value=st.session_state.exam_config_data.get('time_limit', 30),
        key="time_limit_input",
        help="الوقت المحدد لإنهاء الاختبار"
    )
    
    max_questions = st.number_input(
        "العدد الأقصى للأسئلة",
        min_value=1,
        max_value=50,
        value=st.session_state.exam_config_data.get('max_questions', 10),
        key="max_questions_input",
        help="العدد المطلوب من الأسئلة في الاختبار"
    )
    
    shuffle_questions = st.checkbox(
        "خلط ترتيب الأسئلة", 
        value=st.session_state.exam_config_data.get('shuffle_questions', True),
        key="shuffle_questions_input"
    )
    shuffle_options = st.checkbox(
        "خلط ترتيب الخيارات", 
        value=st.session_state.exam_config_data.get('shuffle_options', False),
        key="shuffle_options_input"
    )

# Auto-build options
st.markdown("### 🤖 خيارات البناء الآلي")
col1, col2 = st.columns(2)

with col1:
    auto_build = st.checkbox(
        "بناء آلي متوازن", 
        value=False, 
        key="auto_build_checkbox",
        help="اختيار تلقائي للأسئلة مع توزيع متوازن"
    )
    if auto_build:
        selected_categories = st.multiselect(
            "الفئات المطلوبة:",
            options=list(set(q['category'] for q in approved_questions)),
            default=list(set(q['category'] for q in approved_questions))[:3],
            key="selected_categories_input"
        )

with col2:
    manual_selection = st.checkbox(
        "اختيار يدوي", 
        value=True, 
        key="manual_selection_checkbox",
        help="اختيار الأسئلة بشكل يدوي"
    )
    if manual_selection:
        filter_category = st.selectbox(
            "فلترة حسب الفئة:",
            ["جميع الفئات"] + list(set(q['category'] for q in approved_questions)),
            key="filter_category_input"
        )

# Build exam button
if st.button("🏗️ بناء الاختبار", type="primary", use_container_width=True):
    if not exam_title.strip():
        st.error("❌ يرجى إدخال عنوان للاختبار")
    else:
        # Store exam configuration in session state
        st.session_state.exam_config_data = {
            'title': exam_title,
            'description': exam_description,
            'difficulty': exam_difficulty,
            'time_limit': time_limit,
            'max_questions': max_questions,
            'shuffle_questions': shuffle_questions,
            'shuffle_options': shuffle_options
        }
        
        if auto_build and 'selected_categories' in locals():
            # Auto-build logic
            selected_questions = []
            questions_per_category = max(1, max_questions // len(selected_categories))
            
            for category in selected_categories:
                category_questions = [q for q in approved_questions if q['category'] == category]
                selected_questions.extend(category_questions[:questions_per_category])
            
            # Fill remaining slots if needed
            if len(selected_questions) < max_questions:
                remaining = max_questions - len(selected_questions)
                other_questions = [q for q in approved_questions if q not in selected_questions]
                selected_questions.extend(other_questions[:remaining])
            
            st.session_state.selected_questions = selected_questions[:max_questions]
            st.session_state.auto_build_done = True
            st.success(f"✅ تم بناء الاختبار تلقائياً مع {len(st.session_state.selected_questions)} سؤال")
        
        elif manual_selection:
            st.session_state.auto_build_done = False
            st.success("✅ تم حفظ إعدادات الاختبار - اختر الأسئلة أدناه")

# Question selection interface
if manual_selection and not st.session_state.auto_build_done:
    st.subheader("📝 اختيار الأسئلة")
    
    # Filter questions
    display_questions = approved_questions
    if 'filter_category' in locals() and filter_category != "جميع الفئات":
        display_questions = [q for q in approved_questions if q['category'] == filter_category]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**الأسئلة المتاحة ({len(display_questions)}):**")
        
        # Quick selection buttons
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            if st.button("✅ اختيار الكل", help="اختيار جميع الأسئلة المعروضة", key="select_all"):
                max_limit = st.session_state.exam_config_data.get('max_questions', 10)
                st.session_state.selected_questions = display_questions[:max_limit]
                st.experimental_rerun()
        
        with col1b:
            if st.button("❌ إلغاء الاختيار", help="إلغاء اختيار جميع الأسئلة", key="deselect_all"):
                st.session_state.selected_questions = []
                st.experimental_rerun()
        
        with col1c:
            if st.button("🎲 اختيار عشوائي", help="اختيار عشوائي للأسئلة", key="random_select"):
                import random
                max_limit = st.session_state.exam_config_data.get('max_questions', 10)
                random_questions = random.sample(display_questions, min(len(display_questions), max_limit))
                st.session_state.selected_questions = random_questions
                st.experimental_rerun()
        
        # Question list with checkboxes
        st.markdown("---")
        for i, question in enumerate(display_questions[:20]):  # Show first 20
            is_selected = question in st.session_state.selected_questions
            
            col_check, col_content = st.columns([0.1, 0.9])
            
            with col_check:
                if st.checkbox(
                    "",
                    value=is_selected,
                    key=f"q_select_{question['id']}_{i}",
                    label_visibility="collapsed"
                ):
                    if not is_selected:
                        max_limit = st.session_state.exam_config_data.get('max_questions', 10)
                        if len(st.session_state.selected_questions) < max_limit:
                            st.session_state.selected_questions.append(question)
                        else:
                            st.warning(f"تم الوصول للحد الأقصى ({max_limit} أسئلة)")
                            st.experimental_rerun()
                else:
                    if is_selected:
                        st.session_state.selected_questions.remove(question)
            
            with col_content:
                # Question display
                css_class = "question-selected" if is_selected else "question-item"
                st.markdown(f"""
                <div class="{css_class}">
                    <strong>{question['category']}</strong> - {question['id']}<br>
                    <span style="color: #4b5563;">{question['text'][:100]}{'...' if len(question['text']) > 100 else ''}</span><br>
                    <small style="color: #6b7280;">صعوبة: {question.get('difficulty', 'متوسط')} | خيارات: {len(question.get('options', []))}</small>
                </div>
                """, unsafe_allow_html=True)
        
        if len(display_questions) > 20:
            st.info(f"عرض 20 من أصل {len(display_questions)} سؤال. استخدم الفلاتر لتضييق النتائج.")
    
    with col2:
        st.markdown("**الأسئلة المختارة:**")
        
        if st.session_state.selected_questions:
            for i, q in enumerate(st.session_state.selected_questions, 1):
                st.markdown(f"""
                <div style="background: #e0f2fe; padding: 10px; border-radius: 6px; margin: 5px 0;">
                    <small><strong>{i}.</strong> {q['category']}<br>
                    {q['text'][:50]}...</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Category distribution
            categories = {}
            for q in st.session_state.selected_questions:
                cat = q['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            st.markdown("**توزيع الفئات:**")
            for cat, count in categories.items():
                st.markdown(f"• {cat}: {count}")
        else:
            st.info("لم يتم اختيار أي أسئلة بعد")

# Exam preview and creation
if st.session_state.selected_questions and st.session_state.exam_config_data:
    st.subheader("👁️ معاينة الاختبار")
    
    config = st.session_state.exam_config_data
    
    # Exam info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("عدد الأسئلة", len(st.session_state.selected_questions))
    with col2:
        st.metric("المدة المقدرة", f"{config.get('time_limit', 30)} دقيقة")
    with col3:
        categories_count = len(set(q['category'] for q in st.session_state.selected_questions))
        st.metric("عدد الفئات", categories_count)
    
    # Detailed preview
    with st.expander("🔍 معاينة تفصيلية للأسئلة", expanded=False):
        for i, q in enumerate(st.session_state.selected_questions, 1):
            st.markdown(f"""
            **السؤال {i}:** {q['text']}
            
            **الخيارات:**
            {chr(10).join([f"• {opt}" for opt in q.get('options', [])])}
            
            **الإجابة الصحيحة:** {q.get('answer', 'غير محدد')}
            
            **الفئة:** {q['category']} | **الصعوبة:** {q.get('difficulty', 'متوسط')}
            
            ---
            """)
    
    # Create exam button
    if st.button("🎯 إنشاء الاختبار النهائي", type="primary", use_container_width=True):
        if not config.get('title'):
            st.error("❌ خطأ في إعدادات الاختبار")
        else:
            # Create new exam
            new_exam = {
                "id": str(uuid.uuid4())[:8],
                "title": config['title'],
                "description": config.get('description', ''),
                "question_ids": [q['id'] for q in st.session_state.selected_questions],
                "time_limit": config.get('time_limit', 30),
                "difficulty": config.get('difficulty', 'متوسط'),
                "created_at": datetime.now().isoformat(),
                "settings": {
                    "shuffle_questions": config.get('shuffle_questions', True),
                    "shuffle_options": config.get('shuffle_options', False),
                    "show_results": True,
                    "allow_review": True
                },
                "status": "active"
            }
            
            # Save to database
            db['exams'].append(new_exam)
            save_db(db)
            
            # Clear session
            st.session_state.selected_questions = []
            st.session_state.exam_config_data = {}
            st.session_state.auto_build_done = False
            
            st.success(f"🎉 تم إنشاء الاختبار بنجاح! معرف الاختبار: {new_exam['id']}")
            st.experimental_rerun()

elif st.session_state.selected_questions and not st.session_state.exam_config_data:
    st.warning("⚠️ يرجى إكمال إعدادات الاختبار أولاً عن طريق الضغط على 'بناء الاختبار'")

# Existing exams management
st.subheader("📚 إدارة الاختبارات الموجودة")

if existing_exams:
    for exam in existing_exams:
        with st.expander(f"📋 {exam['title']} - {exam['id']}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**الوصف:** {exam.get('description', 'لا يوجد وصف')}")
                st.markdown(f"**عدد الأسئلة:** {len(exam.get('question_ids', []))}")
                st.markdown(f"**المدة:** {exam.get('time_limit', 'غير محدد')} دقيقة")
                st.markdown(f"**الصعوبة:** {exam.get('difficulty', 'غير محدد')}")
                
                # Question categories breakdown
                exam_questions = [q for q in approved_questions if q['id'] in exam.get('question_ids', [])]
                if exam_questions:
                    categories = {}
                    for q in exam_questions:
                        cat = q['category']
                        categories[cat] = categories.get(cat, 0) + 1
                    
                    st.markdown("**توزيع الفئات:**")
                    for cat, count in categories.items():
                        st.caption(f"• {cat}: {count} سؤال")
            
            with col2:
                st.markdown("**الإعدادات:**")
                settings = exam.get('settings', {})
                st.caption(f"خلط الأسئلة: {'نعم' if settings.get('shuffle_questions') else 'لا'}")
                st.caption(f"خلط الخيارات: {'نعم' if settings.get('shuffle_options') else 'لا'}")
                st.caption(f"عرض النتائج: {'نعم' if settings.get('show_results') else 'لا'}")
                
                created_at = exam.get('created_at', '')
                if created_at:
                    try:
                        created_date = datetime.fromisoformat(created_at).strftime('%Y-%m-%d %H:%M')
                        st.caption(f"تاريخ الإنشاء: {created_date}")
                    except:
                        st.caption("تاريخ الإنشاء: غير معروف")
            
            with col3:
                st.markdown("**الإجراءات:**")
                
                if st.button(f"👁️ معاينة", key=f"preview_{exam['id']}"):
                    st.info("معاينة الاختبار متاحة في صفحة تقديم الاختبار")
                
                if st.button(f"📝 تعديل", key=f"edit_{exam['id']}"):
                    st.info("تم تحديد الاختبار للتعديل")
                
                if st.button(f"🗑️ حذف", key=f"delete_{exam['id']}"):
                    # Remove exam from database
                    db['exams'] = [e for e in db['exams'] if e['id'] != exam['id']]
                    save_db(db)
                    st.success("تم حذف الاختبار")
                    st.experimental_rerun()
else:
    st.info("💡 لا توجد اختبارات محفوظة بعد. قم بإنشاء اختبارك الأول!")

# Instructions
with st.expander("📖 دليل إنشاء الاختبارات المتقدم"):
    st.markdown("""
    ### 🎯 خطوات إنشاء اختبار احترافي:
    
    **1. إعداد الاختبار:**
    - اختر عنواناً واضحاً ومحدداً
    - اكتب وصفاً يوضح هدف الاختبار
    - حدد المدة الزمنية المناسبة
    - اختر مستوى الصعوبة المطلوب
    
    **2. اختيار الأسئلة:**
    - **البناء الآلي**: للحصول على توزيع متوازن تلقائياً
    - **الاختيار اليدوي**: لتحكم كامل في محتوى الاختبار
    - استخدم الفلاتر لتسهيل الاختيار
    - راجع توزيع الفئات للتأكد من التوازن
    
    **3. المعاينة والتأكيد:**
    - راجع الأسئلة المختارة
    - تأكد من صحة الإعدادات
    - اطلع على التوزيع النهائي
    
    **4. الإعدادات المتقدمة:**
    - **خلط الأسئلة**: يقلل من الغش
    - **خلط الخيارات**: يزيد من التحدي
    - **إظهار النتائج**: للمراجعة الفورية
    
    ### 💡 نصائح للاختبارات عالية الجودة:
    - نوّع في أنواع الأسئلة (3-5 فئات مختلفة)
    - اجعل 70% من الأسئلة متوسطة الصعوبة
    - احرص على وجود أسئلة سهلة في البداية
    - اختبر الاختبار بنفسك قبل النشر
    """)
