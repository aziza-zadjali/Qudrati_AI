import streamlit as st
import json
import datetime
import time
import random
from pathlib import Path

# Page config
st.set_page_config(page_title="تقديم الاختبار", page_icon="📝", layout="wide")

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

.exam-header {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    color: white;
    padding: 30px;
    border-radius: 15px;
    margin: 25px 0;
    text-align: center;
}

.question-container {
    background: white;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin: 30px 0;
    border-right: 5px solid #3b82f6;
}

.question-text {
    font-size: 22px;
    font-weight: 500;
    line-height: 1.8;
    color: #1f2937;
    margin: 25px 0;
    padding: 25px;
    background: #f8fafc;
    border-radius: 12px;
    border-right: 3px solid #1e3a8a;
}

.timer-widget {
    position: fixed;
    top: 20px;
    left: 20px;
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white;
    padding: 15px 20px;
    border-radius: 50px;
    font-weight: bold;
    font-size: 18px;
    z-index: 1000;
    box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3);
}

.student-info {
    background: #f0f9ff;
    padding: 20px;
    border-radius: 12px;
    border-right: 4px solid #0891b2;
    margin: 20px 0;
}

.results-approved {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #10b981;
    padding: 30px;
    border-radius: 15px;
    margin: 30px 0;
    text-align: center;
}

.results-pending {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border: 2px solid #f59e0b;
    padding: 30px;
    border-radius: 15px;
    margin: 30px 0;
    text-align: center;
}

.results-needs-revision {
    background: linear-gradient(135deg, #fef2f2, #fecaca);
    border: 2px solid #ef4444;
    padding: 30px;
    border-radius: 15px;
    margin: 30px 0;
    text-align: center;
}

.score-card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 20px 0;
    text-align: center;
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
        return {"questions": [], "exams": [], "submissions": [], "sme_reviews": []}

def save_db(data):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def auto_correct_exam(exam_questions, student_answers):
    """Auto-correct the exam and prepare detailed results for SME review"""
    detailed_results = []
    correct_count = 0
    total_questions = len(exam_questions)
    
    for question in exam_questions:
        question_id = question['id']
        student_answer = student_answers.get(question_id, "لم يجب")
        correct_answer = question['answer']
        is_correct = student_answer == correct_answer
        
        if is_correct:
            correct_count += 1
        
        result_detail = {
            "question_id": question_id,
            "question_text": question['text'],
            "question_category": question.get('category', 'غير محدد'),
            "question_difficulty": question.get('difficulty', 'متوسط'),
            "options": question.get('options', []),
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "is_correct": is_correct,
            "points": 1 if is_correct else 0
        }
        detailed_results.append(result_detail)
    
    # Calculate scores and percentages
    percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0
    
    # Determine performance level
    if percentage >= 90:
        performance_level = "ممتاز"
        performance_color = "#22c55e"
    elif percentage >= 80:
        performance_level = "جيد جداً"
        performance_color = "#3b82f6"
    elif percentage >= 70:
        performance_level = "جيد"
        performance_color = "#f59e0b"
    elif percentage >= 60:
        performance_level = "مقبول"
        performance_color = "#f97316"
    else:
        performance_level = "يحتاج تحسين"
        performance_color = "#ef4444"
    
    return {
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "incorrect_answers": total_questions - correct_count,
        "percentage": percentage,
        "performance_level": performance_level,
        "performance_color": performance_color,
        "detailed_results": detailed_results
    }

def create_sme_review_record(submission, correction_results):
    """Create a record for SME review of the corrected exam"""
    sme_review = {
        "id": f"sme_{submission['id']}_{int(datetime.datetime.now().timestamp())}",
        "submission_id": submission['id'],
        "exam_id": submission['exam_id'],
        "exam_title": submission['exam_title'],
        "student_name": submission['student_name'],
        "student_id": submission['student_id'],
        "department": submission['department'],
        "auto_correction": correction_results,
        "sme_status": "pending_review",  # pending_review, approved, needs_revision
        "sme_reviewer": None,
        "sme_comments": "",
        "sme_reviewed_at": None,
        "created_at": datetime.datetime.now().isoformat(),
        "priority": "normal"  # normal, high, urgent
    }
    return sme_review

def get_student_results(student_id, student_name):
    """Get all results for a specific student"""
    db = load_db()
    submissions = db.get("submissions", [])
    sme_reviews = db.get("sme_reviews", [])
    
    # Find student's submissions
    student_submissions = [
        s for s in submissions 
        if s.get('student_id') == student_id and s.get('student_name') == student_name
    ]
    
    results = []
    for submission in student_submissions:
        # Find corresponding SME review
        sme_review = next(
            (r for r in sme_reviews if r.get('submission_id') == submission['id']), 
            None
        )
        
        result = {
            "submission": submission,
            "sme_review": sme_review,
            "status": submission.get('status', 'submitted'),
            "sme_approved": submission.get('sme_approved', False),
            "final_results_released": submission.get('final_results_released', False)
        }
        results.append(result)
    
    return results

# Initialize session state
if 'exam_started' not in st.session_state:
    st.session_state.exam_started = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'selected_exam' not in st.session_state:
    st.session_state.selected_exam = None
if 'exam_submitted' not in st.session_state:
    st.session_state.exam_submitted = False
if 'submission_data' not in st.session_state:
    st.session_state.submission_data = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'exams'  # exams or results

# Main UI
st.title("📝 منصة الاختبارات الإلكترونية")

st.markdown("""
<div class="exam-header">
    <h2>🎯 المرحلة الثانية: تقديم وإجراء الاختبارات</h2>
    <p style="font-size: 18px; margin: 10px 0;">
    واجهة محاكاة كاملة لتجربة الطالب في بيئة اختبار آمنة ومريحة
    </p>
</div>
""", unsafe_allow_html=True)

db = load_db()

# Student authentication simulation (would be inherited from MOL portal)
if 'student_name' not in st.session_state:
    st.markdown("""
    <div style="background: #ecfdf5; padding: 25px; border-radius: 15px; border-right: 4px solid #10b981; margin: 25px 0;">
        <h3 style="color: #065f46;">🔐 محاكاة تسجيل الدخول</h3>
        <p style="color: #047857;">في النظام الحقيقي، سيتم توريث بيانات الطالب تلقائياً من بوابة وزارة العمل الموحدة</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("اسم الطالب (للتجربة):", value="أحمد محمد الهنائي")
        student_id = st.text_input("رقم الهوية/الموظف:", value="12345678")
    
    with col2:
        department = st.selectbox("الإدارة/القسم:", ["الموارد البشرية", "التكنولوجيا", "المالية", "الخدمات"])
        position = st.selectbox("المسمى الوظيفي:", ["موظف", "أخصائي", "مشرف", "مدير"])
    
    if st.button("🚀 دخول للنظام", type="primary", use_container_width=True) and student_name and student_id:
        st.session_state.student_name = student_name
        st.session_state.student_id = student_id
        st.session_state.department = department
        st.session_state.position = position
        st.experimental_rerun()
    st.stop()

# Display student info
st.markdown(f"""
<div class="student-info">
    <strong>👤 الطالب:</strong> {st.session_state.student_name} | 
    <strong>🆔 الرقم:</strong> {st.session_state.student_id} | 
    <strong>🏢 القسم:</strong> {st.session_state.department} |
    <strong>💼 المسمى:</strong> {st.session_state.position}
</div>
""", unsafe_allow_html=True)

# View mode selector
st.subheader("🎯 اختر العملية")
view_mode = st.radio(
    "",
    ["تقديم اختبار جديد", "عرض النتائج والشهادات"],
    horizontal=True,
    key="view_mode_selector"
)

if view_mode == "عرض النتائج والشهادات":
    st.session_state.view_mode = 'results'
else:
    st.session_state.view_mode = 'exams'

# RESULTS VIEW MODE
if st.session_state.view_mode == 'results':
    st.subheader("📊 نتائجي وشهاداتي")
    
    # Get student results
    student_results = get_student_results(st.session_state.student_id, st.session_state.student_name)
    
    if not student_results:
        st.info("""
        📝 **لا توجد نتائج متاحة بعد**
        
        لم تقم بتقديم أي اختبارات بعد. قم بتقديم اختبار أولاً لرؤية النتائج هنا.
        """)
    else:
        st.markdown(f"**إجمالي الاختبارات المقدمة:** {len(student_results)}")
        
        for i, result in enumerate(student_results, 1):
            submission = result['submission']
            sme_review = result['sme_review']
            status = result['status']
            
            # Determine display status
            if status == 'approved' and result['final_results_released']:
                status_class = "results-approved"
                status_icon = "✅"
                status_text = "النتائج النهائية معتمدة"
                status_color = "#10b981"
            elif status == 'needs_revision':
                status_class = "results-needs-revision"
                status_icon = "🔄"
                status_text = "قيد المراجعة الإضافية"
                status_color = "#ef4444"
            else:
                status_class = "results-pending"
                status_icon = "⏳"
                status_text = "قيد المراجعة"
                status_color = "#f59e0b"
            
            with st.expander(f"{status_icon} اختبار {i}: {submission['exam_title']} - {status_text}", expanded=True):
                st.markdown(f"""
                <div class="{status_class}">
                    <h3 style="margin-top: 0; color: #1f2937;">📋 {submission['exam_title']}</h3>
                    <p style="color: #6b7280; margin: 10px 0;">
                        <strong>تاريخ التقديم:</strong> {datetime.datetime.fromisoformat(submission['timestamp']).strftime('%Y-%m-%d %H:%M')} | 
                        <strong>الوقت المستغرق:</strong> {submission['time_taken']} دقيقة
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if status == 'approved' and result['final_results_released']:
                    # Show final approved results
                    auto_correction = submission.get('auto_correction', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "النتيجة النهائية",
                            f"{auto_correction.get('correct_answers', 0)}/{auto_correction.get('total_questions', 0)}",
                            help="عدد الإجابات الصحيحة من إجمالي الأسئلة"
                        )
                    
                    with col2:
                        percentage = auto_correction.get('percentage', 0)
                        st.metric(
                            "النسبة المئوية",
                            f"{percentage:.1f}%",
                            help="النسبة المئوية للدرجات"
                        )
                    
                    with col3:
                        performance_level = auto_correction.get('performance_level', 'غير محدد')
                        st.metric(
                            "مستوى الأداء",
                            performance_level,
                            help="تقييم الأداء العام"
                        )
                    
                    with col4:
                        reviewer = submission.get('sme_reviewer', 'غير محدد')
                        st.metric(
                            "الخبير المراجع",
                            reviewer,
                            help="الخبير الذي راجع واعتمد النتائج"
                        )
                    
                    # SME Comments if available
                    if sme_review and sme_review.get('sme_comments'):
                        st.markdown("### 💬 ملاحظات الخبير المراجع:")
                        st.info(sme_review['sme_comments'])
                    
                                       # Detailed results breakdown
                    st.markdown("---")
                    show_details = st.checkbox("📊 عرض تفصيل النتائج حسب الفئات", key=f"details_{submission['id']}")
                    
                    if show_details:
                        detailed_results = auto_correction.get('detailed_results', [])
                        if detailed_results:
                            st.markdown("#### تفصيل الأداء حسب الفئات:")
                            
                            # Category-wise performance
                            category_results = {}
                            for detail in detailed_results:
                                cat = detail['question_category']
                                if cat not in category_results:
                                    category_results[cat] = {'total': 0, 'correct': 0}
                                
                                category_results[cat]['total'] += 1
                                if detail['is_correct']:
                                    category_results[cat]['correct'] += 1
                            
                            for cat, result in category_results.items():
                                percentage = (result['correct'] / result['total']) * 100
                                st.markdown(f"**{cat}:** {result['correct']}/{result['total']} ({percentage:.0f}%)")
                        else:
                            st.info("لا توجد نتائج تفصيلية متاحة")

                    # Certificate download option
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"📜 تحميل الشهادة", key=f"cert_{submission['id']}", use_container_width=True):
                            st.success("✅ تم إنشاء الشهادة - سيتم التحميل تلقائياً")
                    
                    with col2:
                        if st.button(f"📧 إرسال للبريد", key=f"email_{submission['id']}", use_container_width=True):
                            st.success("✅ تم إرسال النتائج لبريدك الإلكتروني")
                    
                    with col3:
                        if st.button(f"📱 مشاركة النتيجة", key=f"share_{submission['id']}", use_container_width=True):
                            st.success("✅ تم إنشاء رابط المشاركة")
                
                elif status == 'needs_revision':
                    # Show revision status
                    st.warning("""
                    **🔄 النتائج قيد المراجعة الإضافية**
                    
                    تم طلب مراجعة إضافية لنتائج اختبارك من قبل الخبراء المختصين.
                    سيتم إشعارك بالنتائج المحدثة خلال 24-48 ساعة.
                    """)
                    
                    if sme_review and sme_review.get('sme_comments'):
                        st.markdown("### 💬 ملاحظات الخبير:")
                        st.info(sme_review['sme_comments'])
                
                else:
                    # Show pending status
                    st.info("""
                    **⏳ النتائج قيد المراجعة**
                    
                    تم تصحيح اختبارك آلياً وإرسال النتائج للخبراء المختصين للمراجعة والاعتماد.
                    ستصلك النتيجة النهائية خلال 24-48 ساعة من تاريخ التقديم.
                    """)
                    
                    # Show estimated time
                    submission_time = datetime.datetime.fromisoformat(submission['timestamp'])
                    hours_passed = (datetime.datetime.now() - submission_time).total_seconds() / 3600
                    estimated_remaining = max(0, 48 - hours_passed)
                    
                    if estimated_remaining > 0:
                        st.markdown(f"**⏰ الوقت المتوقع المتبقي:** {estimated_remaining:.1f} ساعة")
                    else:
                        st.markdown("**⏰ متوقع صدور النتيجة قريباً**")

# Check if exam was just submitted - show submission confirmation
elif st.session_state.exam_submitted and st.session_state.submission_data:
    submission = st.session_state.submission_data
    
    # Success header
    st.success("🎉 تم تسليم الاختبار بنجاح!")
    st.markdown("## شكراً لك على إكمال الاختبار")
    
    # Exam summary in a clean container
    st.markdown("### 📊 ملخص الاختبار:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **📝 عنوان الاختبار:**  
        {submission['exam_title']}
        
        **⏰ وقت التسليم:**  
        {datetime.datetime.fromisoformat(submission['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
        """)
    
    with col2:
        st.info(f"""
        **🕐 الوقت المستغرق:**  
        {submission['time_taken']} دقيقة
        
        **🔢 معرف التسليم:**  
        {submission['id']}
        """)
    
    st.markdown("---")
    
    # Review process explanation
    st.markdown("### 📋 عملية المراجعة والتصحيح")
    
    # Step 1: Auto correction
    st.markdown("""
    #### 🤖 المرحلة 1: التصحيح الآلي
    ✅ تم تصحيح إجاباتك تلقائياً بواسطة النظام الذكي وحساب النتيجة الأولية.
    """)
    
    # Step 2: Expert review
    st.markdown("""
    #### 👨‍🏫 المرحلة 2: مراجعة الخبراء
    🔄 سيتم الآن إرسال نتائجك للخبراء المختصين لمراجعة التصحيح والتأكد من دقة النتائج.
    """)
    
    # Step 3: Results notification
    st.markdown("""
    #### 📧 إشعار النتائج
    📱 ستصلك النتيجة النهائية عبر البريد الإلكتروني أو رسالة نصية خلال **24-48 ساعة** من تاريخ التسليم.
    """)
    
    # Important notice
    st.warning("""
    **💡 ملاحظة مهمة:**  
    النتائج النهائية ستكون متاحة فقط بعد موافقة واعتماد الخبراء المختصين لضمان أعلى مستويات الدقة والعدالة.
    """)
    
    # Action buttons
    st.markdown("---")
    st.markdown("### 📞 خدمات إضافية")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 إرسال إيصال بالبريد", use_container_width=True):
            st.success("✅ تم إرسال إيصال التسليم لبريدك الإلكتروني")
    
    with col2:
        if st.button("📱 إرسال رسالة نصية", use_container_width=True):
            st.success("✅ تم إرسال رسالة تأكيد لرقم هاتفك")
    
    with col3:
        if st.button("🏠 العودة للصفحة الرئيسية", use_container_width=True):
            # Clear submission state
            st.session_state.exam_submitted = False
            st.session_state.submission_data = None
            st.experimental_rerun()
    
    # Contact information
    st.markdown("---")
    st.markdown("### 📞 معلومات الاتصال")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📞 الدعم الفني:**  
        123-456-7890
        
        **📧 البريد الإلكتروني:**  
        support@mol.gov.om
        """)
    
    with col2:
        st.info("""
        **🕐 ساعات العمل:**  
        الأحد - الخميس  
        8:00 ص - 5:00 م
        """)

# Check if exam is in progress
elif st.session_state.exam_started and st.session_state.selected_exam:
    exam = st.session_state.selected_exam
    exam_questions = [q for q in db["questions"] if q["id"] in exam["question_ids"] and q.get("status") == "approved"]
    
    if not exam_questions:
        st.error("❌ لا توجد أسئلة صالحة لهذا الاختبار")
        st.session_state.exam_started = False
        st.experimental_rerun()
    
    # Shuffle questions if enabled
    if exam.get('settings', {}).get('shuffle_questions', True) and 'shuffled_questions' not in st.session_state:
        shuffled = exam_questions.copy()
        random.shuffle(shuffled)
        st.session_state.shuffled_questions = shuffled
    elif 'shuffled_questions' not in st.session_state:
        st.session_state.shuffled_questions = exam_questions
    
    questions = st.session_state.shuffled_questions
    
    # Timer display
    if st.session_state.start_time:
        elapsed_time = (datetime.datetime.now() - st.session_state.start_time).seconds
        time_limit_seconds = exam.get('time_limit', 30) * 60
        remaining_time = max(0, time_limit_seconds - elapsed_time)
        
        if remaining_time > 0:
            minutes = remaining_time // 60
            seconds = remaining_time % 60
            st.markdown(f"""
            <div class="timer-widget">
                ⏰ الوقت المتبقي: {minutes:02d}:{seconds:02d}
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-refresh timer every 10 seconds
            if remaining_time % 10 == 0:
                time.sleep(1)
                st.experimental_rerun()
        else:
            st.error("⏰ انتهى وقت الاختبار!")
            # Auto submit when time expires
            st.session_state.exam_started = False
            st.info("تم تسليم الاختبار تلقائياً عند انتهاء الوقت")
            st.experimental_rerun()
    
    # Progress indicator
    progress = (st.session_state.current_question + 1) / len(questions)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(progress, text=f"السؤال {st.session_state.current_question + 1} من {len(questions)} ({progress * 100:.0f}% مكتمل)")
    with col2:
        answered_count = len(st.session_state.answers)
        st.metric("الأسئلة المجابة", f"{answered_count}/{len(questions)}")
    
    # Current question
    if st.session_state.current_question < len(questions):
        current_q = questions[st.session_state.current_question]
        
        st.markdown(f"""
        <div class="question-container">
            <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 20px;">
                <h3 style="color: #1e3a8a;">السؤال {st.session_state.current_question + 1}</h3>
                <span style="background: #e0f2fe; color: #0891b2; padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 500;">
                    {current_q['category']}
                </span>
            </div>
            <div class="question-text">
                {current_q['text']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Answer options
        st.markdown("### اختر الإجابة الصحيحة:")
        
        options = current_q['options'].copy()
        
        # Shuffle options if enabled
        if exam.get('settings', {}).get('shuffle_options', False):
            if f"shuffled_options_{current_q['id']}" not in st.session_state:
                options_with_correct = list(zip(options, [opt == current_q['answer'] for opt in options]))
                random.shuffle(options_with_correct)
                st.session_state[f"shuffled_options_{current_q['id']}"] = options_with_correct
            else:
                options_with_correct = st.session_state[f"shuffled_options_{current_q['id']}"]
            options = [opt for opt, is_correct in options_with_correct]
        
        # Get previously selected answer
        current_answer = st.session_state.answers.get(current_q['id'])
        
        # Create unique key for radio button  
        radio_key = f"q_{current_q['id']}_radio_{st.session_state.current_question}"
        
        # Ensure we have valid options and handle index safely
        if not options or len(options) == 0:
            st.error("خطأ: لا توجد خيارات متاحة لهذا السؤال")
        else:
            # Safely determine the index
            try:
                if current_answer and current_answer in options:
                    current_index = options.index(current_answer)
                else:
                    current_index = 0  # Default to first option
            except (ValueError, IndexError):
                current_index = 0
            
            # Ensure index is within valid range
            current_index = max(0, min(current_index, len(options) - 1))
            
            selected_answer = st.radio(
                "",
                options=options,
                index=current_index,
                key=radio_key,
                label_visibility="collapsed"
            )
            
            # Save answer automatically
            if selected_answer:
                st.session_state.answers[current_q['id']] = selected_answer
                st.success("💾 تم حفظ إجابتك تلقائياً")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.current_question > 0:
                if st.button("⏮️ السؤال السابق", use_container_width=True):
                    st.session_state.current_question -= 1
                    st.experimental_rerun()
        
        with col2:
            # Question navigation menu
            with st.expander("📋 الانتقال السريع للأسئلة"):
                cols = st.columns(5)
                for i in range(len(questions)):
                    col_idx = i % 5
                    with cols[col_idx]:
                        answered = questions[i]['id'] in st.session_state.answers
                        button_text = f"{'✅' if answered else '⭕'} {i+1}"
                        if st.button(button_text, key=f"nav_{i}", help=f"انتقل للسؤال {i+1}"):
                            st.session_state.current_question = i
                            st.experimental_rerun()
        
        with col3:
            if st.session_state.current_question < len(questions) - 1:
                if st.button("⏭️ السؤال التالي", use_container_width=True):
                    st.session_state.current_question += 1
                    st.experimental_rerun()
    
    # Submit exam section
    if st.session_state.current_question >= len(questions) - 1:
        st.markdown("---")
        st.markdown("### 🎯 إنهاء الاختبار")
        
        answered_count = len(st.session_state.answers)
        total_questions = len(questions)
        completion_rate = (answered_count / total_questions) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("الأسئلة المجابة", f"{answered_count}/{total_questions}")
        with col2:
            st.metric("نسبة الإكمال", f"{completion_rate:.0f}%")
        with col3:
            # Calculate actual time taken
            if st.session_state.start_time:
                time_taken_seconds = (datetime.datetime.now() - st.session_state.start_time).total_seconds()
                time_taken = max(1, int(time_taken_seconds // 60))  # At least 1 minute
            else:
                time_taken = 1  # Default to 1 minute if no start time
            st.metric("الوقت المستغرق", f"{time_taken} دقيقة")
        
        if answered_count < total_questions:
            st.warning(f"⚠️ لم تجب على {total_questions - answered_count} أسئلة. هل تريد المتابعة والتسليم؟")
        
        if st.button("📤 تسليم الاختبار النهائي", type="primary", use_container_width=True):
            # Auto-correct the exam
            correction_results = auto_correct_exam(questions, st.session_state.answers)
            
            # Create submission record
            submission = {
                "id": f"sub_{int(datetime.datetime.now().timestamp())}",
                "exam_id": exam["id"],
                "exam_title": exam["title"],
                "student_name": st.session_state.student_name,
                "student_id": st.session_state.student_id,
                "department": st.session_state.department,
                "position": st.session_state.position,
                "answers": st.session_state.answers,
                "score": correction_results["correct_answers"],
                "total_questions": correction_results["total_questions"],
                "percentage": correction_results["percentage"],
                "time_taken": time_taken,  # Use the calculated time
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "submitted",
                "auto_correction": correction_results,
                "sme_approved": False,
                "final_results_released": False
            }
            
            # Create SME review record
            sme_review = create_sme_review_record(submission, correction_results)
            
            # Save to database
            if "sme_reviews" not in db:
                db["sme_reviews"] = []
            
            db["submissions"].append(submission)
            db["sme_reviews"].append(sme_review)
            save_db(db)
            
            # Clear exam session and set submission state
            st.session_state.exam_started = False
            st.session_state.current_question = 0
            st.session_state.answers = {}
            st.session_state.start_time = None
            st.session_state.selected_exam = None
            if 'shuffled_questions' in st.session_state:
                del st.session_state.shuffled_questions
            
            # Set submission confirmation state
            st.session_state.exam_submitted = True
            st.session_state.submission_data = submission
            
            st.experimental_rerun()

else:
    # Exam selection interface
    st.subheader("📋 الاختبارات المتاحة")
    
    available_exams = db.get("exams", [])
    
    if not available_exams:
        st.warning("📭 لا توجد اختبارات متاحة حالياً")
        st.info("""
        💡 **لمشاهدة اختبارات:**
        1. انتقل لصفحة "توليد الأسئلة" وأنشئ أسئلة
        2. اعتمد الأسئلة في صفحة "مراجعة الخبراء"
        3. أنشئ اختباراً في صفحة "تجميع الاختبارات"
        4. عُد هنا لتجربة الاختبار
        """)
    else:
        for exam in available_exams:
            # Check if exam has approved questions
            exam_questions = [q for q in db["questions"] if q["id"] in exam["question_ids"] and q.get("status") == "approved"]
            
            if not exam_questions:
                continue
            
            with st.expander(f"📝 {exam['title']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**الوصف:** {exam.get('description', 'اختبار القدرات المعرفية الشامل')}")
                    
                    st.markdown(f"""
                    **تفاصيل الاختبار:**
                    - 📊 عدد الأسئلة: **{len(exam_questions)}** سؤال
                    - ⏰ المدة المحددة: **{exam.get('time_limit', 30)}** دقيقة  
                    - 📈 مستوى الصعوبة: **{exam.get('difficulty', 'متوسط')}**
                    - 📝 نوع الأسئلة: اختيار من متعدد
                    - 🔀 ترتيب الأسئلة: {'عشوائي' if exam.get('settings', {}).get('shuffle_questions') else 'ثابت'}
                    """)
                    
                    # Question categories breakdown
                    categories = {}
                    for q in exam_questions:
                        cat = q['category']
                        categories[cat] = categories.get(cat, 0) + 1
                    
                    st.markdown("**توزيع الأسئلة حسب الفئات:**")
                    for cat, count in categories.items():
                        st.caption(f"• {cat}: {count} سؤال")
                
                with col2:
                    st.markdown("**إرشادات مهمة:**")
                    st.info("""
                    ✅ اقرأ كل سؤال بعناية فائقة
                    ✅ اختر إجابة واحدة فقط لكل سؤال  
                    ✅ يمكنك العودة للأسئلة السابقة
                    ✅ احفظ إجابتك قبل الانتقال
                    ✅ راقب الوقت المتبقي باستمرار
                    ⚠️ لا يمكن إيقاف الاختبار مؤقتاً
                    """)
                    
                    estimated_time = len(exam_questions) * 2  # 2 minutes per question
                    if estimated_time <= exam.get('time_limit', 30):
                        st.success(f"⏰ الوقت كافٍ ({estimated_time} دقيقة مقدرة)")
                    else:
                        st.warning(f"⚠️ الوقت محدود ({estimated_time} دقيقة مقدرة)")
                    
                    if st.button(f"🚀 ابدأ الاختبار", key=f"start_{exam['id']}", type="primary", use_container_width=True):
                        st.session_state.selected_exam = exam
                        st.session_state.exam_started = True
                        st.session_state.start_time = datetime.datetime.now()
                        st.session_state.current_question = 0
                        st.session_state.answers = {}
                        st.experimental_rerun()
        
        # Filter out exams without valid questions
        valid_exams = []
        for exam in available_exams:
            exam_questions = [q for q in db["questions"] if q["id"] in exam["question_ids"] and q.get("status") == "approved"]
            if exam_questions:
                valid_exams.append(exam)
        
        if not valid_exams:
            st.warning("⚠️ لا توجد اختبارات بأسئلة صالحة متاحة حالياً")

# Technical information and help
with st.expander("📖 معلومات تقنية وإرشادات"):
    st.markdown("""
    ### 🔧 البيئة التقنية:
    
    **متطلبات النظام:**
    - متصفح حديث (Chrome, Firefox, Safari, Edge)
    - اتصال إنترنت مستقر (الحد الأدنى 1 Mbps)
    - دقة شاشة: 1024×768 أو أعلى
    - تفعيل JavaScript
    
    **الأمان والخصوصية:**
    - تشفير شامل لجميع البيانات المرسلة
    - حفظ تلقائي للإجابات كل 30 ثانية  
    - عدم إمكانية الوصول لمواقع خارجية أثناء الاختبار
    - تسجيل كامل لسجل الأنشطة
    
    **عملية التصحيح والمراجعة:**
    - تصحيح آلي فوري للإجابات
    - مراجعة خبراء مختصين لضمان الدقة
    - إشعارات تلقائية بالنتائج النهائية
    - تقارير مفصلة لكل طالب
    
    **في حالة المشاكل التقنية:**
    - تأكد من استقرار الاتصال
    - أعد تحميل الصفحة (F5)
    - اتصل بالدعم الفني: 123-456-7890
    - البريد الإلكتروني: support@mol.gov.om
    
    ### 📞 الدعم والمساعدة:
    - **الخط الساخن**: متاح 24/7 أثناء فترات الاختبار
    - **الدعم الفني**: فريق متخصص لحل المشاكل فوراً  
    - **دليل المستخدم**: متاح للتحميل قبل الاختبار
    """)

# Footer - Clean and simple
st.markdown("---")
st.markdown("""
**منصة تقييم القدرات المعرفية - وزارة العمل، سلطنة عُمان**  
واجهة آمنة ومعتمدة لإجراء الاختبارات الرسمية  
جميع الحقوق محفوظة © ٢٠٢٥ | النسخة التجريبية المتقدمة
""")
