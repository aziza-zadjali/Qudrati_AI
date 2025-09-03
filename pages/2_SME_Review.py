import streamlit as st
import json
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(page_title="مراجعة الخبراء", page_icon="👨‍🏫", layout="wide")

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

.review-card {
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin: 25px 0;
    border-right: 5px solid #1e3a8a;
}

.question-text {
    background: #f8fafc;
    padding: 25px;
    border-radius: 12px;
    border-right: 3px solid #3b82f6;
    font-size: 20px;
    font-weight: 500;
    margin: 20px 0;
    line-height: 1.8;
}

.option-correct {
    background: #dcfce7;
    border-right: 3px solid #16a34a;
    padding: 15px;
    border-radius: 10px;
    margin: 8px 0;
    font-weight: 500;
}

.option-incorrect {
    background: #f8fafc;
    border-right: 3px solid #6b7280;
    padding: 15px;
    border-radius: 10px;
    margin: 8px 0;
}

.option-student-wrong {
    background: #fee2e2;
    border-right: 3px solid #ef4444;
    padding: 15px;
    border-radius: 10px;
    margin: 8px 0;
    font-weight: 500;
}

.exam-review-card {
    background: #fef9c3;
    border-right: 4px solid #f59e0b;
    padding: 25px;
    border-radius: 12px;
    margin: 20px 0;
}

.approve-btn {
    background: linear-gradient(45deg, #16a34a, #22c55e) !important;
    color: white !important;
    border: none !important;
    font-weight: 600;
    font-size: 16px !important;
}

.reject-btn {
    background: linear-gradient(45deg, #dc2626, #ef4444) !important;
    color: white !important;
    border: none !important;
    font-weight: 600;
    font-size: 16px !important;
}

.needs-revision-btn {
    background: linear-gradient(45deg, #f59e0b, #fbbf24) !important;
    color: white !important;
    border: none !important;
    font-weight: 600;
    font-size: 16px !important;
}

.notification-sent {
    background: #dcfce7;
    border-right: 4px solid #16a34a;
    padding: 20px;
    border-radius: 12px;
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
        return {"questions": [], "exams": [], "submissions": [], "sme_reviews": [], "notifications": []}

def save_db(data):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def send_results_notification(submission, sme_review):
    """Simulate sending notification to student about approved results"""
    # In real system, this would send email/SMS to student
    notification = {
        "id": f"notif_{submission['id']}_{int(datetime.now().timestamp())}",
        "student_id": submission['student_id'],
        "student_name": submission['student_name'],
        "student_email": f"{submission['student_name'].replace(' ', '.')}@mol.gov.om",  # Mock email
        "message": f"تم اعتماد نتائج اختبار '{submission['exam_title']}' من قبل الخبير {sme_review.get('sme_reviewer', 'غير محدد')}",
        "message_details": f"حصلت على {submission['score']}/{submission['total_questions']} ({submission['percentage']:.1f}%)",
        "type": "results_approved",
        "exam_title": submission['exam_title'],
        "score": submission['score'],
        "total_questions": submission['total_questions'],
        "percentage": submission['percentage'],
        "performance_level": submission.get('auto_correction', {}).get('performance_level', 'غير محدد'),
        "sent_at": datetime.now().isoformat(),
        "status": "sent",
        "channels": ["email", "sms", "portal"]
    }
    return notification

def send_revision_notification(submission, sme_review):
    """Send notification about results needing revision"""
    notification = {
        "id": f"notif_rev_{submission['id']}_{int(datetime.now().timestamp())}",
        "student_id": submission['student_id'],
        "student_name": submission['student_name'],
        "student_email": f"{submission['student_name'].replace(' ', '.')}@mol.gov.om",  # Mock email
        "message": f"نتائج اختبار '{submission['exam_title']}' قيد المراجعة الإضافية",
        "message_details": "سيتم إشعارك بالنتيجة المحدثة خلال 24-48 ساعة",
        "type": "results_under_revision",
        "exam_title": submission['exam_title'],
        "reviewer_comments": sme_review,
        "sent_at": datetime.now().isoformat(),
        "status": "sent",
        "channels": ["email", "portal"]
    }
    return notification

def approve_exam_results(db, sme_review_id, reviewer_name, comments=""):
    """Approve exam results and update submission status"""
    for i, review in enumerate(db["sme_reviews"]):
        if review["id"] == sme_review_id:
            # Update SME review
            db["sme_reviews"][i]["sme_status"] = "approved"
            db["sme_reviews"][i]["sme_reviewer"] = reviewer_name
            db["sme_reviews"][i]["sme_comments"] = comments
            db["sme_reviews"][i]["sme_reviewed_at"] = datetime.now().isoformat()
            
            # Update corresponding submission
            submission_id = review["submission_id"]
            for j, submission in enumerate(db["submissions"]):
                if submission["id"] == submission_id:
                    db["submissions"][j]["status"] = "approved"
                    db["submissions"][j]["sme_approved"] = True
                    db["submissions"][j]["final_results_released"] = True
                    db["submissions"][j]["sme_reviewer"] = reviewer_name
                    db["submissions"][j]["sme_review_date"] = datetime.now().isoformat()
                    db["submissions"][j]["sme_comments"] = comments
                    break
            break

def reject_exam_results(db, sme_review_id, reviewer_name, comments=""):
    """Reject exam results and mark for revision"""
    for i, review in enumerate(db["sme_reviews"]):
        if review["id"] == sme_review_id:
            # Update SME review
            db["sme_reviews"][i]["sme_status"] = "needs_revision"
            db["sme_reviews"][i]["sme_reviewer"] = reviewer_name
            db["sme_reviews"][i]["sme_comments"] = comments
            db["sme_reviews"][i]["sme_reviewed_at"] = datetime.now().isoformat()
            
            # Update corresponding submission
            submission_id = review["submission_id"]
            for j, submission in enumerate(db["submissions"]):
                if submission["id"] == submission_id:
                    db["submissions"][j]["status"] = "needs_revision"
                    db["submissions"][j]["sme_approved"] = False
                    db["submissions"][j]["final_results_released"] = False
                    db["submissions"][j]["sme_reviewer"] = reviewer_name
                    db["submissions"][j]["sme_review_date"] = datetime.now().isoformat()
                    db["submissions"][j]["sme_comments"] = comments
                    break
            break

# Initialize session state for navigation
if 'current_review_index' not in st.session_state:
    st.session_state.current_review_index = 0
if 'last_action' not in st.session_state:
    st.session_state.last_action = None
if 'questions_reviewed' not in st.session_state:
    st.session_state.questions_reviewed = []
if 'review_mode' not in st.session_state:
    st.session_state.review_mode = "questions"  # questions or exams
if 'current_exam_review_index' not in st.session_state:
    st.session_state.current_exam_review_index = 0

# Main UI
st.title("👨‍🏫 بوابة مراجعة الخبراء المتطورة")

st.markdown("""
<div style="background: #f0f9ff; padding: 25px; border-radius: 15px; border-right: 4px solid #0891b2; margin: 25px 0;">
<h3 style="color: #0c4a6e; margin-top: 0;">🔍 المرحلة الأولى: ضمان الجودة الشاملة</h3>
<p style="font-size: 16px; color: #075985;">
مراجعة شاملة للأسئلة المولدة وتقييم نتائج الاختبارات المصححة آلياً قبل إصدار النتائج النهائية للطلاب
</p>
</div>
""", unsafe_allow_html=True)

# Load data and calculate statistics
try:
    db = load_db()
    all_questions = db.get('questions', [])
    pending_questions = [q for q in all_questions if q.get('status') == 'pending']
    approved_questions = [q for q in all_questions if q.get('status') == 'approved']
    rejected_questions = [q for q in all_questions if q.get('status') == 'rejected']
    
    # SME review data for exam results
    sme_reviews = db.get('sme_reviews', [])
    pending_exam_reviews = [r for r in sme_reviews if r.get('sme_status') == 'pending_review']
    approved_exam_reviews = [r for r in sme_reviews if r.get('sme_status') == 'approved']
    revision_exam_reviews = [r for r in sme_reviews if r.get('sme_status') == 'needs_revision']
    
    # Notifications
    notifications = db.get('notifications', [])
    
    total_questions = len(all_questions)
    total_exam_reviews = len(sme_reviews)
    
    # Review mode selector
    st.subheader("🎯 نوع المراجعة")
    
    review_mode = st.radio(
        "",
        ["مراجعة الأسئلة الجديدة", "مراجعة نتائج الاختبارات", "عرض الإشعارات المرسلة"],
        horizontal=True,
        key="review_mode_selector"
    )
    
    if review_mode == "مراجعة نتائج الاختبارات":
        st.session_state.review_mode = "exams"
    elif review_mode == "عرض الإشعارات المرسلة":
        st.session_state.review_mode = "notifications"
    else:
        st.session_state.review_mode = "questions"
    
    # Statistics Dashboard
    st.subheader("📊 لوحة إحصائيات المراجعة")
    
    if st.session_state.review_mode == "questions":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔄 أسئلة في الانتظار", 
                len(pending_questions),
                help="الأسئلة التي تحتاج لمراجعة من الخبراء"
            )
        
        with col2:
            st.metric(
                "✅ أسئلة معتمدة", 
                len(approved_questions),
                help="الأسئلة المعتمدة والجاهزة للاستخدام"
            )
        
        with col3:
            st.metric(
                "❌ أسئلة مرفوضة", 
                len(rejected_questions),
                help="الأسئلة المرفوضة وتحتاج إعادة نظر"
            )
        
        with col4:
            if total_questions > 0:
                approval_rate = (len(approved_questions) / total_questions) * 100
                st.metric(
                    "📈 معدل الموافقة", 
                    f"{approval_rate:.1f}%",
                    help="نسبة الأسئلة المعتمدة من المجموع الكلي"
                )
            else:
                st.metric("📈 معدل الموافقة", "0%")
    
    elif st.session_state.review_mode == "exams":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔄 نتائج في الانتظار", 
                len(pending_exam_reviews),
                help="نتائج الاختبارات التي تحتاج مراجعة خبراء"
            )
        
        with col2:
            st.metric(
                "✅ نتائج معتمدة", 
                len(approved_exam_reviews),
                help="النتائج المعتمدة والمرسلة للطلاب"
            )
        
        with col3:
            st.metric(
                "🔄 تحتاج مراجعة", 
                len(revision_exam_reviews),
                help="النتائج التي تحتاج مراجعة إضافية"
            )
        
        with col4:
            if total_exam_reviews > 0:
                exam_approval_rate = (len(approved_exam_reviews) / total_exam_reviews) * 100
                st.metric(
                    "📈 معدل الموافقة", 
                    f"{exam_approval_rate:.1f}%",
                    help="نسبة النتائج المعتمدة"
                )
            else:
                st.metric("📈 معدل الموافقة", "0%")
        
        # Additional exam review metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            if pending_exam_reviews:
                avg_score = sum(r.get('auto_correction', {}).get('percentage', 0) for r in pending_exam_reviews) / len(pending_exam_reviews)
                st.metric("متوسط درجات قيد المراجعة", f"{avg_score:.1f}%")
            else:
                st.metric("متوسط درجات قيد المراجعة", "0%")
        
        with col2:
            # Count high priority reviews
            high_priority = len([r for r in pending_exam_reviews if r.get('priority') == 'high'])
            st.metric("مراجعات عالية الأولوية", high_priority)
        
        with col3:
            # Count notifications sent today
            today = datetime.now().strftime('%Y-%m-%d')
            today_notifications = len([n for n in notifications if n.get('sent_at', '').startswith(today)])
            st.metric("الإشعارات المرسلة اليوم", today_notifications)
    
    elif st.session_state.review_mode == "notifications":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("إجمالي الإشعارات", len(notifications))
        
        with col2:
            approved_notifications = len([n for n in notifications if n.get('type') == 'results_approved'])
            st.metric("إشعارات الموافقة", approved_notifications)
        
        with col3:
            revision_notifications = len([n for n in notifications if n.get('type') == 'results_under_revision'])
            st.metric("إشعارات المراجعة", revision_notifications)
        
        with col4:
            today = datetime.now().strftime('%Y-%m-%d')
            today_notifications = len([n for n in notifications if n.get('sent_at', '').startswith(today)])
            st.metric("اليوم", today_notifications)

    # NOTIFICATIONS VIEW MODE
    if st.session_state.review_mode == "notifications":
        st.markdown("---")
        st.subheader("📧 الإشعارات المرسلة للطلاب")
        
        if not notifications:
            st.info("📭 لا توجد إشعارات مرسلة بعد")
        else:
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                notification_type_filter = st.selectbox(
                    "فلترة حسب النوع:",
                    ["جميع الأنواع", "نتائج معتمدة", "قيد المراجعة"],
                    key="notif_type_filter"
                )
            
            with col2:
                date_filter = st.selectbox(
                    "فلترة حسب التاريخ:",
                    ["جميع التواريخ", "اليوم", "أمس", "آخر أسبوع"],
                    key="notif_date_filter"
                )
            
            # Filter notifications
            filtered_notifications = notifications.copy()
            
            if notification_type_filter == "نتائج معتمدة":
                filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'results_approved']
            elif notification_type_filter == "قيد المراجعة":
                filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'results_under_revision']
            
            # Sort by date (newest first)
            filtered_notifications.sort(key=lambda x: x.get('sent_at', ''), reverse=True)
            
            # Display notifications
            for i, notification in enumerate(filtered_notifications[:20]):  # Show latest 20
                notification_time = datetime.fromisoformat(notification['sent_at']).strftime('%Y-%m-%d %H:%M')
                
                if notification['type'] == 'results_approved':
                    icon = "✅"
                    color = "#dcfce7"
                    border_color = "#16a34a"
                else:
                    icon = "🔄" 
                    color = "#fef3c7"
                    border_color = "#f59e0b"
                
                with st.expander(f"{icon} {notification['student_name']} - {notification['exam_title']} - {notification_time}", expanded=False):
                    st.markdown(f"""
                    <div style="background: {color}; border-right: 4px solid {border_color}; padding: 20px; border-radius: 12px; margin: 10px 0;">
                        <h4 style="margin-top: 0; color: #1f2937;">📧 تفاصيل الإشعار</h4>
                        <p><strong>الطالب:</strong> {notification['student_name']} ({notification['student_id']})</p>
                        <p><strong>الاختبار:</strong> {notification['exam_title']}</p>
                        <p><strong>الرسالة:</strong> {notification['message']}</p>
                        <p><strong>التفاصيل:</strong> {notification.get('message_details', 'غير متوفر')}</p>
                        <p><strong>تاريخ الإرسال:</strong> {notification_time}</p>
                        <p><strong>القنوات:</strong> {', '.join(notification.get('channels', []))}</p>
                        <p><strong>الحالة:</strong> {'تم الإرسال ✅' if notification.get('status') == 'sent' else 'فشل الإرسال ❌'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if notification['type'] == 'results_approved':
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("النتيجة", f"{notification.get('score', 0)}/{notification.get('total_questions', 0)}")
                        with col2:
                            st.metric("النسبة", f"{notification.get('percentage', 0):.1f}%")
                        with col3:
                            st.metric("المستوى", notification.get('performance_level', 'غير محدد'))

    # EXAM REVIEW INTERFACE
    elif st.session_state.review_mode == "exams":
        st.markdown("---")
        st.subheader("🔍 مراجعة نتائج الاختبارات")
        
        if pending_exam_reviews:
            # Exam review selector
            selected_exam_index = st.selectbox(
                "اختر نتيجة اختبار للمراجعة:",
                range(len(pending_exam_reviews)),
                index=st.session_state.current_exam_review_index,
                format_func=lambda x: f"طالب: {pending_exam_reviews[x]['student_name']} - اختبار: {pending_exam_reviews[x]['exam_title']} - النتيجة: {pending_exam_reviews[x]['auto_correction']['percentage']:.1f}%",
                key="exam_review_selector"
            )
            
            if selected_exam_index != st.session_state.current_exam_review_index:
                st.session_state.current_exam_review_index = selected_exam_index
            
            current_review = pending_exam_reviews[st.session_state.current_exam_review_index]
            auto_correction = current_review.get('auto_correction', {})
            
            # Exam review details card
            st.markdown(f"""
            <div class="exam-review-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h3 style="color: #92400e; margin: 0;">مراجعة نتيجة الاختبار: {current_review['id']}</h3>
                    <div style="background: #fbbf24; color: white; padding: 8px 16px; border-radius: 25px; font-size: 14px; font-weight: 600;">
                        {auto_correction.get('performance_level', 'غير محدد')}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Student and exam information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 👤 بيانات الطالب")
                st.markdown(f"""
                <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border-right: 3px solid #6b7280;">
                    <strong>الاسم:</strong> {current_review['student_name']}<br>
                    <strong>رقم الهوية:</strong> {current_review['student_id']}<br>
                    <strong>القسم:</strong> {current_review['department']}<br>
                    <strong>تاريخ التقديم:</strong> {datetime.fromisoformat(current_review['created_at']).strftime('%Y-%m-%d %H:%M')}<br>
                    <strong>حالة الأولوية:</strong> {current_review.get('priority', 'عادية')}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 نتائج التصحيح الآلي")
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    st.metric("الدرجة", f"{auto_correction.get('correct_answers', 0)}/{auto_correction.get('total_questions', 0)}")
                
                with col2b:
                    st.metric("النسبة المئوية", f"{auto_correction.get('percentage', 0):.1f}%")
                
                with col2c:
                    st.metric("الأداء", auto_correction.get('performance_level', 'غير محدد'))
            
            # Detailed question-by-question review
            st.markdown("### 📝 مراجعة تفصيلية للإجابات")
            
            detailed_results = auto_correction.get('detailed_results', [])
            if detailed_results:
                for i, result in enumerate(detailed_results, 1):
                    with st.expander(f"السؤال {i}: {result['question_category']} - {'✅ صحيح' if result['is_correct'] else '❌ خطأ'}", expanded=False):
                        st.markdown(f"**نص السؤال:** {result['question_text']}")
                        
                        # Display options with correct/incorrect highlighting
                        st.markdown("**الخيارات:**")
                        for j, option in enumerate(result['options'], 1):
                            if option == result['correct_answer']:
                                st.markdown(f"""
                                <div class="option-correct">
                                    ✅ {j}. {option} (الإجابة الصحيحة)
                                </div>
                                """, unsafe_allow_html=True)
                            elif option == result['student_answer'] and not result['is_correct']:
                                st.markdown(f"""
                                <div class="option-student-wrong">
                                    ❌ {j}. {option} (إجابة الطالب - خاطئة)
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="option-incorrect">
                                    {j}. {option}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Question metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"**الفئة:** {result['question_category']}")
                        with col2:
                            st.caption(f"**الصعوبة:** {result['question_difficulty']}")
                        with col3:
                            st.caption(f"**النقاط:** {result['points']}")
            
            # SME Review Actions
            st.markdown("---")
            st.subheader("⚡ إجراءات مراجعة النتائج")
            
            # Reviewer name input
            reviewer_name = st.text_input("اسم المراجع:", value="د. محمد الخبير", key=f"reviewer_{current_review['id']}")
            
            # Comments section
            sme_comments = st.text_area(
                "ملاحظات المراجع (اختياري):",
                placeholder="أضف أي ملاحظات أو تعليقات حول مراجعة النتائج...",
                height=100,
                key=f"comments_{current_review['id']}"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ اعتماد النتائج", key=f"approve_exam_{current_review['id']}", help="الموافقة على النتائج وإرسالها للطالب", use_container_width=True):
                    approve_exam_results(db, current_review['id'], reviewer_name, sme_comments)
                    
                    # Get updated submission for notification
                    submission = next(s for s in db["submissions"] if s["id"] == current_review["submission_id"])
                    
                    # Send notification to student
                    notification = send_results_notification(submission, {
                        "sme_reviewer": reviewer_name,
                        "sme_comments": sme_comments
                    })
                    
                    # Store notification in database
                    if "notifications" not in db:
                        db["notifications"] = []
                    db["notifications"].append(notification)
                    
                    save_db(db)
                    
                    st.markdown("""
                    <div class="notification-sent">
                        <h4 style="color: #065f46; margin-top: 0;">✅ تم اعتماد النتائج بنجاح!</h4>
                        <p style="color: #047857; margin: 10px 0;">
                            📧 تم إرسال إشعار للطالب بالنتيجة النهائية عبر:
                        </p>
                        <ul style="color: #047857; margin: 10px 0; padding-right: 20px;">
                            <li>البريد الإلكتروني الرسمي</li>
                            <li>الرسائل النصية SMS</li>
                            <li>بوابة الطالب الإلكترونية</li>
                        </ul>
                        <p style="color: #047857; margin: 10px 0;">
                            🎓 يمكن للطالب الآن تحميل الشهادة والاطلاع على النتائج التفصيلية
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Update navigation
                    remaining_reviews = [r for r in pending_exam_reviews if r["id"] != current_review["id"]]
                    if remaining_reviews:
                        st.session_state.current_exam_review_index = 0
                    
                    st.experimental_rerun()
            
            with col2:
                if st.button("🔄 طلب مراجعة إضافية", key=f"revise_exam_{current_review['id']}", help="طلب مراجعة إضافية للنتائج", use_container_width=True):
                    reject_exam_results(db, current_review['id'], reviewer_name, sme_comments)
                    
                    # Get updated submission for notification
                    submission = next(s for s in db["submissions"] if s["id"] == current_review["submission_id"])
                    
                    # Send revision notification to student
                    notification = send_revision_notification(submission, sme_comments)
                    
                    # Store notification in database
                    if "notifications" not in db:
                        db["notifications"] = []
                    db["notifications"].append(notification)
                    
                    save_db(db)
                    
                    st.warning("🔄 تم تعليق النتائج لمراجعة إضافية")
                    st.info("📧 تم إرسال إشعار للطالب بأن النتائج قيد المراجعة الإضافية")
                    
                    # Update navigation
                    remaining_reviews = [r for r in pending_exam_reviews if r["id"] != current_review["id"]]
                    if remaining_reviews:
                        st.session_state.current_exam_review_index = 0
                    
                    st.experimental_rerun()
            
            with col3:
                if st.button("⚠️ رفع الأولوية", key=f"priority_exam_{current_review['id']}", help="رفع أولوية هذه المراجعة", use_container_width=True):
                    # Update priority
                    for i, review in enumerate(db["sme_reviews"]):
                        if review["id"] == current_review["id"]:
                            db["sme_reviews"][i]["priority"] = "high"
                            break
                    save_db(db)
                    st.info("⚠️ تم رفع أولوية هذه المراجعة")
                    st.experimental_rerun()
            
            # Navigation for exam reviews
            st.markdown("---")
            st.subheader("🧭 التنقل بين مراجعات الاختبارات")
            
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            
            with nav_col1:
                if st.button("⏮️ المراجعة السابقة", disabled=st.session_state.current_exam_review_index == 0, use_container_width=True):
                    st.session_state.current_exam_review_index -= 1
                    st.experimental_rerun()
            
            with nav_col2:
                st.markdown(f"<div style='text-align: center; padding: 10px;'><strong>المراجعة {st.session_state.current_exam_review_index + 1} من {len(pending_exam_reviews)}</strong></div>", unsafe_allow_html=True)
            
            with nav_col3:
                if st.button("⏭️ المراجعة التالية", disabled=st.session_state.current_exam_review_index == len(pending_exam_reviews) - 1, use_container_width=True):
                    st.session_state.current_exam_review_index += 1
                    st.experimental_rerun()
        
        else:
            st.success("🎉 ممتاز! لا توجد نتائج اختبارات في انتظار المراجعة")
            st.info("جميع النتائج تم اعتمادها وإرسالها للطلاب")

    # QUESTIONS REVIEW INTERFACE (existing code)
    elif st.session_state.review_mode == "questions":
        # Handle navigation after actions
        if st.session_state.last_action and pending_questions:
            # Reset to first question if current index is out of bounds
            if st.session_state.current_review_index >= len(pending_questions):
                st.session_state.current_review_index = 0
            st.session_state.last_action = None

        # Progress visualization for questions
        if total_questions > 0:
            reviewed_count = len(approved_questions) + len(rejected_questions)
            progress = reviewed_count / total_questions
            
            st.subheader("📈 تقدم مراجعة الأسئلة")
            st.progress(progress, text=f"تمت مراجعة {reviewed_count} من أصل {total_questions} سؤال ({progress*100:.1f}%)")

        # Review Interface for Questions
        if pending_questions:
            st.markdown("---")
            st.subheader("🔍 مراجعة الأسئلة")
            
            # Ensure current index is valid
            if st.session_state.current_review_index >= len(pending_questions):
                st.session_state.current_review_index = 0
            
            # Question selector
            selected_index = st.selectbox(
                "اختر سؤالاً للمراجعة:",
                range(len(pending_questions)),
                index=st.session_state.current_review_index,
                format_func=lambda x: f"السؤال {x+1}: {pending_questions[x]['category']} - {pending_questions[x]['id']}",
                key="question_selector"
            )
            
            # Update current index when selection changes
            if selected_index != st.session_state.current_review_index:
                st.session_state.current_review_index = selected_index
            
            current_q = pending_questions[st.session_state.current_review_index]
            
            # Question details card
            st.markdown(f"""
            <div class="review-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h3 style="color: #1e3a8a; margin: 0;">مراجعة السؤال: {current_q['id']}</h3>
                    <div style="background: #fef3c7; color: #92400e; padding: 8px 16px; border-radius: 25px; font-size: 14px; font-weight: 600;">
                        {current_q['category']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Question content
            col1, col2 = st.columns([2.5, 1])
            
            with col1:
                st.markdown("### 📝 نص السؤال")
                st.markdown(f"""
                <div class="question-text">
                    {current_q['text']}
                </div>
                """, unsafe_allow_html=True)
                
                # Options display
                st.markdown("### 🎯 الخيارات المتاحة")
                
                for i, option in enumerate(current_q['options'], 1):
                    is_correct = option == current_q['answer']
                    
                    if is_correct:
                        st.markdown(f"""
                        <div class="option-correct">
                            ✅ <strong>{i}. {option}</strong> 
                            <span style="color: #065f46; font-size: 12px; margin-right: 10px;">(الإجابة الصحيحة)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="option-incorrect">
                            {i}. {option}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📋 معلومات إضافية")
                
                info_container = st.container()
                with info_container:
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border-right: 3px solid #6b7280;">
                        <strong>🏷️ الفئة:</strong> {current_q['category']}<br><br>
                        <strong>📊 الصعوبة:</strong> {current_q.get('difficulty', 'غير محدد')}<br><br>
                        <strong>🎯 الموضوع:</strong> {current_q.get('topic', 'عام')}<br><br>
                        <strong>⏰ تاريخ التوليد:</strong><br>{current_q.get('generated_at', 'غير معروف')}<br><br>
                        <strong>🤖 المولد:</strong> {current_q.get('generator', 'غير محدد')}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Review Actions for Questions
            st.markdown("---")
            st.subheader("⚡ إجراءات المراجعة")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("✅ اعتماد السؤال", key=f"approve_{current_q['id']}", help="الموافقة على السؤال واعتماده للاستخدام", use_container_width=True):
                    # Update question status
                    for i, question in enumerate(db["questions"]):
                        if question["id"] == current_q["id"]:
                            db["questions"][i]["status"] = "approved"
                            db["questions"][i]["reviewed_at"] = datetime.now().isoformat()
                            db["questions"][i]["reviewer_action"] = "approved"
                            break
                    
                    save_db(db)
                    
                    # Track action and manage navigation
                    st.session_state.last_action = "approved"
                    st.session_state.questions_reviewed.append(current_q["id"])
                    
                    st.success("✅ تم اعتماد السؤال بنجاح!")
                    
                    # Auto-advance to next question
                    remaining_questions = [q for q in pending_questions if q["id"] != current_q["id"]]
                    if remaining_questions:
                        if st.session_state.current_review_index >= len(remaining_questions):
                            st.session_state.current_review_index = 0
                        st.info("انتقال تلقائي للسؤال التالي...")
                    else:
                        st.info("🎉 تم الانتهاء من مراجعة جميع الأسئلة!")
                    
                    st.experimental_rerun()
            
            with col2:
                if st.button("❌ رفض السؤال", key=f"reject_{current_q['id']}", help="رفض السؤال وعدم استخدامه", use_container_width=True):
                    # Update question status
                    for i, question in enumerate(db["questions"]):
                        if question["id"] == current_q["id"]:
                            db["questions"][i]["status"] = "rejected"
                            db["questions"][i]["reviewed_at"] = datetime.now().isoformat()
                            db["questions"][i]["reviewer_action"] = "rejected"
                            break
                    
                    save_db(db)
                    
                    # Track action and manage navigation
                    st.session_state.last_action = "rejected"
                    st.session_state.questions_reviewed.append(current_q["id"])
                    
                    st.warning("❌ تم رفض السؤال!")
                    
                    # Auto-advance to next question
                    remaining_questions = [q for q in pending_questions if q["id"] != current_q["id"]]
                    if remaining_questions:
                        if st.session_state.current_review_index >= len(remaining_questions):
                            st.session_state.current_review_index = 0
                        st.info("انتقال تلقائي للسؤال التالي...")
                    else:
                        st.info("🎉 تم الانتهاء من مراجعة جميع الأسئلة!")
                    
                    st.experimental_rerun()
            
            with col3:
                if st.button("📝 طلب تعديل", key=f"edit_{current_q['id']}", help="إرسال طلب تعديل للسؤال", use_container_width=True):
                    st.info("📝 تم إرسال طلب التعديل لفريق التطوير")
            
            with col4:
                if st.button("⏭️ تخطي مؤقتاً", key=f"skip_{current_q['id']}", help="تخطي هذا السؤال مؤقتاً", use_container_width=True):
                    # Move to next question without changing status
                    if st.session_state.current_review_index < len(pending_questions) - 1:
                        st.session_state.current_review_index += 1
                    else:
                        st.session_state.current_review_index = 0
                    st.info("⏭️ تم تخطي السؤال مؤقتاً")
                    st.experimental_rerun()
        
        else:
            # No pending questions
            st.markdown("---")
            st.success("🎉 ممتاز! تم الانتهاء من مراجعة جميع الأسئلة في قائمة الانتظار")
            
            if total_questions == 0:
                st.info("💡 لا توجد أسئلة للمراجعة. انتقل إلى صفحة 'توليد الأسئلة' لإنشاء أسئلة جديدة.")

except Exception as e:
    st.error(f"❌ خطأ في تحميل البيانات: {e}")

# Sidebar with reviewer tools
with st.sidebar:
    st.markdown("### 🛠️ أدوات المراجع")
    
    # Quick stats based on review mode
    if 'db' in locals():
        if st.session_state.review_mode == "questions":
            today_questions = [q for q in all_questions if q.get('generated_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))]
            st.metric("الأسئلة الجديدة اليوم", len(today_questions))
        elif st.session_state.review_mode == "exams":
            today_reviews = [r for r in sme_reviews if r.get('created_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))]
            st.metric("المراجعات الجديدة اليوم", len(today_reviews))
        else:
            today = datetime.now().strftime('%Y-%m-%d')
            today_notifications = len([n for n in notifications if n.get('sent_at', '').startswith(today)])
            st.metric("الإشعارات اليوم", today_notifications)
        
        st.metric("متوسط وقت المراجعة", "2.1 دقيقة")
        st.metric("نقاط جودة المراجع", "95%")
    
    # Mode selector
    st.markdown("#### 🎯 نمط المراجعة")
    current_mode_display = {
        "questions": "الأسئلة",
        "exams": "نتائج الاختبارات", 
        "notifications": "الإشعارات"
    }
    current_mode = current_mode_display.get(st.session_state.review_mode, "غير محدد")
    st.info(f"النمط الحالي: {current_mode}")
    
    # Quick actions
    st.markdown("#### ⚡ إجراءات سريعة")
    
    if st.session_state.review_mode == "exams" and pending_exam_reviews:
        if st.button("📧 إشعار الطلاب بالتأخير", use_container_width=True):
            st.success("تم إرسال إشعارات للطلاب بتأخير النتائج")
        
        if st.button("📊 تقرير المراجعات المعلقة", use_container_width=True):
            st.info("تم إنشاء تقرير المراجعات المعلقة")
    
    elif st.session_state.review_mode == "notifications":
        if st.button("📧 إرسال تذكير جماعي", use_container_width=True):
            st.success("تم إرسال تذكير جماعي للطلاب")
        
        if st.button("📊 تقرير الإشعارات", use_container_width=True):
            st.info("تم إنشاء تقرير الإشعارات")
    
    # Help section
    with st.expander("❓ مساعدة المراجعة"):
        if st.session_state.review_mode == "questions":
            st.markdown("""
            **معايير تقييم الأسئلة:**
            - ✅ وضوح وسلامة الصياغة
            - ✅ دقة الإجابة الصحيحة  
            - ✅ منطقية الخيارات الخاطئة
            - ✅ مناسبة المحتوى ثقافياً
            - ✅ ملاءمة مستوى الصعوبة
            """)
        elif st.session_state.review_mode == "exams":
            st.markdown("""
            **معايير مراجعة النتائج:**
            - ✅ دقة التصحيح الآلي
            - ✅ عدالة التقييم
            - ✅ مراجعة الإجابات الحدية
            - ✅ تأكيد صحة النتيجة النهائية
            - ✅ مراعاة الظروف الخاصة
            """)
        else:
            st.markdown("""
            **إدارة الإشعارات:**
            - 📧 متابعة حالة الإرسال
            - 📱 تأكيد وصول الرسائل
            - 🔄 إعادة الإرسال عند الحاجة
            - 📊 تحليل معدلات الاستجابة
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 13px; padding: 15px;">
بوابة مراجعة الخبراء المتطورة - منصة تقييم القدرات المعرفية<br>
مراجعة شاملة للأسئلة ونتائج الاختبارات مع نظام إشعارات متكامل لضمان أعلى معايير الجودة والعدالة
</div>
""", unsafe_allow_html=True)
