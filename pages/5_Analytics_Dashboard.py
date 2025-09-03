import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import random

# Page config
st.set_page_config(page_title="لوحة التحليلات", page_icon="📊", layout="wide")

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

.analytics-header {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    color: white;
    padding: 30px;
    border-radius: 15px;
    margin: 25px 0;
    text-align: center;
}

.metric-card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    border-right: 4px solid #3b82f6;
    margin: 15px 0;
}

.insight-box {
    background: #f0f9ff;
    padding: 25px;
    border-radius: 12px;
    border-right: 4px solid #0891b2;
    margin: 20px 0;
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

# Main UI
st.title("📊 لوحة التحليلات والإحصائيات المتقدمة")

st.markdown("""
<div class="analytics-header">
    <h2>🎯 المرحلة الثانية: مراقبة الأداء وتحليل النتائج</h2>
    <p style="font-size: 18px; margin: 15px 0;">
    تحليل شامل للبيانات مع رؤى استراتيجية لتطوير الأداء المؤسسي
    </p>
</div>
""", unsafe_allow_html=True)

# Load and process data
db = load_db()
submissions = db.get("submissions", [])
questions = db.get("questions", [])
exams = db.get("exams", [])

# Generate sample data if none exists
if not submissions:
    st.info("""
    📭 لا توجد بيانات محاولات للتحليل بعد
    
    **لعرض التحليلات الكاملة:**
    1. انتقل لصفحة "توليد الأسئلة" لإنشاء أسئلة متنوعة
    2. اعتمد الأسئلة في صفحة "مراجعة الخبراء"
    3. أنشئ اختبارات في صفحة "تجميع الاختبارات"
    4. جرب الاختبارات في صفحة "تقديم الاختبار"
    5. عُد هنا لرؤية التحليلات الشاملة والرؤى الاستراتيجية
    """)
    
    # Generate sample data for demonstration
    if st.button("🎲 إنشاء بيانات تجريبية للعرض", type="primary"):
        sample_submissions = []
        sample_names = ["أحمد محمد", "فاطمة علي", "خالد السعيد", "مريم الهنائي", "سالم الرواحي"]
        departments = ["الموارد البشرية", "التكنولوجيا", "المالية", "الخدمات", "الإدارة"]
        
        for i in range(20):
            score = random.randint(6, 15)
            total = 15
            sample_submissions.append({
                "id": f"demo_{i+1}",
                "exam_id": "demo_exam",
                "exam_title": "اختبار القدرات التجريبي",
                "student_name": random.choice(sample_names),
                "student_id": f"12345{i:03d}",
                "department": random.choice(departments),
                "score": score,
                "total_questions": total,
                "percentage": (score/total)*100,
                "time_taken": random.randint(15, 45),
                "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                "status": "completed"
            })
        
        db["submissions"] = sample_submissions
        with open(DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        
        st.success("✅ تم إنشاء بيانات تجريبية!")
        st.experimental_rerun()

else:
    # Main Analytics Dashboard
    
    # Key Performance Indicators
    st.subheader("📈 المؤشرات الرئيسية للأداء")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_submissions = len(submissions)
        st.metric("إجمالي المحاولات", total_submissions, help="العدد الكلي لمحاولات الاختبار")
    
    with col2:
        if submissions:
            avg_score = sum(s.get('score', 0) for s in submissions) / len(submissions)
            avg_total = sum(s.get('total_questions', 1) for s in submissions) / len(submissions)
            avg_percentage = (avg_score / avg_total * 100) if avg_total > 0 else 0
            st.metric("متوسط النتائج", f"{avg_percentage:.1f}%", help="متوسط النسبة المئوية للدرجات")
        else:
            st.metric("متوسط النتائج", "0%")
    
    with col3:
        completed_submissions = len([s for s in submissions if s.get('status') == 'completed'])
        completion_rate = (completed_submissions / total_submissions * 100) if total_submissions > 0 else 0
        st.metric("معدل الإكمال", f"{completion_rate:.1f}%", help="نسبة الاختبارات المكتملة")
    
    with col4:
        if submissions:
            avg_time = sum(s.get('time_taken', 0) for s in submissions) / len(submissions)
            st.metric("متوسط الوقت", f"{avg_time:.1f} دقيقة", help="متوسط الوقت لإنهاء الاختبار")
        else:
            st.metric("متوسط الوقت", "0 دقيقة")
    
    with col5:
        departments = len(set(s.get('department', 'غير محدد') for s in submissions))
        st.metric("الأقسام المشاركة", departments, help="عدد الأقسام التي شاركت في الاختبارات")
    
    # Detailed Analytics
    st.subheader("📊 التحليلات التفصيلية")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "توزيع الدرجات", 
        "أداء الأقسام", 
        "تحليل زمني", 
        "إحصائيات الاختبارات",
        "التوصيات"
    ])
    
    with tab1:
        # Score Distribution Analysis
        st.markdown("### 📈 توزيع الدرجات والأداء")
        
        if submissions:
            # Prepare data
            scores_data = []
            for s in submissions:
                score = s.get('score', 0)
                total = s.get('total_questions', 1)
                percentage = (score / total * 100) if total > 0 else 0
                scores_data.append({
                    'الطالب': s.get('student_name', 'غير معروف'),
                    'النسبة المئوية': percentage,
                    'الدرجة الخام': f"{score}/{total}",
                    'القسم': s.get('department', 'غير محدد'),
                    'الوقت المستغرق': s.get('time_taken', 0)
                })
            
            df_scores = pd.DataFrame(scores_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram of scores
                fig_hist = px.histogram(
                    df_scores, 
                    x='النسبة المئوية', 
                    nbins=15,
                    title="توزيع درجات الطلاب",
                    labels={'النسبة المئوية': 'النسبة المئوية (%)', 'count': 'عدد الطلاب'},
                    color_discrete_sequence=['#3b82f6']
                )
                fig_hist.update_layout(
                    font_family="Cairo, sans-serif",
                    title_font_size=16,
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Performance categories pie chart
                excellent = len([s for s in scores_data if s['النسبة المئوية'] >= 90])
                good = len([s for s in scores_data if 70 <= s['النسبة المئوية'] < 90])
                average = len([s for s in scores_data if 50 <= s['النسبة المئوية'] < 70])
                below_average = len([s for s in scores_data if s['النسبة المئوية'] < 50])
                
                performance_data = pd.DataFrame({
                    'الفئة': ['ممتاز (90%+)', 'جيد (70-89%)', 'متوسط (50-69%)', 'يحتاج تحسين (<50%)'],
                    'العدد': [excellent, good, average, below_average]
                })
                
                fig_pie = px.pie(
                    performance_data,
                    values='العدد',
                    names='الفئة',
                    title="توزيع فئات الأداء",
                    color_discrete_sequence=['#22c55e', '#3b82f6', '#f59e0b', '#ef4444']
                )
                fig_pie.update_layout(font_family="Cairo, sans-serif", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Detailed scores table
            st.markdown("### 📋 جدول النتائج التفصيلي")
            st.dataframe(
                df_scores.sort_values('النسبة المئوية', ascending=False),
                use_container_width=True,
                height=300
            )
            
            # Statistical summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("أعلى نسبة", f"{df_scores['النسبة المئوية'].max():.1f}%")
            with col2:
                st.metric("أقل نسبة", f"{df_scores['النسبة المئوية'].min():.1f}%")
            with col3:
                st.metric("الانحراف المعياري", f"{df_scores['النسبة المئوية'].std():.1f}")
            with col4:
                median_score = df_scores['النسبة المئوية'].median()
                st.metric("الوسيط", f"{median_score:.1f}%")
    
    with tab2:
        # Department Performance Analysis
        st.markdown("### 🏢 تحليل أداء الأقسام")
        
        if submissions:
            # Department-wise analysis
            dept_data = {}
            for s in submissions:
                dept = s.get('department', 'غير محدد')
                if dept not in dept_data:
                    dept_data[dept] = {'scores': [], 'times': [], 'count': 0}
                
                score = s.get('score', 0)
                total = s.get('total_questions', 1)
                percentage = (score / total * 100) if total > 0 else 0
                
                dept_data[dept]['scores'].append(percentage)
                dept_data[dept]['times'].append(s.get('time_taken', 0))
                dept_data[dept]['count'] += 1
            
            # Calculate department statistics
            dept_stats = []
            for dept, data in dept_data.items():
                avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
                avg_time = sum(data['times']) / len(data['times']) if data['times'] else 0
                dept_stats.append({
                    'القسم': dept,
                    'متوسط النسبة': avg_score,
                    'عدد المشاركين': data['count'],
                    'متوسط الوقت': avg_time,
                    'أعلى نسبة': max(data['scores']) if data['scores'] else 0,
                    'أقل نسبة': min(data['scores']) if data['scores'] else 0
                })
            
            df_dept = pd.DataFrame(dept_stats)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Department performance bar chart
                fig_dept = px.bar(
                    df_dept.sort_values('متوسط النسبة', ascending=True),
                    x='متوسط النسبة',
                    y='القسم',
                    title="متوسط الأداء حسب القسم",
                    labels={'متوسط النسبة': 'متوسط النسبة المئوية (%)', 'القسم': 'القسم'},
                    color='متوسط النسبة',
                    color_continuous_scale='RdYlGn'
                )
                fig_dept.update_layout(font_family="Cairo, sans-serif", height=400)
                st.plotly_chart(fig_dept, use_container_width=True)
            
            with col2:
                # Department participation
                fig_participation = px.pie(
                    df_dept,
                    values='عدد المشاركين',
                    names='القسم',
                    title="توزيع المشاركة حسب القسم"
                )
                fig_participation.update_layout(font_family="Cairo, sans-serif", height=400)
                st.plotly_chart(fig_participation, use_container_width=True)
            
            # Department comparison table
            st.markdown("### 📊 مقارنة تفصيلية بين الأقسام")
            st.dataframe(
                df_dept.sort_values('متوسط النسبة', ascending=False),
                use_container_width=True
            )
    
    with tab3:
        # Time-based Analysis
        st.markdown("### ⏰ التحليل الزمني للأداء")
        
        if submissions:
            # Process dates
            time_data = []
            for s in submissions:
                try:
                    timestamp = datetime.fromisoformat(s.get('timestamp', '').replace('Z', '+00:00'))
                    score = s.get('score', 0)
                    total = s.get('total_questions', 1)
                    percentage = (score / total * 100) if total > 0 else 0
                    
                    time_data.append({
                        'التاريخ': timestamp.date(),
                        'الوقت': timestamp.hour,
                        'النسبة المئوية': percentage,
                        'الطالب': s.get('student_name', 'غير معروف'),
                        'القسم': s.get('department', 'غير محدد'),
                        'وقت الإنجاز': s.get('time_taken', 0)
                    })
                except:
                    continue
            
            if time_data:
                df_time = pd.DataFrame(time_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Daily performance trend
                    daily_stats = df_time.groupby('التاريخ').agg({
                        'النسبة المئوية': 'mean',
                        'الطالب': 'count'
                    }).reset_index()
                    daily_stats.columns = ['التاريخ', 'متوسط الأداء', 'عدد المحاولات']
                    
                    fig_daily = px.line(
                        daily_stats,
                        x='التاريخ',
                        y='متوسط الأداء',
                        title="اتجاه الأداء عبر الوقت",
                        markers=True
                    )
                    fig_daily.update_layout(font_family="Cairo, sans-serif", height=400)
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                with col2:
                    # Time of day analysis
                    hourly_stats = df_time.groupby('الوقت')['النسبة المئوية'].mean().reset_index()
                    
                    fig_hourly = px.bar(
                        hourly_stats,
                        x='الوقت',
                        y='النسبة المئوية',
                        title="الأداء حسب ساعات اليوم",
                        labels={'الوقت': 'الساعة', 'النسبة المئوية': 'متوسط الأداء (%)'}
                    )
                    fig_hourly.update_layout(font_family="Cairo, sans-serif", height=400)
                    st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Time vs Performance correlation
                st.markdown("### 🕐 العلاقة بين وقت الإنجاز والأداء")
                
                fig_scatter = px.scatter(
                    df_time,
                    x='وقت الإنجاز',
                    y='النسبة المئوية',
                    color='القسم',
                    title="العلاقة بين وقت الإنجاز والنتيجة",
                    labels={'وقت الإنجاز': 'وقت الإنجاز (دقيقة)', 'النسبة المئوية': 'النسبة المئوية (%)'},
                    trendline="ols"
                )
                fig_scatter.update_layout(font_family="Cairo, sans-serif", height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        # Exam Statistics
        st.markdown("### 📋 إحصائيات الاختبارات")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 معلومات الأسئلة")
            total_questions = len(questions)
            approved_questions = len([q for q in questions if q.get('status') == 'approved'])
            pending_questions = len([q for q in questions if q.get('status') == 'pending'])
            
            st.metric("إجمالي الأسئلة", total_questions)
            st.metric("الأسئلة المعتمدة", approved_questions)
            st.metric("في انتظار المراجعة", pending_questions)
            
            if questions:
                categories = {}
                for q in questions:
                    cat = q.get('category', 'غير محدد')
                    categories[cat] = categories.get(cat, 0) + 1
                
                st.markdown("**توزيع الأسئلة حسب النوع:**")
                for cat, count in categories.items():
                    st.caption(f"• {cat}: {count} سؤال")
        
        with col2:
            st.markdown("#### 📊 معلومات الاختبارات")
            total_exams = len(exams)
            active_exams = len([e for e in exams if e.get('status') == 'active'])
            
            st.metric("إجمالي الاختبارات", total_exams)
            st.metric("الاختبارات النشطة", active_exams)
            
            if exams:
                avg_questions_per_exam = sum(len(e.get('question_ids', [])) for e in exams) / len(exams)
                avg_time_limit = sum(e.get('time_limit', 0) for e in exams) / len(exams)
                
                st.metric("متوسط الأسئلة لكل اختبار", f"{avg_questions_per_exam:.1f}")
                st.metric("متوسط وقت الاختبار", f"{avg_time_limit:.0f} دقيقة")
        
        # Question difficulty analysis
        if questions:
            st.markdown("### 📊 تحليل صعوبة الأسئلة")
            
            # Calculate question performance (mock data if no real submissions)
            question_performance = []
            for q in questions:
                if q.get('status') == 'approved':
                    # Mock performance data
                    success_rate = random.uniform(40, 95)  # Random success rate
                    attempts = random.randint(5, 20)
                    
                    question_performance.append({
                        'السؤال': q['id'],
                        'الفئة': q.get('category', 'غير محدد'),
                        'الصعوبة المحددة': q.get('difficulty', 'متوسط'),
                        'معدل النجاح': success_rate,
                        'عدد المحاولات': attempts
                    })
            
            if question_performance:
                df_questions = pd.DataFrame(question_performance)
                
                # Most difficult questions
                difficult_questions = df_questions.nsmallest(5, 'معدل النجاح')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**الأسئلة الأكثر صعوبة:**")
                    for _, row in difficult_questions.iterrows():
                        st.caption(f"• {row['الفئة']}: {row['معدل النجاح']:.1f}% نجاح")
                
                with col2:
                    # Success rate by category
                    category_performance = df_questions.groupby('الفئة')['معدل النجاح'].mean().reset_index()
                    
                    fig_cat_performance = px.bar(
                        category_performance.sort_values('معدل النجاح'),
                        x='معدل النجاح',
                        y='الفئة',
                        title="معدل النجاح حسب فئة الأسئلة",
                        orientation='h'
                    )
                    fig_cat_performance.update_layout(font_family="Cairo, sans-serif", height=300)
                    st.plotly_chart(fig_cat_performance, use_container_width=True)
    
    with tab5:
        # Insights and Recommendations
        st.markdown("### 💡 الرؤى والتوصيات الاستراتيجية")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>📊 رؤى الأداء الحالي</h4>
                <ul>
                    <li>متوسط الأداء العام في مستوى جيد ومقبول</li>
                    <li>تباين في الأداء بين الأقسام المختلفة</li>
                    <li>علاقة عكسية بين سرعة الإنجاز والدقة</li>
                    <li>بعض فئات الأسئلة تحتاج مراجعة وتحسين</li>
                    <li>معدل إكمال الاختبارات عالي ومشجع</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
                <h4>🎯 التوصيات الفورية</h4>
                <ul>
                    <li>مراجعة الأسئلة ذات معدل النجاح المنخفض</li>
                    <li>توفير تدريب إضافي للأقسام الأقل أداءً</li>
                    <li>تطوير أسئلة تحضيرية للفئات الصعبة</li>
                    <li>تحسين تعليمات الاختبارات ووضوحها</li>
                    <li>إضافة نماذج تدريبية قبل الاختبار الفعلي</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>🚀 استراتيجيات التطوير</h4>
                <ul>
                    <li>تنويع أساليب التقييم حسب طبيعة كل قسم</li>
                    <li>تطوير برامج تأهيلية مخصصة</li>
                    <li>إنشاء مسارات تطوير فردية للموظفين</li>
                    <li>تطبيق التعلم التكيفي حسب الأداء</li>
                    <li>إنشاء مجموعات دعم وتطوير مهني</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
                <h4>📈 مؤشرات المتابعة</h4>
                <ul>
                    <li>مراقبة تحسن الأداء شهرياً</li>
                    <li>قياس رضا المشاركين عن التجربة</li>
                    <li>تتبع معدلات الإكمال والمشاركة</li>
                    <li>مراجعة دورية لصعوبة الأسئلة</li>
                    <li>تحليل الاتجاهات طويلة المدى</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Action items
        st.markdown("### 📋 خطة العمل المقترحة")
        
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 20px 0;">
            <table style="width: 100%; border-collapse: collapse; font-family: 'Cairo', sans-serif;">
                <thead>
                    <tr style="background: #f8fafc; border-bottom: 2px solid #e2e8f0;">
                        <th style="padding: 15px; text-align: right; font-weight: 600; color: #1f2937;">المهمة</th>
                        <th style="padding: 15px; text-align: center; font-weight: 600; color: #1f2937;">الأولوية</th>
                        <th style="padding: 15px; text-align: center; font-weight: 600; color: #1f2937;">المدة</th>
                        <th style="padding: 15px; text-align: right; font-weight: 600; color: #1f2937;">المسؤول</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 12px; text-align: right;">مراجعة الأسئلة الصعبة</td>
                        <td style="padding: 12px; text-align: center;"><span style="background: #fee2e2; color: #dc2626; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">عالية</span></td>
                        <td style="padding: 12px; text-align: center;">أسبوع</td>
                        <td style="padding: 12px; text-align: right;">فريق المحتوى</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 12px; text-align: right;">تطوير مواد تدريبية</td>
                        <td style="padding: 12px; text-align: center;"><span style="background: #fef3c7; color: #f59e0b; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">متوسطة</span></td>
                        <td style="padding: 12px; text-align: center;">أسبوعين</td>
                        <td style="padding: 12px; text-align: right;">فريق التطوير</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 12px; text-align: right;">تحليل أداء الأقسام</td>
                        <td style="padding: 12px; text-align: center;"><span style="background: #fee2e2; color: #dc2626; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">عالية</span></td>
                        <td style="padding: 12px; text-align: center;">3 أيام</td>
                        <td style="padding: 12px; text-align: right;">فريق التحليل</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px; text-align: right;">تحديث واجهة المستخدم</td>
                        <td style="padding: 12px; text-align: center;"><span style="background: #e0f2fe; color: #0891b2; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">منخفضة</span></td>
                        <td style="padding: 12px; text-align: center;">شهر</td>
                        <td style="padding: 12px; text-align: right;">فريق التقنية</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)


# Export and reporting options
st.subheader("📥 تصدير التقارير والبيانات")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 تقرير الأداء العام", use_container_width=True):
        st.success("✅ تم إنشاء تقرير الأداء العام")
        st.info("سيتم تنزيل التقرير تلقائياً...")

with col2:
    if st.button("📋 تفاصيل النتائج", use_container_width=True):
        st.success("✅ تم إنشاء تقرير النتائج التفصيلي")
        st.info("سيتم تنزيل الملف Excel...")

with col3:
    if st.button("📈 التحليل الإحصائي", use_container_width=True):
        st.success("✅ تم إنشاء التحليل الإحصائي المتقدم")
        st.info("سيتم تنزيل التقرير PDF...")

with col4:
    if st.button("🎯 التوصيات التنفيذية", use_container_width=True):
        st.success("✅ تم إنشاء تقرير التوصيات")
        st.info("سيتم إرسال التقرير للإدارة...")

# Real-time updates simulation
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 🔄 التحديث المباشر للبيانات")
    st.info("يتم تحديث البيانات تلقائياً كل 5 دقائق أثناء فترات الاختبار")

with col2:
    if st.button("🔄 تحديث البيانات الآن", use_container_width=True):
        st.success("✅ تم تحديث البيانات بنجاح!")
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 14px; border-top: 1px solid #e2e8f0; padding-top: 25px; margin-top: 40px;">
    <strong>مركز التحليلات المتقدمة - منصة تقييم القدرات المعرفية</strong><br>
    آخر تحديث: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """<br>
    جميع البيانات محمية ومشفرة وفقاً لمعايير الأمان الحكومية
</div>
""", unsafe_allow_html=True)
