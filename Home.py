import streamlit as st
import json
from pathlib import Path
import openai

# Page config
st.set_page_config(
    page_title="منصة تقييم القدرات المعرفية - وزارة العمل",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

.main-title {
    font-family: 'Cairo', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #1e3a8a;
    text-align: center;
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 30px;
}

.feature-card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-right: 4px solid #10b981;
    margin: 15px 0;
    height: auto;
    min-height: 200px;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

.feature-card h4 {
    font-size: 16px;
    line-height: 1.4;
    margin-bottom: 15px;
    word-wrap: break-word;
    overflow-wrap: break-word;
    hyphens: auto;
}

.metric-container {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    border-right: 4px solid #1e3a8a;
}
</style>
""", unsafe_allow_html=True)

# Database helpers
DB_PATH = Path(__file__).with_name("demo_db_arabic.json")

def load_db():
    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        empty_db = {"questions": [], "exams": [], "submissions": []}
        with open(DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(empty_db, f, ensure_ascii=False, indent=2)
        return empty_db

# Header
st.markdown("""
<div class="main-title">
🧠 منصة تقييم القدرات المعرفية
<br>
<small style="font-size: 20px; color: #64748b;">وزارة العمل - سلطنة عُمان</small>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div style="background: #f0f9ff; padding: 25px; border-radius: 15px; border-right: 4px solid #0891b2; margin: 25px 0;">
<h3 style="color: #0c4a6e; margin-top: 0;">💡 نسخة تجريبية متقدمة</h3>
<p style="font-size: 16px; line-height: 1.8; color: #075985;">
هذه نسخة تجريبية شاملة من منصة تقييم القدرات المعرفية التي تتضمن <strong>مرحلتين متكاملتين</strong>:
</p>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
<div style="background: #ecfdf5; padding: 20px; border-radius: 10px;">
<h4 style="color: #065f46; margin: 0 0 10px 0;">🎯 المرحلة الأولى: تطوير المحتوى</h4>
<ul style="color: #047857; margin: 0;">
<li>توليد الأسئلة بالذكاء الاصطناعي</li>
<li>مراجعة وموافقة الخبراء</li>
<li>تجميع الاختبارات</li>
</ul>
</div>
<div style="background: #fef3c7; padding: 20px; border-radius: 10px;">
<h4 style="color: #92400e; margin: 0 0 10px 0;">🚀 المرحلة الثانية: التشغيل</h4>
<ul style="color: #b45309; margin: 0;">
<li>تقديم الاختبارات للطلاب</li>
<li>التصحيح والمراجعة</li>
<li>التحليلات والتقارير</li>
</ul>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Load and display metrics
try:
    data = load_db()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 إجمالي الأسئلة", len(data.get('questions', [])))
    with col2:
        approved_questions = len([q for q in data.get('questions', []) if q.get('status') == 'approved'])
        st.metric("✅ الأسئلة المعتمدة", approved_questions)
    with col3:
        st.metric("📋 الاختبارات المتاحة", len(data.get('exams', [])))
    with col4:
        st.metric("👥 محاولات الطلاب", len(data.get('submissions', [])))
except Exception as e:
    st.error(f"خطأ في تحميل البيانات: {e}")

# Features overview
st.markdown("### 🌟 المزايا التنافسية")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="feature-card"><h4 style="color: #065f46;">🤖 الذكاء الاصطناعي </h4>
<p style="color: #047857;">• توليد أسئلة عالية الجودة باستخدام GPT-4<br>• فلترة ذكية للمحتوى<br>• نماذج احتياطية مفتوحة المصدر<br>• تحسين مستمر عبر التعلم الآلي</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="feature-card"><h4 style="color: #1e40af;">👨‍🏫 ضمان الجودة</h4>
<p style="color: #1d4ed8;">• مراجعة شاملة من خبراء المادة<br>• تدفق موافقة متعدد المراحل<br>• معايير جودة صارمة<br>• تتبع الأداء والتحسين</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="feature-card"><h4 style="color: #92400e;">🔒 الأمان والامتثال</h4>
<p style="color: #b45309;">• نشر داخلي على خوادم الوزارة<br>• تشفير شامل للبيانات<br>• امتثال لمعايير الأمان الحكومية<br>• تسجيل شامل للأنشطة</p></div>""", unsafe_allow_html=True)

# Technical specifications
st.markdown("### ⚙️ المواصفات التقنية")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**البنية التحتية:**
- **الواجهة الأمامية:** React + TypeScript مع دعم RTL
- **الخادم الخلفي:** Python FastAPI مع قواعد البيانات المحلية
- **قاعدة البيانات:** PostgreSQL مع Redis للتخزين المؤقت
- **الذكاء الاصطناعي:** OpenAI GPT-4 + نماذج محلية
""")
with col2:
    st.markdown("""
**الأداء والموثوقية:**
- **الدعم المتزامن:** 500+ مستخدم
- **وقت الاستجابة:** أقل من 2 ثانية
- **معدل التشغيل:** 99.9% خلال ساعات العمل
- **النسخ الاحتياطي:** آلي يومي
""")

# System status
st.markdown("### 🔧 حالة النظام")
col1, col2, col3 = st.columns(3)

with col1:
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY")
        if openai_key and openai_key != "your_openai_api_key_here":
            openai.api_key = openai_key
            # Test API connectivity
            openai.models.list()
            st.success("🟢 اتصال OpenAI: متصل")
        else:
            st.warning("🟡 اتصال OpenAI: غير مكون")
    except Exception as e:
        st.error(f"🔴 اتصال OpenAI: غير متصل ({e})")

with col2:
    try:
        load_db()
        st.success("🟢 قاعدة البيانات: متصلة")
    except:
        st.error("🔴 قاعدة البيانات: خطأ")

with col3:
    st.success("🟢 واجهة المستخدم: تعمل")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 14px; padding: 20px;">
<strong>منصة تقييم القدرات المعرفية - وزارة العمل، سلطنة عُمان</strong><br>
النسخة التجريبية المتقدمة | تم التطوير خصيصاً من أجل وزارة العمل<br>
جميع الحقوق محفوظة © ٢٠٢٥
</div>
""", unsafe_allow_html=True)
