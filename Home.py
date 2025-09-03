import streamlit as st
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="منصة تقييم القدرات المعرفية - وزارة العمل",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Arabic RTL styling
st.markdown(""" """, unsafe_allow_html=True)

# Database helpers
DB_PATH = Path(__file__).with_name("demo_db_arabic.json")

def load_db():
    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Initialize empty database
        empty_db = {
            "questions": [],
            "exams": [],
            "submissions": []
        }
        with open(DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(empty_db, f, ensure_ascii=False, indent=2)
        return empty_db

# Header
st.markdown("""
🧠 منصة تقييم القدرات المعرفية  
وزارة العمل - سلطنة عُمان
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
#### 💡 نسخة تجريبية متقدمة

هذه نسخة تجريبية شاملة من منصة تقييم القدرات المعرفية التي تتضمن مرحلتين متكاملتين:  
##### 🎯 المرحلة الأولى: تطوير المحتوى
- توليد الأسئلة بالذكاء الاصطناعي
- مراجعة وموافقة الخبراء
- تجميع الاختبارات
##### 🚀 المرحلة الثانية: التشغيل
- تقديم الاختبارات للطلاب
- التصحيح والمراجعة
- التحليلات والتقارير
""", unsafe_allow_html=True)

# Load and display metrics
try:
    data = load_db()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_questions = len(data.get('questions', []))
        st.metric(label="📝 إجمالي الأسئلة", value=total_questions, help="العدد الإجمالي للأسئلة في النظام")
    with col2:
        approved_questions = len([q for q in data.get('questions', []) if q.get('status') == 'approved'])
        st.metric(label="✅ الأسئلة المعتمدة", value=approved_questions, help="الأسئلة التي تمت الموافقة عليها")
    with col3:
        total_exams = len(data.get('exams', []))
        st.metric(label="📋 الاختبارات المتاحة", value=total_exams, help="عدد الاختبارات الجاهزة")
    with col4:
        total_submissions = len(data.get('submissions', []))
        st.metric(label="👥 محاولات الطلاب", value=total_submissions, help="عدد المحاولات المسجلة")
except Exception as e:
    st.error(f"خطأ في تحميل البيانات: {e}")

# Features overview
st.markdown("### 🌟 المزايا التنافسية")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
##### 🤖 الذكاء الاصطناعي
• توليد أسئلة عالية الجودة باستخدام GPT-4  
• فلترة ذكية للمحتوى  
• نماذج احتياطية مفتوحة المصدر  
• تحسين مستمر عبر التعلم الآلي
""", unsafe_allow_html=True)
with col2:
    st.markdown("""
##### 👨‍🏫 ضمان الجودة
• مراجعة شاملة من خبراء المادة  
• تدفق موافقة متعدد المراحل  
• معايير جودة صارمة  
• تتبع الأداء والتحسين
""", unsafe_allow_html=True)
with col3:
    st.markdown("""
##### 🔒 الأمان والامتثال
• نشر داخلي على خوادم الوزارة  
• تشفير شامل للبيانات  
• امتثال لمعايير الأمان الحكومية  
• تسجيل شامل للأنشطة
""", unsafe_allow_html=True)

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

# Navigation guide
st.markdown("### 📱 دليل الاستخدام")
st.markdown("""
للاستفادة الكاملة من النسخة التجريبية:  
1️⃣ توليد الأسئلة  
استخدم الذكاء الاصطناعي لإنشاء أسئلة بـ 12 نوعاً مختلفاً  
2️⃣ مراجعة الخبراء  
راجع واعتمد الأسئلة المولدة قبل الاستخدام  
3️⃣ تجميع الاختبارات  
أنشئ اختبارات مخصصة من الأسئلة المعتمدة  
4️⃣ تقديم الاختبار  
جرب تجربة الطالب الكاملة مع التوقيت  
5️⃣ التحليلات  
استعرض التقارير والإحصائيات التفصيلية  
💡 نصيحة: ابدأ بتوليد بعض الأسئلة، ثم اعتمدها، وأنشئ اختباراً لرؤية التدفق الكامل في العمل.
""", unsafe_allow_html=True)

# System status
st.markdown("### 🔧 حالة النظام")
col1, col2, col3 = st.columns(3)

with col1:
    # Improved OpenAI connectivity check
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY")
        if openai_key and openai_key.strip() and openai_key != "your_openai_api_key_here":
            masked_key = "****" + openai_key[-4:]
            st.success(f"🟢 اتصال OpenAI: مفتاح مكوَّن ({masked_key})")
        else:
            st.warning("🟡 اتصال OpenAI: غير مكوَّن")
    except Exception as e:
        st.error(f"🔴 اتصال OpenAI: خطأ ({e})")

with col2:
    # Check database
    try:
        data = load_db()
        st.success("🟢 قاعدة البيانات: متصلة")
    except:
        st.error("🔴 قاعدة البيانات: خطأ")

with col3:
    # Check UI
    st.success("🟢 واجهة المستخدم: تعمل")

# Footer
st.markdown("---")
st.markdown("""
منصة تقييم القدرات المعرفية - وزارة العمل، سلطنة عُمان  
النسخة التجريبية المتقدمة  
تم التطوير خصيصاً من أجل وزارة العمل  
جميع الحقوق محفوظة © ٢٠٢٥
""", unsafe_allow_html=True)
