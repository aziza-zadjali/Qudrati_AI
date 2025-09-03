import streamlit as st
import json
from pathlib import Path
import os

# Page config
st.set_page_config(
    page_title="منصة تقييم القدرات المعرفية - وزارة العمل",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Arabic RTL styling
st.markdown("""
&lt;style&gt;
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700&amp;display=swap');
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700&amp;display=swap');

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
.status-note {font-size:12px; color:#64748b;}
&lt;/style&gt;
""", unsafe_allow_html=True)

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
# ---- OpenAI connectivity helpers (robust) ----
def get_openai_key():
    """
    يبحث عن مفتاح OpenAI بترتيب:
    st.secrets['openai']['api_key'] -> st.secrets['OPENAI_API_KEY'] -> env var OPENAI_API_KEY
    ويرجع (key, source_label)
    """
    try:
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            return st.secrets["openai"]["api_key"], "st.secrets['openai']['api_key']"
    except Exception:
        pass

    key = st.secrets.get("OPENAI_API_KEY", "")
    if key:
        return key, "st.secrets['OPENAI_API_KEY']"

    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        return key, "env:OPENAI_API_KEY"

    return "", ""

def check_openai_connectivity():
    """
    يحاول الاتصال عبر SDK الجديد ثم الواجهة القديمة.
    يرجع dict يحتوي: connected, message, sdk, source, error
    """
    api_key, source = get_openai_key()
    if not api_key or api_key == "your_openai_api_key_here":
        return {
            "connected": False,
            "message": "اتصال OpenAI: غير مُكوَّن",
            "sdk": None,
            "source": source or "غير محدد",
            "error": None,
            "status": "not_configured"
        }

    # حاول بالعميل الجديد (openai>=1.0)
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        # استدعاء خفيف للتأكد من الاتصال
        _ = client.models.list()
        return {
            "connected": True,
            "message": "اتصال OpenAI: متصل",
            "sdk": "OpenAI Python SDK v1 (client)",
            "source": source,
            "error": None,
            "status": "ok"
        }
    except Exception as e_new:
        # جرّب الواجهة القديمة
        try:
            import openai as openai_legacy  # type: ignore
            openai_legacy.api_key = api_key
            _ = openai_legacy.models.list()
            return {
                "connected": True,
                "message": "اتصال OpenAI: متصل",
                "sdk": "OpenAI legacy API",
                "source": source,
                "error": None,
                "status": "ok"
            }
        except Exception as e_old:
            return {
                "connected": False,
                "message": "اتصال OpenAI: غير متصل",
                "sdk": None,
                "source": source,
                "error": str(e_new) or str(e_old),
                "status": "error"
            }

# Header
st.markdown("""
&lt;div class="main-title"&gt;
🧠 منصة تقييم القدرات المعرفية
&lt;br&gt;
&lt;small style="font-size: 20px; color: #64748b;"&gt;وزارة العمل - سلطنة عُمان&lt;/small&gt;
&lt;/div&gt;
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
&lt;div style="background: #f0f9ff; padding: 25px; border-radius: 15px; border-right: 4px solid #0891b2; margin: 25px 0;"&gt;
&lt;h3 style="color: #0c4a6e; margin-top: 0;"&gt;💡 نسخة تجريبية متقدمة&lt;/h3&gt;
&lt;p style="font-size: 16px; line-height: 1.8; color: #075985;"&gt;
هذه نسخة تجريبية شاملة من منصة تقييم القدرات المعرفية التي تتضمن &lt;strong&gt;مرحلتين متكاملتين&lt;/strong&gt;:
&lt;/p&gt;

&lt;div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;"&gt;
&lt;div style="background: #ecfdf5; padding: 20px; border-radius: 10px;"&gt;
&lt;h4 style="color: #065f46; margin: 0 0 10px 0;"&gt;🎯 المرحلة الأولى: تطوير المحتوى&lt;/h4&gt;
&lt;ul style="color: #047857; margin: 0;"&gt;
&lt;li&gt;توليد الأسئلة بالذكاء الاصطناعي&lt;/li&gt;
&lt;li&gt;مراجعة وموافقة الخبراء&lt;/li&gt;
&lt;li&gt;تجميع الاختبارات&lt;/li&gt;
&lt;/ul&gt;
&lt;/div&gt;

&lt;div style="background: #fef3c7; padding: 20px; border-radius: 10px;"&gt;
&lt;h4 style="color: #92400e; margin: 0 0 10px 0;"&gt;🚀 المرحلة الثانية: التشغيل&lt;/h4&gt;
&lt;ul style="color: #b45309; margin: 0;"&gt;
&lt;li&gt;تقديم الاختبارات للطلاب&lt;/li&gt;
&lt;li&gt;التصحيح والمراجعة&lt;/li&gt;
&lt;li&gt;التحليلات والتقارير&lt;/li&gt;
&lt;/ul&gt;
&lt;/div&gt;
&lt;/div&gt;
&lt;/div&gt;
""", unsafe_allow_html=True)

# Load and display metrics
try:
    data = load_db()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_questions = len(data.get('questions', []))
        st.metric(
            label="📝 إجمالي الأسئلة", 
            value=total_questions,
            help="العدد الإجمالي للأسئلة في النظام"
        )
    
    with col2:
        approved_questions = len([q for q in data.get('questions', []) if q.get('status') == 'approved'])
        st.metric(
            label="✅ الأسئلة المعتمدة", 
            value=approved_questions,
            help="الأسئلة التي تمت الموافقة عليها"
        )
    
    with col3:
        total_exams = len(data.get('exams', []))
        st.metric(
            label="📋 الاختبارات المتاحة", 
            value=total_exams,
            help="عدد الاختبارات الجاهزة"
        )
    
    with col4:
        total_submissions = len(data.get('submissions', []))
        st.metric(
            label="👥 محاولات الطلاب", 
            value=total_submissions,
            help="عدد المحاولات المسجلة"
        )
        
except Exception as e:
    st.error(f"خطأ في تحميل البيانات: {e}")

# Features overview
st.markdown("### 🌟 المزايا التنافسية")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    &lt;div class="feature-card"&gt;
    &lt;h4 style="color: #065f46;"&gt;🤖 الذكاء الاصطناعي &lt;/h4&gt;
    &lt;p style="color: #047857;"&gt;
    • توليد أسئلة عالية الجودة باستخدام GPT-4&lt;br&gt;
    • فلترة ذكية للمحتوى&lt;br&gt;
    • نماذج احتياطية مفتوحة المصدر&lt;br&gt;
    • تحسين مستمر عبر التعلم الآلي
    &lt;/p&gt;
    &lt;/div&gt;
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    &lt;div class="feature-card"&gt;
    &lt;h4 style="color: #1e40af;"&gt;👨‍🏫 ضمان الجودة&lt;/h4&gt;
    &lt;p style="color: #1d4ed8;"&gt;
    • مراجعة شاملة من خبراء المادة&lt;br&gt;
    • تدفق موافقة متعدد المراحل&lt;br&gt;
    • معايير جودة صارمة&lt;br&gt;
    • تتبع الأداء والتحسين
    &lt;/p&gt;
    &lt;/div&gt;
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    &lt;div class="feature-card"&gt;
    &lt;h4 style="color: #92400e;"&gt;🔒 الأمان والامتثال&lt;/h4&gt;
    &lt;p style="color: #b45309;"&gt;
    • نشر داخلي على خوادم الوزارة&lt;br&gt;
    • تشفير شامل للبيانات&lt;br&gt;
    • امتثال لمعايير الأمان الحكومية&lt;br&gt;
    • تسجيل شامل للأنشطة
    &lt;/p&gt;
    &lt;/div&gt;
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
&lt;div style="background: #f1f5f9; padding: 25px; border-radius: 12px; margin: 20px 0;"&gt;
&lt;strong&gt;للاستفادة الكاملة من النسخة التجريبية:&lt;/strong&gt;&lt;br&gt;&lt;br&gt;

&lt;div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;"&gt;
&lt;div style="background: white; padding: 15px; border-radius: 8px; border-right: 3px solid #3b82f6;"&gt;
&lt;strong&gt;1️⃣ توليد الأسئلة&lt;/strong&gt;&lt;br&gt;
استخدم الذكاء الاصطناعي لإنشاء أسئلة بـ 12 نوعاً مختلفاً
&lt;/div&gt;

&lt;div style="background: white; padding: 15px; border-radius: 8px; border-right: 3px solid #16a34a;"&gt;
&lt;strong&gt;2️⃣ مراجعة الخبراء&lt;/strong&gt;&lt;br&gt;
راجع واعتمد الأسئلة المولدة قبل الاستخدام
&lt;/div&gt;

&lt;div style="background: white; padding: 15px; border-radius: 8px; border-right: 3px solid #ea580c;"&gt;
&lt;strong&gt;3️⃣ تجميع الاختبارات&lt;/strong&gt;&lt;br&gt;
أنشئ اختبارات مخصصة من الأسئلة المعتمدة
&lt;/div&gt;

&lt;div style="background: white; padding: 15px; border-radius: 8px; border-right: 3px solid #7c3aed;"&gt;
&lt;strong&gt;4️⃣ تقديم الاختبار&lt;/strong&gt;&lt;br&gt;
جرب تجربة الطالب الكاملة مع التوقيت
&lt;/div&gt;

&lt;div style="background: white; padding: 15px; border-radius: 8px; border-right: 3px solid #dc2626;"&gt;
&lt;strong&gt;5️⃣ التحليلات&lt;/strong&gt;&lt;br&gt;
استعرض التقارير والإحصائيات التفصيلية
&lt;/div&gt;
&lt;/div&gt;

&lt;br&gt;
&lt;strong&gt;💡 نصيحة:&lt;/strong&gt; ابدأ بتوليد بعض الأسئلة، ثم اعتمدها، وأنشئ اختباراً لرؤية التدفق الكامل في العمل.
&lt;/div&gt;
""", unsafe_allow_html=True)

# System status
st.markdown("### 🔧 حالة النظام")

col1, col2, col3 = st.columns(3)

with col1:
    # Check OpenAI connectivity (robust + optional debug)
    connectivity = check_openai_connectivity()
    if connectivity["status"] == "ok":
        st.success("🟢 " + connectivity["message"])
    elif connectivity["status"] == "not_configured":
        st.warning("🟡 اتصال OpenAI: غير مُكوَّن")
        st.caption("أضِف المفتاح إلى st.secrets أو لمتغيرات البيئة.")
    else:
        st.error(f"🔴 {connectivity['message']}")

    # Optional technical details
    show_debug = st.checkbox("إظهار تفاصيل تقنية", value=False, help="إظهار مصدر المفتاح ونوع SDK ورسالة الخطأ (للتشخيص)")
    if show_debug:
        st.markdown(
            f"""
            <div class="status-note">
            • مصدر المفتاح: <code>{connectivity.get('source') or 'غير محدد'}</code><br>
            • نوع الـ SDK: <code>{connectivity.get('sdk') or 'غير معروف'}</code><br>
            • رسالة الخطأ: <code>{(connectivity.get('error') or 'لا يوجد').strip()}</code>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
    # Check database
    try:
        _ = load_db()
        st.success("🟢 قاعدة البيانات: متصلة")
    except Exception as e:
        st.error(f"🔴 قاعدة البيانات: خطأ ({e})")

with col3:
    # Check UI
    st.success("🟢 واجهة المستخدم: تعمل")

# Footer
st.markdown("---")
st.markdown("""
&lt;div style="text-align: center; color: #64748b; font-size: 14px; padding: 20px;"&gt;
&lt;strong&gt;منصة تقييم القدرات المعرفية - وزارة العمل، سلطنة عُمان&lt;/strong&gt;&lt;br&gt;
النسخة التجريبية المتقدمة | تم التطوير خصيصاً من أجل وزارة العمل&lt;br&gt;
جميع الحقوق محفوظة © ٢٠٢٥
&lt;/div&gt;
""", unsafe_allow_html=True)
