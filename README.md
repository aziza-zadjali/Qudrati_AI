# منصة تقييم القدرات المعرفية - وزارة العمل 
# MOL Cognitive Assessment Platform - Professional Demo

## 🎯 نظرة عامة / Overview

هذا مشروع تجريبي متقدم لمنصة تقييم القدرات المعرفية المطورة خصيصاً لوزارة العمل في سلطنة عُمان. يتضمن المشروع نظاماً شاملاً لتوليد الأسئلة بالذكاء الاصطناعي، مراجعة الخبراء، تجميع الاختبارات، تقديم الاختبارات للطلاب، وتحليل النتائج.

This is an advanced demo project for a cognitive assessment platform specifically developed for the Ministry of Labor in the Sultanate of Oman. The project includes a comprehensive system for AI-powered question generation, expert review, exam assembly, student assessment delivery, and results analytics.

## 🚀 المزايا الرئيسية / Key Features

### ✨ توليد الأسئلة بالذكاء الاصطناعي / AI-Powered Question Generation
- **12 نوع سؤال مختلف** / 12 Different Question Types
- **دعم اللغة العربية الكامل** / Full Arabic Language Support  
- **تكامل مع OpenAI GPT-4** / OpenAI GPT-4 Integration
- **آليات احتياطية محلية** / Local Fallback Mechanisms

### 👨‍🏫 مراجعة الخبراء / Expert Review System
- **واجهة مراجعة تفاعلية** / Interactive Review Interface
- **إحصائيات الجودة** / Quality Statistics
- **تدفق الموافقة** / Approval Workflow
- **ملاحظات مفصلة** / Detailed Feedback

### 📋 تجميع الاختبارات / Exam Assembly
- **بناء آلي ويدوي** / Automatic & Manual Building
- **توزيع متوازن** / Balanced Distribution
- **إعدادات متقدمة** / Advanced Settings
- **معاينة شاملة** / Comprehensive Preview

### 📝 تقديم الاختبارات / Assessment Delivery
- **واجهة طالب احترافية** / Professional Student Interface
- **مؤقت متقدم** / Advanced Timer
- **حفظ تلقائي** / Auto-save Functionality
- **تنقل سهل** / Easy Navigation

### 📊 تحليلات متقدمة / Advanced Analytics
- **إحصائيات شاملة** / Comprehensive Statistics
- **رسوم بيانية تفاعلية** / Interactive Charts
- **تقارير تفصيلية** / Detailed Reports
- **رؤى استراتيجية** / Strategic Insights

## 🛠️ التثبيت والإعداد / Installation & Setup

### المتطلبات / Requirements
- Python 3.8+
- Streamlit 1.28+
- OpenAI API Key (اختياري / Optional)

### خطوات التثبيت / Installation Steps

1. **استنساخ المشروع / Clone the Project**


## 🎓 دليل الاستخدام / Usage Guide

### 1. توليد الأسئلة / Question Generation
- انتقل لصفحة "توليد الأسئلة العربية"
- اختر نوع السؤال والصعوبة
- اضغط "توليد السؤال"
- راجع النتيجة واحفظها

### 2. مراجعة الخبراء / Expert Review  
- انتقل لصفحة "مراجعة الخبراء"
- راجع الأسئلة في قائمة الانتظار
- اعتمد أو ارفض الأسئلة
- أضف ملاحظات إضافية

### 3. تجميع الاختبار / Exam Assembly
- انتقل لصفحة "تجميع الاختبارات"
- أدخل معلومات الاختبار
- اختر الأسئلة يدوياً أو آلياً
- راجع واحفظ الاختبار

### 4. تقديم الاختبار / Assessment Delivery
- انتقل لصفحة "تقديم الاختبار"
- أدخل بيانات الطالب (محاكاة)
- اختر الاختبار وابدأ
- أجب على الأسئلة وسلّم

### 5. عرض التحليلات / Analytics Review
- انتقل لصفحة "التحليلات"
- راجع الإحصائيات والرسوم البيانية
- اطلع على الرؤى والتوصيات
- صدّر التقارير

## 🔧 التكوين المتقدم / Advanced Configuration

### إعدادات OpenAI / OpenAI Settings

### إعدادات قاعدة البيانات / Database Settings
- يستخدم المشروع ملف JSON محلي للبيانات
- يمكن التكامل مع قواعد بيانات أخرى
- النسخ الاحتياطي تلقائي

### إعدادات الأمان / Security Settings
- جميع البيانات محلية
- تشفير البيانات الحساسة
- تسجيل شامل للأنشطة

## 🧪 بيانات تجريبية / Demo Data

لتجربة النظام بالكامل:

1. **توليد أسئلة تجريبية** / Generate Demo Questions
2. **اعتماد الأسئلة** / Approve Questions  
3. **إنشاء اختبار** / Create Exam
4. **تجربة الاختبار** / Take Exam
5. **عرض التحليلات** / View Analytics

أو استخدم الزر "إنشاء بيانات تجريبية" في صفحة التحليلات.

## 🚀 النشر للإنتاج / Production Deployment

### متطلبات الخادم / Server Requirements
- **المعالج / CPU**: 4+ cores
- **الذاكرة / RAM**: 8GB+  
- **التخزين / Storage**: 100GB+
- **الشبكة / Network**: 100Mbps+

### خطوات النشر / Deployment Steps
1. إعداد خادم Ubuntu 22.04 LTS
2. تثبيت Python 3.11 و dependencies
3. إعداد Nginx كـ reverse proxy  
4. تكوين SSL/TLS certificates
5. إعداد قاعدة بيانات PostgreSQL
6. تفعيل المراقبة والنسخ الاحتياطي

## 🔒 الأمان / Security

- **تشفير البيانات** / Data Encryption: AES-256
- **الشبكة** / Network: TLS 1.3
- **التوثيق** / Authentication: تكامل مع Active Directory
- **التدقيق** / Auditing: تسجيل شامل للأنشطة
- **النسخ الاحتياطي** / Backup: آلي يومي

## 📞 الدعم / Support

- **البريد الإلكتروني / Email**: support@mol.gov.om
- **الهاتف / Phone**: +968-123-456-7890
- **ساعات العمل / Hours**: 8:00-17:00 (أحد-خميس / Sun-Thu)

## 📄 الترخيص / License

هذا المشروع مطور خصيصاً لوزارة العمل في سلطنة عُمان. جميع الحقوق محفوظة.

This project is developed specifically for the Ministry of Labor in the Sultanate of Oman. All rights reserved.

---

## 🎯 للعقد / For Contract

هذا العرض التوضيحي يُظهر:

This demonstration showcases:

✅ **التقنيات المتقدمة** / Advanced Technologies
✅ **التصميم العربي الأصيل** / Authentic Arabic Design  
✅ **التكامل الكامل** / Complete Integration
✅ **الجودة العالية** / High Quality
✅ **القابلية للتوسع** / Scalability
✅ **الأمان المتقدم** / Advanced Security

**جاهز للنشر في بيئة الإنتاج / Ready for Production Deployment**

---

*تم التطوير بواسطة فريق تقني متخصص لعقد وزارة العمل*  
*Developed by specialized technical team for MOL contract*
