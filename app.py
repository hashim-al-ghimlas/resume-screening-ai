import streamlit as st
import joblib
import re

st.set_page_config(page_title="Resume Screening AI", page_icon="📄", layout="centered")

@st.cache_resource
def load_models():
    m = 'trained_models'
    model   = joblib.load(f'{m}/xgboost_model.joblib')
    tfidf   = joblib.load(f'{m}/tfidf_vectorizer.joblib')
    encoder = joblib.load(f'{m}/label_encoder.joblib')
    return model, tfidf, encoder

def clean_text(text):
    text = re.sub(r'http\S+\s*', ' ', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).lower().strip()

def extract_info(text):
    info = {}
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    info['email'] = emails[0] if emails else None
    phones = re.findall(r'[\+\(]?[1-9][0-9\s\.\-\(\)]{7,}[0-9]', text)
    info['phone'] = phones[0].strip() if phones else None
    skills_list = [
        'python','java','sql','javascript','r','c++','scala',
        'machine learning','deep learning','nlp','tensorflow','pytorch',
        'keras','scikit-learn','pandas','numpy','matplotlib','seaborn',
        'tableau','power bi','excel','aws','azure','gcp','docker',
        'kubernetes','git','linux','hadoop','spark','mongodb',
        'mysql','postgresql','react','node','html','css',
        'data analysis','data science','statistics','visualization'
    ]
    text_lower = text.lower()
    info['skills'] = [s for s in skills_list if s in text_lower][:12]
    exp_patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'(\d+)\+?\s*years?\s*experience',
        r'experience\s*of\s*(\d+)\+?\s*years?',
    ]
    info['experience'] = None
    for p in exp_patterns:
        m = re.search(p, text_lower)
        if m:
            info['experience'] = m.group(1) + ' years'
            break
    edu_keywords = {
        'PhD':      ['phd','ph.d','doctorate','doctor of'],
        'Master':   ["master's",'masters','m.sc','msc','mba','m.eng'],
        'Bachelor': ["bachelor's",'bachelors','b.sc','bsc','b.eng','b.tech'],
        'Diploma':  ['diploma','associate'],
    }
    info['education'] = None
    for level, keywords in edu_keywords.items():
        if any(kw in text_lower for kw in keywords):
            info['education'] = level
            break
    lang_list = ['english','arabic','french','german','spanish',
                 'chinese','japanese','hindi','portuguese']
    info['languages'] = [l.capitalize() for l in lang_list if l in text_lower]
    return info

def classify(text, model, tfidf, encoder):
    vec      = tfidf.transform([clean_text(text)])
    pred_enc = model.predict(vec)[0]
    proba    = model.predict_proba(vec)[0]
    category = encoder.inverse_transform([pred_enc])[0]
    confidence = proba.max() * 100
    top3_idx = proba.argsort()[-3:][::-1]
    top3 = [(encoder.classes_[i], round(proba[i]*100,1)) for i in top3_idx]
    return category, confidence, top3

def show_results(text):
    with st.spinner("جاري التحليل..."):
        category, confidence, top3 = classify(text, model, tfidf, encoder)
        info = extract_info(text)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**التصنيف: {category}**")
    with col2:
        st.metric("الثقة", f"{confidence:.1f}%")

    st.divider()
    st.subheader("🔢 أعلى 3 فئات محتملة")
    medals = ['🥇','🥈','🥉']
    for i, (cat, prob) in enumerate(top3):
        st.progress(prob/100, text=f"{medals[i]} {cat} — {prob}%")

    st.divider()
    st.subheader("📋 معلومات مستخرجة من السيرة الذاتية")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📧 البريد الإلكتروني**")
        st.write(info['email'] or "غير موجود")
    with c2:
        st.markdown("**📞 رقم الهاتف**")
        st.write(info['phone'] or "غير موجود")
    with c3:
        st.markdown("**🎓 المستوى التعليمي**")
        st.write(info['education'] or "غير محدد")

    c4, c5 = st.columns(2)
    with c4:
        st.markdown("**💼 سنوات الخبرة**")
        st.write(info['experience'] or "غير محدد")
    with c5:
        st.markdown("**🌍 اللغات**")
        st.write(", ".join(info['languages']) if info['languages'] else "غير محددة")

    st.divider()
    st.subheader("🛠️ المهارات المكتشفة")
    if info['skills']:
        skills_html = " ".join([
            f"<span style='background:#E6F1FB;color:#0C447C;padding:5px 12px;"
            f"border-radius:15px;margin:3px;display:inline-block;"
            f"font-size:13px;font-weight:500'>{s.title()}</span>"
            for s in info['skills']
        ])
        st.markdown(skills_html, unsafe_allow_html=True)
    else:
        st.info("لم يتم اكتشاف مهارات محددة")

    st.divider()
    st.subheader("📊 ملخص")
    m1, m2, m3 = st.columns(3)
    m1.metric("عدد الكلمات",        len(text.split()))
    m2.metric("المهارات المكتشفة",  len(info['skills']))
    m3.metric("اللغات",             len(info['languages']))


model, tfidf, encoder = load_models()

st.title("📄 Resume Screening AI")
st.markdown("ارفع سيرة ذاتية أو الصق النص لمعرفة التصنيف والتفاصيل الكاملة")
st.divider()

tab1, tab2 = st.tabs(["📋 لصق النص", "📁 رفع ملف PDF"])

with tab1:
    text = st.text_area("الصق نص السيرة الذاتية هنا", height=220)
    if st.button("تصنيف وتحليل", key="t1", type="primary"):
        if text.strip():
            show_results(text)
        else:
            st.warning("الرجاء إدخال نص")

with tab2:
    file = st.file_uploader("ارفع ملف PDF", type=['pdf'])
    if file and st.button("تصنيف وتحليل", key="t2", type="primary"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            text   = " ".join(p.extract_text() or "" for p in reader.pages)
            if text.strip():
                show_results(text)
            else:
                st.error("لم يتم استخراج نص من الملف")
        except Exception as e:
            st.error(f"تعذّر قراءة الملف: {e}")
