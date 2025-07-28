import streamlit as st
import joblib
import re
from underthesea import word_tokenize, text_normalize

try:
    model = joblib.load('App/svm_model_2_nhan.pkl')
    tfidf_vectorizer = joblib.load('App/vectorizer_2_nhan.pkl')
except FileNotFoundError:
    st.error("Không tìm thấy file model/vectorizer. Vui lòng tạo lại .pkl trước khi chạy.")
    st.stop()

st.set_page_config(
    page_title="Phân tích cảm xúc",
    layout="wide"
)

st.markdown("""
    <style>
    .banner {
        background-color: #00A0FF;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .result-card {
        background-color: #f9fafb;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='banner'>Big Data</div>", unsafe_allow_html=True)
    st.markdown("""
        Ứng dụng phân tích cảm xúc bình luận của khách hàng trên trang thương mại điện tử Tiki  
        **Mô hình sử dụng: SVM ( 2 nhãn)**  
        **Author:** Nhóm 9   
    """)
    st.markdown("---")
    st.info("Nhập bình luận ở phần chính, nhấn **Phân tích** để xem kết quả.")

short_word_dict = {
    "ko": "không", "kg": "không", "khong": "không", "k": "không", "kh": "không", "cx": "cũng",
    "mik": "mình", "mn": "mọi người", "bt": "bình thường", "nv": "nhân viên", "sp": "sản phẩm",
    "đc": "được", "dc": "được", "đk": "điều khoản", "đt": "điện thoại", "j": "gì", "vs": "với",
    "hok": "không", "lun": "luôn", "z": "gì", "zậy": "gì vậy", "thik": "thích", "hum": "hôm",
    "wa": "qua", "m": "mình", "mk": "mình", "bn": "bạn", "ok": "ổn", "t":  "tôi", "e": "em",
    "a": "anh", "bc": "bác", "ad": "admin", "mod": "moderator", "ib": "inbox", "inb": "inbox",
    "stt": "status", "tl": "trả lời", "tks": "cảm ơn", "thx": "cảm ơn", "thank": "cảm ơn",
    "cn": "công nghệ", "tech": "công nghệ", "sv": "sinh viên", "nt": "nhắn tin", "tn": "tin nhắn",
    "mxh": "mạng xã hội", "cf": "cà phê", "bp": "bộ phận", "pk": "phụ kiện", "tb": "thông báo",
    "hp": "hạnh phúc", "cv": "công việc", "ck": "chuyển khoản", "vk": "vợ", "ckho": "chồng",
    "sk": "sức khỏe", "gw": "gửi", "vl": "vãi l*n", "vkl": "vãi cả l*n", "sml": "sấp mặt luôn"
}

emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
    u"\U0001F926-\U0001F937" u"\U00010000-\U0010FFFF" u"\u200d" u"\u2640-\u2642"
    u"\u2600-\u2B55" u"\u23cf" u"\u23e9" u"\u231a" u"\u3030" u"\ufe0f"
    "]+", flags=re.UNICODE
)

def clean_data(text):
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = emoji_pattern.sub("", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text_normalize(text)
    words = [short_word_dict.get(w, w) for w in text.split()]
    return word_tokenize(" ".join(words), format="text")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Nhập bình luận")
    with st.form(key="sentiment_form"):
        user_input = st.text_area(
            label="Bình luận của bạn",
            height=180,
            placeholder="Ví dụ: Sản phẩm tuyệt vời, giao nhanh..."
        )
        submit = st.form_submit_button("Phân tích", use_container_width=True)

with col2:
    st.header("Kết quả")
    if submit:
        if not user_input.strip():
            st.warning("Vui lòng nhập bình luận trước khi phân tích.")
        else:
            progress = st.progress(0)
            cleaned = clean_data(user_input)
            progress.progress(50)
            vec = tfidf_vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            progress.progress(100)

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            if pred == 'cực kỳ hài lòng':
                st.success(f"{pred.upper()}")
            else:
                st.error(f"{pred.upper()}")
            st.write("**Bình luận gốc:**", user_input)
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("© 2025 Nhóm 9 • Github: [Huy](https://github.com/n1tee/SIC_BD_09.git)")