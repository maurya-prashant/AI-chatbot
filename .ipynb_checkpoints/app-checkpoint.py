import streamlit as st
import pandas as pd
import pickle
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="AI Chatbot",
    layout="wide"
)

# --------------------------
# CSS
# --------------------------
st.markdown("""
<style>

/* Import Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* ==========================
Background
========================== */

.stApp{
    background: linear-gradient(
        135deg,
        #FFF8F4 0%,
        #FFE6D9 55%,
        #FFF4EF 100%
    );
    color:#3D2C2E;
}

/* Floating blur circles */

.stApp::before{
    content:"";
    position:fixed;
    width:450px;
    height:450px;
    top:-180px;
    right:-160px;
    background:#FFD5C4;
    border-radius:50%;
    filter:blur(120px);
    opacity:.65;
    z-index:-1;
}

.stApp::after{
    content:"";
    position:fixed;
    width:420px;
    height:420px;
    bottom:-170px;
    left:-170px;
    background:#FFBEA3;
    border-radius:50%;
    filter:blur(130px);
    opacity:.55;
    z-index:-1;
}

/* ==========================
Main Container
========================== */

.block-container{
    max-width:1100px;
    padding-top:2rem;
}

/* ==========================
Title
========================== */

.title{

    font-size:60px;
    font-weight:800;
    text-align:center;

    background:linear-gradient(
    90deg,
    #FF8A65,
    #FFB38A);

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    margin-bottom:5px;

}

.subtitle{

    text-align:center;
    color:#7A5A55;
    font-size:18px;
    margin-bottom:35px;

}

/* ==========================
User Bubble
========================== */

.chat-user{

    background:linear-gradient(
        135deg,
        #FFB38A,
        #FF8A65);

    padding:18px;

    border-radius:22px;

    color:white;

    margin:12px 0;

    box-shadow:
        0px 10px 25px rgba(255,138,101,.28);

    transition:.3s;

}

.chat-user:hover{

    transform:translateY(-2px);

}

/* ==========================
Bot Bubble
========================== */

.chat-bot{

    background:rgba(255,255,255,.75);

    backdrop-filter:blur(15px);

    padding:18px;

    border-radius:22px;

    color:#3D2C2E;

    border:1px solid #FFE2D4;

    margin:12px 0;

    box-shadow:
        0px 8px 20px rgba(0,0,0,.06);

    transition:.3s;

}

.chat-bot:hover{

    transform:translateY(-2px);

}

/* ==========================
Sidebar
========================== */

section[data-testid="stSidebar"]{

    background:#FFEADF;

    border-right:1px solid #FFD5C8;

}

/* ==========================
Buttons
========================== */

.stButton>button{

    background:linear-gradient(
        90deg,
        #FFB38A,
        #FF8A65);

    color:white;

    border:none;

    border-radius:30px;

    font-weight:600;

    padding:10px 22px;

    transition:.3s;

}

.stButton>button:hover{

    transform:translateY(-2px);

    box-shadow:
        0px 10px 22px rgba(255,138,101,.35);

}

/* ==========================
Chat Input
========================== */

.stChatInput{

    background:rgba(255,255,255,.85);

    backdrop-filter:blur(15px);

    border-radius:40px;

    border:1px solid #FFD9CA;

    box-shadow:
        0px 12px 28px rgba(255,160,122,.18);

}

.stChatInput input{

    color:#4B3632;

    font-size:17px;

    background:transparent;

}

/* ==========================
Text Input
========================== */

.stTextInput input{

    background:white;

    color:#4B3632;

    border-radius:15px;

    border:1px solid #FFD7C9;

}

/* ==========================
Scrollbar
========================== */

::-webkit-scrollbar{

    width:10px;

}

::-webkit-scrollbar-thumb{

    background:#FFB38A;

    border-radius:50px;

}

::-webkit-scrollbar-track{

    background:#FFF4EF;

}

/* ==========================
Selection
========================== */

::selection{

    background:#FFB38A;

    color:white;

}

</style>
""", unsafe_allow_html=True)

# --------------------------
# Download NLTK
# --------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# --------------------------
# Load Files
# --------------------------
model = pickle.load(open("chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
df = pd.read_csv("chatbot.csv")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# --------------------------
# Cleaning Function
# --------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = word_tokenize(text)

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# --------------------------
# Prediction
# --------------------------
def get_response(user_input):

    clean = clean_text(user_input)

    vector = vectorizer.transform([clean])

    intent = model.predict(vector)[0]

    response = df[df["intent"] == intent]["response"].iloc[0]

    return response

# --------------------------
# Session
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages=[]

# --------------------------
# Header
# --------------------------
st.markdown("<div class='title'>AI Chatbot</div>",unsafe_allow_html=True)

# st.markdown("<div class='subtitle'>Machine Learning Powered Intent Classification Chatbot</div>",unsafe_allow_html=True)

# --------------------------
# Sidebar
# --------------------------
with st.sidebar:

    st.title("Controls")

    if st.button("Clear Chat"):

        st.session_state.messages=[]

        st.rerun()

    st.markdown("---")

    st.write("### Model")

    st.success("Logistic Regression")

    st.write("### NLP")

    st.info("TF-IDF + NLTK")

# --------------------------
# Chat History
# --------------------------
for role,msg in st.session_state.messages:

    if role=="user":

        st.markdown(
        f"<div class='chat-user'><b>You</b><br>{msg}</div>",
        unsafe_allow_html=True)

    else:

        st.markdown(
        f"<div class='chat-bot'><b>Bot</b><br>{msg}</div>",
        unsafe_allow_html=True)

# --------------------------
# Input
# --------------------------
prompt=st.chat_input("Ask me anything...")

if prompt:

    st.session_state.messages.append(("user",prompt))

    with st.spinner("Thinking..."):

        response=get_response(prompt)

    st.session_state.messages.append(("bot",response))

    st.rerun()