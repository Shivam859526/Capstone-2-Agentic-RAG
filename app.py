import uuid

import requests

import streamlit as st

# -------------------- CONFIG --------------------

st.set_page_config(

    page_title="MediAssist | Healthcare AI",

    page_icon="🩺",

    layout="wide",

    initial_sidebar_state="expanded"

)

API_URL = "http://127.0.0.1:8000/query"

# -------------------- SESSION STATE --------------------

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:

    st.session_state.messages = []

if "pending_question" not in st.session_state:

    st.session_state.pending_question = None

# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {

    font-family: 'DM Sans', sans-serif;

}

.stApp {

    background: #f5f8fc;

}

[data-testid="stHeader"] {

    background: rgba(245, 248, 252, 0.9);

}

.block-container {

    max-width: 1100px;

    padding-top: 2rem;

    padding-bottom: 6rem;

}

[data-testid="stSidebar"] {

    background: #102a43;

}

[data-testid="stSidebar"] * {

    color: #f2f7fc;

}

.brand {

    font-size: 25px;

    font-weight: 700;

    color: #ffffff;

    padding: 10px 0 4px 0;

}

.brand span {

    color: #62e6c7;

}

.sidebar-note {

    color: #bdcfe0;

    font-size: 13px;

    line-height: 1.7;

}

.hero {

    background: linear-gradient(120deg, #123b56, #087f8c);

    padding: 32px;

    border-radius: 22px;

    color: white;

    margin-bottom: 24px;

    box-shadow: 0 10px 28px rgba(16, 42, 67, 0.12);

}

.hero-tag {

    display: inline-block;

    background: rgba(255,255,255,0.14);

    border: 1px solid rgba(255,255,255,0.2);

    padding: 6px 12px;

    border-radius: 30px;

    font-size: 12px;

    margin-bottom: 12px;

}

.hero h1 {

    color: white;

    font-size: 34px;

    line-height: 1.2;

    margin: 0 0 10px 0;

}

.hero p {

    color: #e1f5f5;

    font-size: 15px;

    margin: 0;

    max-width: 650px;

    line-height: 1.7;

}

.section-title {

    color: #16324f;

    font-size: 19px;

    font-weight: 700;

    margin: 20px 0 12px 0;

}

.info-card {

    background: white;

    padding: 17px;

    border-radius: 15px;

    border: 1px solid #e2eaf2;

    min-height: 110px;

}

.info-card h4 {

    color: #123b56;

    margin: 0 0 7px 0;

    font-size: 15px;

}

.info-card p {

    color: #60758a;

    margin: 0;

    font-size: 13px;

    line-height: 1.6;

}

div.stButton > button {

    border-radius: 12px;

    border: 1px solid #dce7ef;

    background: white;

    color: #16324f;

    min-height: 48px;

    font-weight: 500;

    transition: all 0.2s ease;

}

div.stButton > button:hover {

    border-color: #078b91;

    color: #087f8c;

    background: #effcf9;

}

[data-testid="stChatMessage"] {

    background: white;

    border: 1px solid #e6edf4;

    border-radius: 16px;

    padding: 14px;

    margin-bottom: 12px;

}

[data-testid="stChatInput"] {

    border-radius: 18px;

}

div[data-testid="stChatInput"] textarea {

    background: white;

}

.disclaimer {

    background: #fff8e8;

    color: #775a20;

    padding: 13px 15px;

    border-radius: 12px;

    border: 1px solid #f3e3b8;

    font-size: 12px;

    line-height: 1.6;

    margin-top: 20px;

}

.footer {

    color: #8192a5;

    font-size: 12px;

    text-align: center;

    padding-top: 22px;

}
</style>

""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.markdown(

        '<div class="brand">✚ Medi<span>Assist</span></div>',

        unsafe_allow_html=True

    )

    st.markdown(

        '<p class="sidebar-note">Your AI-powered healthcare knowledge companion.</p>',

        unsafe_allow_html=True

    )

    st.divider()

    if st.button("＋  Start New Chat", use_container_width=True):

        st.session_state.thread_id = str(uuid.uuid4())

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()

    st.markdown("### Your assistant")

    st.markdown(

        """
<div class="sidebar-note">

        ◈ Medical information retrieval<br><br>

        ◈ Context-aware AI responses<br><br>

        ◈ Reference sources when available
</div>

        """,

        unsafe_allow_html=True

    )

    st.divider()

    st.markdown(

        """
<div class="sidebar-note">
<b>Privacy reminder</b><br>

        Avoid entering names, contact details, or other identifiable patient information.
</div>

        """,

        unsafe_allow_html=True

    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.caption("MediAssist • Academic project")

# -------------------- HERO HEADER --------------------

st.markdown("""
<div class="hero">
<div class="hero-tag">✦ AI-POWERED HEALTHCARE KNOWLEDGE</div>
<h1>Your health questions,<br>answered with clarity.</h1>
<p>

        Explore healthcare information with an AI assistant that retrieves

        relevant medical knowledge to help explain your questions.
</p>
</div>

""", unsafe_allow_html=True)

# -------------------- FEATURE CARDS --------------------

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
<div class="info-card">
<h4>⌕ Knowledge retrieval</h4>
<p>Find relevant information from your medical knowledge base.</p>
</div>

    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
<div class="info-card">
<h4>✧ AI explanations</h4>
<p>Get clear, conversational answers to healthcare questions.</p>
</div>

    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
<div class="info-card">
<h4>▤ Source references</h4>
<p>Review returned references when your knowledge base provides them.</p>
</div>

    """, unsafe_allow_html=True)

# -------------------- SUGGESTED QUESTIONS --------------------

if not st.session_state.messages:

    st.markdown(

        '<div class="section-title">What would you like to know?</div>',

        unsafe_allow_html=True

    )

    suggestions = [

        "What are common symptoms of diabetes?",

        "What is high blood pressure?",

        "What are common symptoms of anemia?",

        "How can I maintain a healthy lifestyle?"

    ]

    left, right = st.columns(2)

    for i, suggestion in enumerate(suggestions):

        with (left if i % 2 == 0 else right):

            if st.button(suggestion, key=f"suggestion_{i}",

                         use_container_width=True):

                st.session_state.pending_question = suggestion

                st.rerun()

# -------------------- DISPLAY CHAT HISTORY --------------------

st.markdown(

    '<div class="section-title">Conversation</div>',

    unsafe_allow_html=True

)

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("View sources"):

                for source in message["sources"]:

                    st.write(source)

# -------------------- HANDLE QUESTION --------------------

question = st.chat_input("Ask your healthcare question...")

if question:

    st.session_state.pending_question = question

if st.session_state.pending_question:

    question = st.session_state.pending_question

    st.session_state.pending_question = None

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching the knowledge base and preparing your answer..."):

            try:

                response = requests.post(

                    API_URL,

                    json={

                        "question": question,

                        "thread_id": st.session_state.thread_id

                    },

                    timeout=180

                )

                response.raise_for_status()

                data = response.json()

                answer = data.get(

                    "answer",

                    "No answer was returned by the assistant."

                )

                sources = data.get("sources") or []

                st.markdown(answer)

                if sources:

                    with st.expander("View sources"):

                        for source in sources:

                            st.write(source)

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": sources

                })

            except requests.exceptions.ConnectionError:

                answer = (

                    "I couldn't connect to the backend. "

                    "Please make sure the FastAPI server is running."

                )

                st.error(answer)

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })

            except requests.exceptions.Timeout:

                answer = (

                    "The request took too long. Please try again."

                )

                st.error(answer)

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })

            except requests.exceptions.HTTPError as error:

                answer = (

                    f"The backend returned an error: {error.response.status_code}. "

                    "Check the FastAPI terminal for details."

                )

                st.error(answer)

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })

            except (ValueError, KeyError):

                answer = "The backend returned an unexpected response."

                st.error(answer)

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })

# -------------------- DISCLAIMER & FOOTER --------------------

st.markdown("""
<div class="disclaimer">
<b>Medical disclaimer:</b> MediAssist provides educational information

    and is not a substitute for professional medical advice, diagnosis, or

    treatment. For urgent or severe symptoms, seek appropriate medical care.
</div>
<div class="footer">

    MediAssist · Healthcare Knowledge Assistant · Powered by your RAG workflow
</div>

""", unsafe_allow_html=True)
 