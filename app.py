import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------- HTML + CSS STYLING --------------------
st.markdown("""
    <style>
    /* Background gradient for entire app */
    .stApp {
        background: linear-gradient(to right, #ffecd2, #fcb69f);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
    }

    /* Card style for each section */
    .card {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
    }

    /* Button styling */
    .stButton>button {
        background-color: #ff7f50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff4500;
        color: white;
    }

    /* Header style */
    h1 {
        color: #ff4500;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- APP TITLE --------------------
st.markdown('<h1>🎙 Voice + Chat Sentiment Detection</h1>', unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []

tab1, tab2 = st.tabs(["🗣 Voice Input", "⌨ Chat Input"])

# -------------------- VOICE INPUT --------------------
with tab1:
    st.markdown('<div class="card"><h2>🎤 Speak Now</h2></div>', unsafe_allow_html=True)
    if st.button("Start Listening"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... please speak into your mic")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            if polarity > 0:
                st.markdown(f"<h3 style='color:green'>😊 Positive Sentiment (Polarity: {polarity})</h3>", unsafe_allow_html=True)
            elif polarity < 0:
                st.markdown(f"<h3 style='color:red'>😡 Negative Sentiment (Polarity: {polarity})</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:orange'>😐 Neutral Sentiment (Polarity: {polarity})</h3>", unsafe_allow_html=True)
            st.session_state.sentiment_history.append((datetime.now(), polarity))
        except:
            st.error("Sorry, I couldn't recognize what you said.")

# -------------------- CHAT INPUT --------------------
with tab2:
    st.markdown('<div class="card"><h2>💬 Enter Text</h2></div>', unsafe_allow_html=True)
    chat_input = st.text_input("Type a sentence")
    if st.button("Analyze Text"):
        if chat_input:
            analysis = TextBlob(chat_input)
            polarity = analysis.sentiment.polarity
            if polarity > 0:
                st.markdown(f"<h3 style='color:green'>😊 Positive Sentiment (Polarity: {polarity})</h3>", unsafe_allow_html=True)
            elif polarity < 0:
                st.markdown(f"<h3 style='color:red'>😡 Negative Sentiment (Polarity: {polarity})</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:orange'>😐 Neutral Sentiment (Polarity: {polarity})</h3>", unsafe_allow_html=True)
            st.session_state.sentiment_history.append((datetime.now(), polarity))
        else:
            st.warning("Please enter some text.")

# -------------------- TREND CHART --------------------
if len(st.session_state.sentiment_history) > 1:
    st.markdown('<div class="card"><h2>📊 Sentiment Trend</h2></div>', unsafe_allow_html=True)
    times, polarities = zip(*st.session_state.sentiment_history)
    plt.figure(figsize=(8, 3))
    colors = ['green' if p > 0 else 'red' if p < 0 else 'orange' for p in polarities]
    plt.plot(times, polarities, marker='o', linestyle='-', color='orange')
    plt.scatter(times, polarities, color=colors, s=100)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Polarity")
    plt.title("Sentiment Trend Over Time")
    plt.xticks(rotation=30)
    st.pyplot(plt)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown('<p style="text-align:center;">🔧 Built with Python + Streamlit + TextBlob + SpeechRecognition</p>', unsafe_allow_html=True)
