import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
from datetime import datetime
import matplotlib.pyplot as plt
from googletrans import Translator
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# -------------------- DETECT LOCAL VS CLOUD --------------------
IS_LOCAL = os.environ.get("IS_LOCAL", "true") == "true"

# -------------------- LOAD CHATBOT --------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
translator = Translator()

# -------------------- STYLES --------------------
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to right, #ffecd2, #fcb69f); }
    .card { background-color: rgba(255, 255, 255, 0.9); padding:20px; border-radius:15px; margin-bottom:20px; }
    .stButton>button {
        background-color: #ff7f50; color:white; font-weight:bold; border-radius:10px; padding:0.5em 1em;
    }
    .stButton>button:hover { background-color:#ff4500; }
    h1 { color:#ff4500; text-align:center; }
    </style>
""", unsafe_allow_html=True)

# -------------------- APP TITLE --------------------
st.markdown('<h1>🌍 Multilingual Voice + Chat Sentiment Chatbot</h1>', unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "reminders" not in st.session_state:
    st.session_state.reminders = []

# -------------------- FUNCTION: Chatbot --------------------
def chatbot_reply(user_text):
    try:
        new_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")

        # Handle chat history safely
        if st.session_state.chat_history_ids is not None:
            st.session_state.chat_history_ids = st.session_state.chat_history_ids.to("cpu")
            MAX_HISTORY = 6
            truncated_history = st.session_state.chat_history_ids[:, -MAX_HISTORY:]
            bot_input_ids = torch.cat([truncated_history, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        st.session_state.chat_history_ids = model.generate(
            bot_input_ids,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id
        )

        reply = tokenizer.decode(
            st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

    except RuntimeError as e:
        st.error(f"Chatbot error: {e}")
        return "Sorry, there was an error generating a reply."

    return reply

# -------------------- FUNCTION: Multilingual Processing --------------------
def process_text(user_text, lang="en"):
    if lang != "en":
        translated = translator.translate(user_text, src=lang, dest="en")
        user_text_en = translated.text
    else:
        user_text_en = user_text

    analysis = TextBlob(user_text_en)
    polarity = analysis.sentiment.polarity
    reply_en = chatbot_reply(user_text_en)

    if lang != "en":
        translated_reply = translator.translate(reply_en, src="en", dest=lang)
        reply = translated_reply.text
    else:
        reply = reply_en

    return reply, reply_en, polarity

# -------------------- LANG SELECTION --------------------
lang_choice = st.selectbox("🌐 Choose Language", ["English", "Hindi", "Odia"])
lang_map = {"English": "en", "Hindi": "hi", "Odia": "or"}
user_lang = lang_map[lang_choice]

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["🗣 Voice Input", "⌨ Chat Input"])

# -------------------- VOICE INPUT --------------------
with tab1:
    st.markdown('<div class="card"><h2>🎤 Speak Now</h2></div>', unsafe_allow_html=True)
    if IS_LOCAL:
        if st.button("Start Listening"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening... please speak into your mic")
                audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio, language=f"{user_lang}-IN" if user_lang != "en" else "en-IN")
                st.success(f"Recognized: {text}")

                reply, reply_en, polarity = process_text(text, lang=user_lang)

                if polarity > 0:
                    st.markdown(f"<h3 style='color:green'>😊 Positive (Polarity: {polarity})</h3>", unsafe_allow_html=True)
                elif polarity < 0:
                    st.markdown(f"<h3 style='color:red'>😡 Negative (Polarity: {polarity})</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:orange'>😐 Neutral (Polarity: {polarity})</h3>", unsafe_allow_html=True)

                st.markdown(f"<div class='card'><b>🤖 Bot (Translated):</b> {reply}</div>", unsafe_allow_html=True)
                if lang_choice != "English":
                    st.markdown(f"<div class='card'><b>🤖 Bot (Original English):</b> {reply_en}</div>", unsafe_allow_html=True)

                try:
                    engine = pyttsx3.init()
                    engine.say(reply)
                    engine.runAndWait()
                except:
                    pass

                st.session_state.sentiment_history.append((datetime.now(), polarity))

            except:
                st.error("Sorry, I couldn't recognize what you said.")
    else:
        st.info("Voice input is disabled on Streamlit Cloud. Use text input instead.")

# -------------------- CHAT INPUT --------------------
with tab2:
    st.markdown('<div class="card"><h2>💬 Enter Text</h2></div>', unsafe_allow_html=True)
    chat_input = st.text_input("Type a message")
    if st.button("Analyze & Reply"):
        if chat_input:
            reply, reply_en, polarity = process_text(chat_input, lang=user_lang)

            if polarity > 0:
                st.markdown(f"<h3 style='color:green'>😊 Positive (Polarity: {polarity})</h3>", unsafe_allow_html=True)
            elif polarity < 0:
                st.markdown(f"<h3 style='color:red'>😡 Negative (Polarity: {polarity})</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:orange'>😐 Neutral (Polarity: {polarity})</h3>", unsafe_allow_html=True)

            st.markdown(f"<div class='card'><b>🤖 Bot (Translated):</b> {reply}</div>", unsafe_allow_html=True)
            if lang_choice != "English":
                st.markdown(f"<div class='card'><b>🤖 Bot (Original English):</b> {reply_en}</div>", unsafe_allow_html=True)

            try:
                if IS_LOCAL:
                    engine = pyttsx3.init()
                    engine.say(reply)
                    engine.runAndWait()
            except:
                pass

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
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time")
    plt.ylabel("Polarity")
    plt.title("Sentiment Trend Over Time")
    st.pyplot(plt)

# -------------------- REMINDERS --------------------
st.markdown('<div class="card"><h2>⏰ Reminders</h2></div>', unsafe_allow_html=True)
reminder_input = st.text_input("Add a new reminder")
if st.button("Add Reminder"):
    if reminder_input:
        st.session_state.reminders.append((datetime.now().strftime("%Y-%m-%d %H:%M"), reminder_input))
        st.success("Reminder added!")
    else:
        st.warning("Enter a reminder first")

# Voice input for reminders (local only)
if IS_LOCAL:
    with st.expander("🎤 Speak a Reminder"):
        if st.button("Start Listening for Reminder"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening... please speak your reminder")
                audio = recognizer.listen(source)
            try:
                reminder_text = recognizer.recognize_google(audio, language=f"{user_lang}-IN" if user_lang!="en" else "en-IN")
                st.success(f"Recognized: {reminder_text}")
                st.session_state.reminders.append((datetime.now().strftime("%Y-%m-%d %H:%M"), reminder_text))
            except:
                st.error("Could not recognize the reminder")

# Display and delete reminders
if st.session_state.reminders:
    st.markdown("*Your Reminders:*")
    for i, (t, r) in enumerate(st.session_state.reminders):
        col1, col2 = st.columns([8,1])
        col1.markdown(f"- [{t}] {r}")
        if col2.button("❌", key=f"del{i}"):
            st.session_state.reminders.pop(i)
            st.experimental_rerun()

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    '<p style="text-align:center;">🌐 Multilingual AI Chatbot | Built with Python + Streamlit + HuggingFace + GoogleTrans</p>',
    unsafe_allow_html=True)