import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from st_audiorec import st_audiorec   # 🎙 mic recorder
from wordcloud import WordCloud       # ☁️ word cloud

# -------------------- LOAD CHATBOT --------------------
@st.cache_resource
def load_chatbot():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

# ✅ spinner + success instead of "Running load_chatbot()"
with st.spinner("🤖 Loading chatbot model, please wait..."):
    tokenizer, model = load_chatbot()
st.success("✅ Chatbot ready!")

# -------------------- STYLES --------------------
def set_theme(dark=False):
    if dark:
        return """
        <style>
        .stApp { background: linear-gradient(to right, #1e1e2f, #2d2d44); color: white; }
        .card { background-color: #2f2f40; padding:20px; border-radius:15px; margin-bottom:20px; }
        .user-bubble { background:#ff7f50; color:white; padding:8px 15px; border-radius:15px; margin:5px 0; text-align:right; }
        .bot-bubble { background:#444; color:#eee; padding:8px 15px; border-radius:15px; margin:5px 0; text-align:left; }
        </style>
        """
    else:
        return """
        <style>
        .stApp { background: linear-gradient(to right, #ffecd2, #fcb69f); }
        .card { background-color: rgba(255, 255, 255, 0.9); padding:20px; border-radius:15px; margin-bottom:20px; }
        .user-bubble { background:#ff7f50; color:white; padding:8px 15px; border-radius:15px; margin:5px 0; text-align:right; }
        .bot-bubble { background:#f1f1f1; color:#333; padding:8px 15px; border-radius:15px; margin:5px 0; text-align:left; }
        </style>
        """

dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
st.markdown(set_theme(dark_mode), unsafe_allow_html=True)

# -------------------- APP TITLE --------------------
st.markdown('<h1 style="text-align:center;">🌍 Multilingual Voice + Chat Sentiment Chatbot</h1>', unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "reminders" not in st.session_state:
    st.session_state.reminders = []
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# -------------------- FUNCTIONS --------------------
def chatbot_reply(user_text):
    new_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply

def translate_text(text, src="en", dest="en"):
    if src == dest: return text
    return GoogleTranslator(source=src, target=dest).translate(text)

def process_text(user_text, lang="en"):
    user_text_en = translate_text(user_text, src=lang, dest="en")
    polarity = TextBlob(user_text_en).sentiment.polarity
    reply_en = chatbot_reply(user_text_en)
    reply = translate_text(reply_en, src="en", dest=lang)
    return reply, polarity

def speak_text(text):
    tts = gTTS(text=text, lang="en")
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    st.audio(audio_bytes.getvalue(), format="audio/mp3")

# -------------------- LANG SELECTION --------------------
lang_choice = st.sidebar.selectbox("🌐 Choose Language", ["English", "Hindi", "Odia"])
lang_map = {"English": "en", "Hindi": "hi", "Odia": "or"}
user_lang = lang_map[lang_choice]

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["🗣 Voice Input", "⌨ Chat Input"])

# -------------------- VOICE INPUT --------------------
with tab1:
    st.markdown('<div class="card"><h2>🎤 Voice Input</h2></div>', unsafe_allow_html=True)

    # 🎙 Live mic recording
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None and st.button("Analyze Recorded Voice"):
        try:
            with open("temp.wav", "wb") as f:
                f.write(wav_audio_data)
            recognizer = sr.Recognizer()
            with sr.AudioFile("temp.wav") as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="en-IN")
            st.success(f"Recognized: {text}")
            reply, polarity = process_text(text, lang=user_lang)

            if polarity > 0:
                st.markdown(f"<h3 style='color:green'>😊 Positive ({polarity:.2f})</h3>", unsafe_allow_html=True)
            elif polarity < 0:
                st.markdown(f"<h3 style='color:red'>😡 Negative ({polarity:.2f})</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:orange'>😐 Neutral ({polarity:.2f})</h3>", unsafe_allow_html=True)

            st.session_state.chat_log.append(("You", text))
            st.session_state.chat_log.append(("Bot", reply))
            speak_text(reply)
            st.session_state.sentiment_history.append((datetime.now(), polarity))
        except Exception as e:
            st.error(f"Speech recognition failed: {e}")

    # File upload
    audio_file = st.file_uploader("Upload voice (wav/mp3)", type=["wav", "mp3"])
    if audio_file and st.button("Analyze Uploaded Voice"):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="en-IN")
            st.success(f"Recognized: {text}")
            reply, polarity = process_text(text, lang=user_lang)
            if polarity > 0:
                st.markdown(f"<h3 style='color:green'>😊 Positive ({polarity:.2f})</h3>", unsafe_allow_html=True)
            elif polarity < 0:
                st.markdown(f"<h3 style='color:red'>😡 Negative ({polarity:.2f})</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:orange'>😐 Neutral ({polarity:.2f})</h3>", unsafe_allow_html=True)
            st.session_state.chat_log.append(("You", text))
            st.session_state.chat_log.append(("Bot", reply))
            speak_text(reply)
            st.session_state.sentiment_history.append((datetime.now(), polarity))
        except Exception as e:
            st.error(f"Speech recognition failed: {e}")

# -------------------- CHAT INPUT --------------------
with tab2:
    st.markdown('<div class="card"><h2>💬 Chat</h2></div>', unsafe_allow_html=True)
    chat_input = st.text_input("Type your message")
    if st.button("Send"):
        if chat_input:
            reply, polarity = process_text(chat_input, lang=user_lang)
            if polarity > 0:
                st.markdown(f"<h3 style='color:green'>😊 Positive ({polarity:.2f})</h3>", unsafe_allow_html=True)
            elif polarity < 0:
                st.markdown(f"<h3 style='color:red'>😡 Negative ({polarity:.2f})</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:orange'>😐 Neutral ({polarity:.2f})</h3>", unsafe_allow_html=True)
            st.session_state.chat_log.append(("You", chat_input))
            st.session_state.chat_log.append(("Bot", reply))
            speak_text(reply)
            st.session_state.sentiment_history.append((datetime.now(), polarity))
        else:
            st.warning("Please type a message.")

# -------------------- CHAT LOG --------------------
if st.session_state.chat_log:
    st.markdown('<div class="card"><h2>💬 Conversation</h2></div>', unsafe_allow_html=True)
    for sender, msg in st.session_state.chat_log:
        if sender == "You":
            st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>{msg}</div>", unsafe_allow_html=True)

# -------------------- SENTIMENT TREND --------------------
if len(st.session_state.sentiment_history) > 1:
    st.markdown('<div class="card"><h2>📊 Sentiment Trend</h2></div>', unsafe_allow_html=True)
    times, polarities = zip(*st.session_state.sentiment_history)
    plt.figure(figsize=(8, 3))
    colors = ['green' if p > 0 else 'red' if p < 0 else 'orange' for p in polarities]
    plt.plot(times, polarities, marker='o', linestyle='-', color='#ff7f50')
    plt.scatter(times, polarities, color=colors, s=100)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Polarity")
    plt.title("Sentiment Trend Over Time")
    plt.xticks(rotation=30)
    st.pyplot(plt)

    # ⬇️ Export CSV
    df = pd.DataFrame(st.session_state.sentiment_history, columns=["time", "polarity"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Sentiment History", csv, "sentiment_history.csv", "text/csv")

# -------------------- WORD CLOUD --------------------
if st.session_state.chat_log:
    st.markdown('<div class="card"><h2>☁️ Word Cloud</h2></div>', unsafe_allow_html=True)
    all_text = " ".join([msg for sender, msg in st.session_state.chat_log if sender == "You"])
    if all_text.strip():
        wordcloud = WordCloud(width=600, height=300, background_color="white").generate(all_text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
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

if st.session_state.reminders:
    for t, r in st.session_state.reminders:
        st.markdown(f"- [{t}] {r}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>✨ Super-Polished Multilingual AI Chatbot | Python + Streamlit</p>", unsafe_allow_html=True)
