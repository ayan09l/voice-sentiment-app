
import streamlit as st
from textblob import TextBlob
import speech_recognition as sr
import tempfile

st.set_page_config(page_title="Voice + Chat Sentiment App", layout="centered")
st.title("🎙 Real-Time Chat with Sentiment Detection")

# Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "😊 Positive"
    elif polarity < 0:
        sentiment = "😠 Negative"
    else:
        sentiment = "😐 Neutral"
    return sentiment, polarity

# Voice to Text
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, couldn't understand the audio."
        except sr.RequestError:
            return "API unavailable."

# Text Input
st.subheader("💬 Type Your Message")
text_input = st.text_input("Enter something here")

if text_input:
    sentiment, polarity = analyze_sentiment(text_input)
    st.success(f"You said: {text_input}")
    st.info(f"Sentiment: {sentiment} (Polarity: {polarity})")

# Voice Input
st.subheader("🎤 Upload Voice (WAV only)")
audio_file = st.file_uploader("Choose an audio file", type=["wav"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        temp_file_path = tmp.name

    st.write("🕒 Transcribing...")
    transcript = transcribe_audio(temp_file_path)
    st.success(f"Transcription: {transcript}")
    sentiment, polarity = analyze_sentiment(transcript)
    st.info(f"Sentiment: {sentiment} (Polarity: {polarity})")

st.markdown("---")
st.caption("🧠 Built using Python • Streamlit • SpeechRecognition • TextBlob")