# 🎤 Real-Time Chat with Sentiment Detection

> Built using Python · Streamlit · SpeechRecognition · TextBlob

A smart, AI-powered web app that lets users *type or speak* messages and instantly detects the *emotional tone*—Positive, Negative, or Neutral.

---

## 🚀 Live Demo  
👉 [Click here to try the app](https://voice-sentiment-app-<your-id>.streamlit.app)  
_(Replace <your-id> with your actual Streamlit app link)_

---

## ✨ Features

- 💬 *Real-Time Text Sentiment*: Instantly analyzes text messages as you type.
- 🎙 *Voice Input (WAV)*: Upload .wav audio files for sentiment detection.
- 🧠 *NLP-powered: Uses TextBlob to classify messages as **Positive, **Negative, or **Neutral*.
- 📱 *Mobile-Friendly UI*: Clean and responsive interface powered by Streamlit.
- 📁 *Safe File Handling*: Uploaded files are processed securely in memory.
- 🔁 *Two Input Modes*: Text box and audio file upload—choose as per your preference.

---

## 📸 Screenshot

![App Screenshot](screenshot.png)  
_(You can replace screenshot.png with a real screenshot image of your app in the repo.)_

---

## 🛠 Tech Stack

- *Python 3.9+*
- *Streamlit* – Frontend and deployment
- *TextBlob* – Sentiment analysis
- *SpeechRecognition* – Converts voice to text
- *PyDub* – Audio preprocessing
- *Tempfile* – Temporary file handling

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ayan09l/voice-sentiment-app.git
cd voice-sentiment-app
pip install -r requirements.txt