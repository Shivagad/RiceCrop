import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import tempfile
import soundfile as sf
from gtts import gTTS
import numpy as np

# ---------------- FIRST COMMAND ----------------
st.set_page_config(page_title="🌾 तांदूळ शेतकरी सहाय्यक", layout="centered")

# ------------------- BACKGROUND CSS -------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: green;
}

.output-box {
    background-color: green;
    padding: 18px;
    border-radius: 12px;
    border: 2px solid #e6d574;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------

st.title("🌾 तांदूळ शेतकरी सहाय्यक Chatbot")
st.write("### इंग्रजी & मराठी | मजकूर & आवाज मार्गदर्शन")

load_dotenv()

# Load API Key
# API_KEY = os.getenv("GEMINI_API_KEY")
API_KEY ="AIzaSyDa8cZqhlkBU7xBdJCUiHft2dPV-pkCjLY"
genai.configure(api_key=API_KEY)

# Select model
model = genai.GenerativeModel("models/gemma-3-4b-it")

# ---------------- LANGUAGE SELECTION ----------------
language = st.selectbox("भाषा निवडा / Choose language:", ["Marathi", "English"])

# ---------------- TEXT INPUT ----------------
if language == "Marathi":
    user_text = st.text_area("तुमचा प्रश्न टाका:")
else:
    user_text = st.text_area("Enter your question:")

# ---------------- AUDIO FILE FUNCTIONS ----------------
def save_uploaded_audio(uploaded_file):
    if uploaded_file is None:
        return None
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    return temp_file.name

def transcribe_audio(file_path):
    audio_data = genai.upload_file(file_path)
    response = model.generate_content([audio_data, "Transcribe this audio to text"])
    return response.text

# ---------------- AUDIO UPLOAD ----------------
# st.write("### 🎤 आवाज अपलोड करा / Upload your voice (optional)")
# uploaded_audio = st.file_uploader("Choose an audio file (wav, mp3)", type=["wav", "mp3"])

# ---------------- HANDLE QUERY ----------------
if st.button("🌾 उत्तर मिळवा / Get Answer"):

    # Check if audio file uploaded
    if uploaded_audio is not None:
        try:
            audio_file_path = save_uploaded_audio(uploaded_audio)
            user_text = transcribe_audio(audio_file_path)
            st.success("ऑडिओ ट्रान्सक्राइब झाला! / Audio transcribed!")
            st.write("**ओळखलेले शब्द / Recognized Text:** ", user_text)
        except Exception as e:
            st.error("ऑडिओ ट्रान्सक्रिप्शन अयशस्वी.")
            st.error(str(e))

    # Check empty input
    if not user_text:
        st.warning("कृपया मजकूर किंवा आवाज द्या.")
        st.stop()

    # Prepare system prompt
    if language == "Marathi":
        system_prompt = "तुम्ही तांदूळ शेततज्ज्ञ आहात. अगदी सोप्या मराठीत सविस्तर उत्तर द्या."
    else:
        system_prompt = "You are a rice crop expert. Give a simple clear answer."

    # ---------------- LLM Response ----------------
    with st.spinner("सल्ला तयार करत आहे / Generating advice..."):
        try:
            response = model.generate_content(system_prompt + "\n\nUser: " + user_text)
            bot_answer = response.text

            st.subheader("🌾 उत्तर / Answer:")
            st.markdown(f"<div class='output-box'>{bot_answer}</div>", unsafe_allow_html=True)

            # ---------------- TTS OUTPUT ----------------
            st.write("### 🔊 Bot Voice Output")
            if language == "Marathi":
                tts = gTTS(bot_answer, lang='mr')
            else:
                tts = gTTS(bot_answer, lang='en')

            audio_file = "bot_voice.mp3"
            tts.save(audio_file)
            st.audio(audio_file)

        except Exception as e:
            st.error("मॉडेल उत्तर देऊ शकले नाही.")
            st.error(str(e))
