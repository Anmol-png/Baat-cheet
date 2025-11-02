# ==========================================
# 🤖 BaatGPT - Streamlit + Groq AI Assistant (Offline TTS)
# ==========================================

import streamlit as st
import requests
import tempfile
import pyttsx3
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="BaatGPT Voice Assistant", layout="centered")
st.title("🎙️ BaatGPT - Pakistani AI Voice Assistant")

# --- API KEY from Streamlit secrets ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found! Add it to .streamlit/secrets.toml")
    st.stop()

headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

# --- Conversation memory ---
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# --- Audio input ---
st.markdown("### 🎤 Upload your English audio question")
audio_input = st.file_uploader("Upload WAV / MP3", type=["wav", "mp3"])

if audio_input is not None:
    st.info("⏳ Processing your audio...")

    # --- Save temp file ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_input.read())
        tmp_audio_path = tmp_file.name

    # --- Transcribe using Groq Whisper ---
    try:
        with open(tmp_audio_path, "rb") as f:
            files = {"file": (tmp_audio_path, f, "audio/wav")}
            data = {"model": "whisper-large-v3-turbo"}
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data
            )
        r.raise_for_status()
        user_text = r.json().get("text", "")
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        st.stop()

    if not user_text:
        st.error("❌ Could not transcribe audio")
        st.stop()

    st.success(f"🗣️ You said: {user_text}")
    st.session_state.conversation_history.append({"role": "user", "content": user_text})

    # --- AI response ---
    chat_payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "system", "content": "You are BaatGPT, a friendly AI assistant for Pakistan."}] + st.session_state.conversation_history,
        "temperature": 0.7
    }

    try:
        chat_resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json=chat_payload
        )
        chat_resp.raise_for_status()
        resp_json = chat_resp.json()
        ai_reply = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "❌ AI did not return a valid reply")
    except Exception as e:
        st.error(f"❌ Chat API failed: {e}")
        st.stop()

    st.session_state.conversation_history.append({"role": "assistant", "content": ai_reply})
    st.markdown(f"### 🤖 BaatGPT says:\n{ai_reply}")

    # --- TTS using pyttsx3 (offline) ---
    try:
        tts_engine = pyttsx3.init()
        tts_engine.save_to_file(ai_reply, "baatgpt_reply.mp3")
        tts_engine.runAndWait()

        # Read saved audio
        with open("baatgpt_reply.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", start_time=0)
    except Exception as e:
        st.warning(f"TTS failed: {e}")
