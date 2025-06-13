import whisper
import warnings
import streamlit as st
import os

# Ignore FP16 warning
warnings.filterwarnings("ignore", category=UserWarning)

st.title("🎙️ Whisper Transcription App")

# File upload widget
uploaded_file = st.file_uploader("Upload an MP3/WAV audio file", type=["mp3", "wav"])

if uploaded_file:
    # Save to temp file
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    st.info("⏳ Transcribing...")

    # Load Whisper model (use "base", "small", "medium", "large" as needed)
    model = whisper.load_model("tiny")

    # Transcribe
    result = model.transcribe("temp_audio.mp3")

    # Show result
    st.success("✅ Transcription Complete")
    st.text_area("Transcribed Text:", result["text"], height=200)

    # Optional: clean up temp file
    os.remove("temp_audio.mp3")
