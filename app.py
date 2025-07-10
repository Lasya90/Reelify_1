import streamlit as st
import os
import ffmpeg
import shutil
import glob
import time
import whisper
import json
from transformers import pipeline
import random

# ----------------------------
# Helper Functions
# ----------------------------

def safe_delete_dir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except PermissionError:
                st.warning(f"⚠ Skipped locked file: {name}")
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass
    try:
        os.rmdir(path)
    except OSError:
        pass

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

def format_timestamp(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02}"

def get_reel_segments(transcript: str, video_duration_secs=600) -> str:
    chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
    highlights = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        highlights.append(summary[0]['summary_text'])

    total_segments = min(5, len(highlights))
    time_step = video_duration_secs // (total_segments + 1)
    start_time = time_step

    formatted = ""
    for i, point in enumerate(highlights[:total_segments], 1):
        end_time = start_time + 30
        formatted += f"{i}. {format_timestamp(start_time)} - {format_timestamp(end_time)}: {point.strip()}\n"
        start_time += time_step

    return formatted

# ----------------------------
# Paths & Constants
# ----------------------------

TEMP_DIR = "temp"
AUDIO_WAV = os.path.normpath(os.path.join(TEMP_DIR, "audio.wav"))
AUDIO_MP3 = os.path.normpath(os.path.join(TEMP_DIR, "audio.mp3"))
VERTICAL_VIDEO = os.path.normpath(os.path.join(TEMP_DIR, "vertical_output.mp4"))
CHUNK_PATTERN = os.path.normpath(os.path.join(TEMP_DIR, "chunk_%03d.mp4"))

os.makedirs(TEMP_DIR, exist_ok=True)

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🎨 Reelify - Video Processor")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
uploaded_audios = st.file_uploader("Upload audio files for transcription", type=["mp3", "wav"], accept_multiple_files=True)

reel_video_slot = st.empty()
chunk_video_slots = []

if uploaded_file:
    input_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ File uploaded successfully.")

    if st.button("Extract Audio & Convert Video to Reel Format"):
        st.info("🔊 Extracting audio (.wav)...")
        try:
            ffmpeg.input(input_path).output(AUDIO_WAV, **{'q:a': 0, 'map': 'a'}).run(overwrite_output=True)
            st.success("✅ Audio extracted: audio.wav")
        except ffmpeg.Error as e:
            st.error("❌ WAV extraction failed:")
            st.text(e.stderr.decode('utf-8') if e.stderr else str(e))

        st.info("🔊 Extracting audio (.mp3) for Whisper...")
        try:
            ffmpeg.input(input_path).output(AUDIO_MP3, **{'q:a': 0, 'map': 'a'}).run(overwrite_output=True)
            st.success("✅ Audio ready: audio.mp3")
            with open(AUDIO_MP3, "rb") as f:
                st.download_button("⬇ Download MP3", f, file_name="audio.mp3")
        except ffmpeg.Error as e:
            st.error("❌ MP3 extraction failed:")
            st.text(e.stderr.decode('utf-8') if e.stderr else str(e))

        st.info("🎥 Converting to vertical (Reel) format 1080x1920...")
        try:
            ffmpeg.input(input_path).output(
                VERTICAL_VIDEO,
                vf="scale=1080:-2,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
            ).run(overwrite_output=True)
            st.success("✅ Reel format created: vertical_output.mp4")
            reel_video_slot.video(VERTICAL_VIDEO)
        except ffmpeg.Error as e:
            st.error("❌ Reel conversion failed:")
            st.text(e.stderr.decode('utf-8') if e.stderr else str(e))

    if st.button("Create 30s Chunks from Reel Format"):
        st.info("✂ Splitting vertical video into 30-second chunks...")

        if not os.path.exists(VERTICAL_VIDEO):
            st.error("❌ Reel format video not found. Please run 'Extract Audio & Convert Video to Reel Format' first.")
            st.stop()

        try:
            for f in glob.glob(os.path.join(TEMP_DIR, "chunk_*.mp4")):
                try:
                    os.remove(f)
                except Exception:
                    pass

            ffmpeg.input(VERTICAL_VIDEO).output(
                CHUNK_PATTERN,
                f='segment',
                segment_time='30',
                c='copy'
            ).run(overwrite_output=True)

            chunks = sorted(glob.glob(os.path.join(TEMP_DIR, "chunk_*.mp4")))
            st.success(f"✅ Created {len(chunks)} chunks.")
            for chunk in chunks:
                video_slot = st.empty()
                video_slot.video(chunk)
                chunk_video_slots.append(video_slot)

        except ffmpeg.Error as e:
            st.error("❌ Chunking failed:")
            st.text(e.stderr.decode('utf-8') if e.stderr else str(e))

if uploaded_audios:
    st.header("🔍 Transcribe Multiple Audio Files (Whisper Only)")

    model = whisper.load_model("tiny")

    for audio in uploaded_audios:
        file_path = os.path.join(TEMP_DIR, audio.name)
        with open(file_path, "wb") as f:
            f.write(audio.read())

        st.info(f"📝 Transcribing {audio.name} using Whisper...")

        try:
            st.info("🌐 Detecting language...")
            audio_loaded = whisper.load_audio(file_path)
            audio_loaded = whisper.pad_or_trim(audio_loaded)
            mel = whisper.log_mel_spectrogram(audio_loaded).to(model.device)
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            st.success(f"✅ Detected Language: {detected_lang}")

            result = model.transcribe(file_path, language="en", fp16=False)
            transcript = result["text"]

            with open(os.path.join(TEMP_DIR, f"{audio.name}.txt"), "w", encoding="utf-8") as f:
                f.write(transcript)

            st.success(f"✅ Transcript for {audio.name} generated.")
            st.download_button(f"⬇ Download Transcript ({audio.name})", data=transcript, file_name=f"{audio.name}.txt")
            st.text_area(f"📜 Transcript Preview: {audio.name}", transcript, height=300)

            st.info("✨ Analyzing transcript for engaging reel segments...")
            try:
                highlights = get_reel_segments(transcript)
                st.success("✅ Highlighted Moments Extracted:")
                st.text_area("🌟 Reel-Worthy Moments", highlights, height=200)
            except Exception as e:
                st.error("❌ Failed to extract highlight moments.")
                st.text(str(e))

        except Exception as e:
            st.error(f"❌ Failed to transcribe {audio.name}")
            st.text(str(e))

if st.button("Clean Temporary Files"):
    reel_video_slot.empty()
    for slot in chunk_video_slots:
        slot.empty()
    try:
        safe_delete_dir(TEMP_DIR)
        time.sleep(1)
        os.makedirs(TEMP_DIR, exist_ok=True)
        st.success("🧹 Temporary files cleaned.")
    except Exception as e:
        st.error(f"❌ Failed to clean temp files: {e}")
