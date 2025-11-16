import os
import cv2
import logging
from google import genai
import whisper
import subprocess
import uuid
from dotenv import load_dotenv
load_dotenv()

# ---------------------------
# GLOBAL CONFIG VARIABLE
# ---------------------------
FRAME_EVERY_SECONDS = 60  # <--- SET THIS: 1, 2, 5, 10, 30, etc.

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='[INFO] %(message)s')

# ---------------------------
# Setup
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")
client = genai.Client(api_key=GEMINI_API_KEY)
whisper_model = whisper.load_model("small")


# ---------------------------
# Audio extraction helper
# ---------------------------
def extract_audio_from_video(video_path):
    temp_wav = f"_temp_{uuid.uuid4().hex}.wav"

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        temp_wav,
        "-y"
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return temp_wav


# ---------------------------
# FINAL VIDEO FUNCTION
# ---------------------------
def extract_video(path: str, frame_seconds: float = FRAME_EVERY_SECONDS,
                  model: str = "gemini-2.5-flash"):

    logging.info("=== VIDEO EXTRACTION STARTED ===")
    logging.info(f"Opening video: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"success": False, "error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    logging.info(f"Video FPS: {fps}")

    frame_interval = int(fps * frame_seconds)
    logging.info(f"Frame every {frame_seconds}s -> interval {frame_interval}")

    idx = 0
    captions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_interval == 0:
            time_sec = round(idx / fps, 2)
            logging.info(f"[FRAME] Extracting @ {time_sec}s (frame {idx})")

            temp_img = f"_frame_{idx}.jpg"
            cv2.imwrite(temp_img, frame)

            with open(temp_img, "rb") as f:
                img_bytes = f.read()
            os.remove(temp_img)

            logging.info("[GEMINI] Captioning frame...")

            contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Describe this frame in 40-50 words (no long paragraphs)."},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_bytes
                            }
                        }
                    ]
                }
            ]

            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents
                )
                caption = resp.text.strip() if resp.text else ""
            except Exception as e:
                caption = f"[Caption failed: {e}]"

            captions.append({
                "time": time_sec,
                "caption": caption
            })

        idx += 1

    cap.release()

    # ---------------------------
    # AFTER FRAME CAPTIONS — NOW EXTRACT AUDIO
    # ---------------------------
    logging.info("[WHISPER] Extracting audio...")

    try:
        wav_file = extract_audio_from_video(path)
        audio = whisper_model.transcribe(wav_file)
        transcript = audio.get("text", "").strip()
        os.remove(wav_file)
    except Exception as e:
        transcript = f"[Audio failed: {e}]"

    # ---------------------------
    # FINAL RESULT
    # ---------------------------
    logging.info("=== VIDEO EXTRACTION FINISHED ===")

    return {
        "success": True,
        "file_name": os.path.basename(path),
        "captions": captions,
        "audio_transcript": transcript
    }


# ---------------------------
# TEST
# ---------------------------
if __name__ == "__main__":
    result = extract_video(r"Data\videoplayback.mp4")

    if result["success"]:
        print("Extraction OK")
        print(result)
    else:
        print("Extraction Failed:", result)
