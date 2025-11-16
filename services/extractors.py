# services/extractors.py

import os
import cv2
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# Imports for Gemini
from google import genai
from google.genai import types

# Whisper
import whisper

# PDF / DOCX / Text
import PyPDF2
import docx

# ---------------------------------
# Config
# ---------------------------------
FRAME_EVERY_SECONDS = 100   # <--- YOUR GLOBAL FRAME INTERVAL

logging.basicConfig(level=logging.INFO, format='[INFO] %(message)s')

# ---------------------------------
# Setup
# ---------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

client = genai.Client(api_key=GEMINI_API_KEY)
whisper_model = whisper.load_model("small")

DEFAULT_MODEL = "gemini-2.5-flash"

# ---------------------------------
# Helper functions
# ---------------------------------
def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def mime_type_for_ext(ext: str) -> str:
    ext = ext.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".mp3": "audio/mp3",
        ".m4a": "audio/m4a"
    }.get(ext, "application/octet-stream")

# ---------------------------------
# Audio extraction helper (NEW)
# ---------------------------------
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


# ---------------------------------
# Extractor class
# ---------------------------------
class FileExtractor:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model
        self.whisper = whisper_model

    @staticmethod
    def detect_type(path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".pdf": "pdf",
            ".docx": "docx",
            ".txt": "text",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".bmp": "image",
            ".mp3": "audio",
            ".wav": "audio",
            ".m4a": "audio",
            ".mp4": "video",
            ".mov": "video",
            ".avi": "video"
        }.get(ext, "unknown")

    # -----------------------------
    # IMAGE EXTRACTION
    # -----------------------------
    def extract_image(self, path: str) -> Dict[str, Any]:
        try:
            img_bytes = read_bytes(path)
            mime = mime_type_for_ext(Path(path).suffix)

            resp = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    "Extract all text from this image."
                ]
            )

            text = resp.text.strip() if hasattr(resp, "text") else ""
            return {"success": True, "text": text, "file_name": os.path.basename(path), "file_type": "image"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # AUDIO EXTRACTION (UPDATED)
    # -----------------------------
    def extract_audio(self, path: str) -> Dict[str, Any]:
        try:
            result = self.whisper.transcribe(path)
            text = result.get("text", "").strip()
            return {"success": True, "text": text,
                    "file_name": os.path.basename(path), "file_type": "audio"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # TEXT EXTRACTION
    # -----------------------------
    def extract_text(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                t = f.read().strip()
            return {"success": True, "text": t,
                    "file_name": os.path.basename(path), "file_type": "text"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # PDF EXTRACTION
    # -----------------------------
    def extract_pdf(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = []
                for i, p in enumerate(reader.pages):
                    txt = p.extract_text() or ""
                    if txt.strip():
                        pages.append(f"[Page {i+1}]\n{txt.strip()}")
            return {"success": True, "text": "\n\n".join(pages),
                    "file_name": os.path.basename(path), "file_type": "pdf"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # DOCX EXTRACTION
    # -----------------------------
    def extract_docx(self, path: str) -> Dict[str, Any]:
        try:
            docf = docx.Document(path)
            paras = [p.text.strip() for p in docf.paragraphs if p.text.strip()]
            return {"success": True, "text": "\n\n".join(paras),
                    "file_name": os.path.basename(path), "file_type": "docx"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # VIDEO EXTRACTION (YOUR FINAL CODE)
    # -----------------------------
    def extract_video(self, path: str, frame_seconds: float = FRAME_EVERY_SECONDS) -> Dict[str, Any]:

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
                            {"text": "Describe this frame in 40-50 words."},
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
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=contents
                    )
                    caption = resp.text.strip() if resp.text else ""
                except Exception as e:
                    caption = f"[Caption failed: {e}]"

                captions.append({"time": time_sec, "caption": caption})

            idx += 1

        cap.release()

        # Extract audio
        logging.info("[WHISPER] Extracting audio...")
        try:
            wav_file = extract_audio_from_video(path)
            audio = self.whisper.transcribe(wav_file)
            transcript = audio.get("text", "").strip()
            os.remove(wav_file)
        except Exception as e:
            transcript = f"[Audio failed: {e}]"

        logging.info("=== VIDEO EXTRACTION FINISHED ===")

        return {
            "success": True,
            "file_name": os.path.basename(path),
            "captions": captions,
            "audio_transcript": transcript
        }


    # -----------------------------
    # ROUTER
    # -----------------------------
    def extract(self, path: str, video_fps: float = 1.0) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {"success": False, "error": "File not found"}

        t = self.detect_type(path)

        if t == "image":
            return self.extract_image(path)
        if t == "audio":
            return self.extract_audio(path)
        if t == "text":
            return self.extract_text(path)
        if t == "pdf":
            return self.extract_pdf(path)
        if t == "docx":
            return self.extract_docx(path)
        if t == "video":
            return self.extract_video(path)

        return {"success": False, "error": f"Unsupported type: {t}"}

# ---------------------------------
# Runner
# ---------------------------------
def extract_all(paths: List[str], video_fps: float = 1.0) -> List[Dict[str, Any]]:
    extractor = FileExtractor()
    results = []

    print("\n=== EXTRACTION STARTED ===\n")
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] Processing: {p}")
        res = extractor.extract(p)
        results.append(res)

        if res.get("success"):
            print(f"✅ {res['file_name']} ({res.get('file_type','')})")
        else:
            print(f"❌ {p} → {res.get('error')}")

        print("-" * 60)

    print("\n=== EXTRACTION FINISHED ===\n")
    print("Results Summary:", results)
    return results


if __name__ == "__main__":
    test_paths = [
        r"Data/Premature_graying_of_hair.pdf",
        r"Data/hair.txt",
        r"Data/Screenshot 2025-11-15 142915.png",
        r"Data/harvard.wav",
        r"Data/videoplayback.mp4"
    ]

    extract_all(test_paths)
