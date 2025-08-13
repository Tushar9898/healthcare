import os
import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import your ML & utility functions
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech

# =====================
# SETUP
# =====================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================
# ROUTES
# =====================
@app.post("/complete-diagnosis")
async def complete_diagnosis(
    audio_file: UploadFile,
    image_file: UploadFile,
    tts_provider: str = Form("gtts")
):
    logger.info("Diagnosis request received.")

    try:
        # Save uploaded audio
        audio_path = os.path.join(UPLOAD_DIR, audio_file.filename)
        with open(audio_path, "wb") as f:
            f.write(await audio_file.read())
        logger.info(f"Audio saved to {os.path.abspath(audio_path)}")

        # Save uploaded image
        image_path = os.path.join(UPLOAD_DIR, image_file.filename)
        with open(image_path, "wb") as f:
            f.write(await image_file.read())
        logger.info(f"Image saved to {os.path.abspath(image_path)}")

        # Transcribe audio
        # Transcribe audio
        transcript = transcribe_with_groq(
                GROQ_API_KEY=GROQ_API_KEY,
                audio_filepath=audio_path,
                stt_model="whisper-large-v3"
        )
        logger.info(f"Transcript: {transcript}")


        # Analyze image with transcript
        diagnosis = analyze_image_with_query(image_path, transcript)
        logger.info(f"Diagnosis: {diagnosis}")

        # Convert diagnosis to speech
        if tts_provider.lower() == "elevenlabs":
            tts_audio_path = text_to_speech(diagnosis)
        else:
            tts_audio_path = text_to_speech(diagnosis)

        logger.info(f"TTS audio generated: {tts_audio_path}")

        return {
            "success": True,
            "transcript": transcript,
            "diagnosis": diagnosis,
            "tts_audio_path": tts_audio_path
        }

    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        return {"success": False, "detail": str(e)}


@app.get("/")
async def root():
    return {"message": "Healthcare AI API is running."}
