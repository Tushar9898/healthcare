import os
import logging
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Mount static files
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System prompt
SYSTEM_PROMPT = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
What's in this image?. Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

@app.post("/complete-diagnosis")
async def complete_diagnosis(
    audio_file: UploadFile,
    image_file: UploadFile,
    tts_provider: str = Form("gtts")
):
    logger.info("Diagnosis request received.")
    
    try:
        # Validate file types
        if not audio_file.filename.endswith(('.mp3', '.wav', '.m4a')):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        if not image_file.filename.endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Invalid image file format")
        
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
        transcript = transcribe_with_groq(
            GROQ_API_KEY=GROQ_API_KEY,
            audio_filepath=audio_path,
            stt_model="whisper-large-v3"
        )
        logger.info(f"Transcript: {transcript}")

        # Analyze image with transcript - FIXED FUNCTION CALL
        diagnosis = analyze_image_with_query(
            query=SYSTEM_PROMPT + transcript,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            image_path=image_path
        )
        logger.info(f"Diagnosis: {diagnosis}")

        # Convert diagnosis to speech - FIXED TTS LOGIC
        tts_filename = f"diagnosis_{audio_file.filename.split('.')[0]}.mp3"
        tts_path = os.path.join(UPLOAD_DIR, tts_filename)
        
        if tts_provider.lower() == "elevenlabs":
            text_to_speech(diagnosis, engine="elevenlabs", output_filepath=tts_path)
        else:
            text_to_speech(diagnosis, engine="gtts", output_filepath=tts_path)
            
        logger.info(f"TTS audio generated: {tts_path}")

        return {
            "success": True,
            "transcript": transcript,
            "diagnosis": diagnosis,
            "tts_audio_url": f"/static/{tts_filename}"  # Return URL instead of path
        }

    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        return {"success": False, "detail": str(e)}

@app.get("/")
async def root():
    return {"message": "Healthcare AI API is running."}