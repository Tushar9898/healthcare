# main.py - FastAPI Backend for AI Doctor Application

from dotenv import load_dotenv
load_dotenv()

import os
import base64
import logging
import tempfile
import uuid
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import your existing modules
from groq import Groq
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import subprocess
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Doctor API",
    description="Backend API for AI Doctor with Vision and Voice capabilities",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")

# Validate API keys
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")
if not ELEVENLABS_API_KEY:
    logger.warning("ELEVEN_API_KEY not found. ElevenLabs TTS will not be available")

# System prompt for the doctor
SYSTEM_PROMPT = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

# Pydantic models for request/response
class AudioTranscriptionResponse(BaseModel):
    transcription: str
    success: bool
    message: str

class ImageAnalysisResponse(BaseModel):
    analysis: str
    success: bool
    message: str

class TTSResponse(BaseModel):
    audio_file_path: str
    success: bool
    message: str

class CompleteDiagnosisResponse(BaseModel):
    transcription: str
    doctor_response: str
    audio_file_path: str
    success: bool
    message: str

# Create temp directory for storing files
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)

# === BRAIN OF THE DOCTOR FUNCTIONS ===
def encode_image(image_path: str) -> str:
    """Convert image to base64 format for API consumption"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to encode image: {str(e)}")

def analyze_image_with_query(query: str, encoded_image: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
    """Analyze image using GROQ's multimodal LLM"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ]
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {str(e)}")

# === VOICE OF THE PATIENT FUNCTIONS ===
def transcribe_with_groq(audio_filepath: str, stt_model: str = "whisper-large-v3") -> str:
    """Transcribe audio using GROQ's Whisper model"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

# === VOICE OF THE DOCTOR FUNCTIONS ===
def text_to_speech_with_gtts(input_text: str, output_filepath: str) -> str:
    """Convert text to speech using Google TTS"""
    try:
        language = "en"
        audioobj = gTTS(
            text=input_text,
            lang=language,
            slow=False
        )
        audioobj.save(output_filepath)
        return output_filepath
    except Exception as e:
        logger.error(f"Error with gTTS: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

def text_to_speech_with_elevenlabs(input_text: str, output_filepath: str) -> str:
    """Convert text to speech using ElevenLabs"""
    try:
        if not ELEVENLABS_API_KEY:
            raise HTTPException(status_code=400, detail="ElevenLabs API key not configured")
            
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client.generate(
            text=input_text,
            voice="Aria",
            output_format="mp3_22050_32",
            model="eleven_turbo_v2"
        )
        elevenlabs.save(audio, output_filepath)
        return output_filepath
    except Exception as e:
        logger.error(f"Error with ElevenLabs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech with ElevenLabs: {str(e)}")

# === API ENDPOINTS ===

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Doctor API is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.post("/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe uploaded audio file to text"""
    if not audio_file.filename.endswith(('.mp3', '.wav', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid audio format. Supported: mp3, wav, m4a, ogg")
    
    try:
        # Save uploaded file temporarily
        temp_filename = f"{uuid.uuid4()}_{audio_file.filename}"
        temp_path = TEMP_DIR / temp_filename
        
        with open(temp_path, "wb") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
        
        # Transcribe audio
        transcription = transcribe_with_groq(str(temp_path))
        
        # Clean up temp file
        os.remove(temp_path)
        
        return AudioTranscriptionResponse(
            transcription=transcription,
            success=True,
            message="Audio transcribed successfully"
        )
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_medical_image(
    image_file: UploadFile = File(...),
    query: Optional[str] = Form(None)
):
    """Analyze medical image and provide diagnosis"""
    if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid image format. Supported: jpg, jpeg, png, bmp")
    
    try:
        # Save uploaded file temporarily
        temp_filename = f"{uuid.uuid4()}_{image_file.filename}"
        temp_path = TEMP_DIR / temp_filename
        
        with open(temp_path, "wb") as temp_file:
            content = await image_file.read()
            temp_file.write(content)
        
        # Encode image
        encoded_image = encode_image(str(temp_path))
        
        # Use provided query or default system prompt
        analysis_query = query if query else SYSTEM_PROMPT
        
        # Analyze image
        analysis = analyze_image_with_query(analysis_query, encoded_image)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return ImageAnalysisResponse(
            analysis=analysis,
            success=True,
            message="Image analyzed successfully"
        )
    
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-speech", response_model=TTSResponse)
async def convert_text_to_speech(
    text: str = Form(...),
    tts_provider: str = Form("gtts")  # "gtts" or "elevenlabs"
):
    """Convert text to speech audio"""
    try:
        # Generate unique filename
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = TEMP_DIR / audio_filename
        
        if tts_provider.lower() == "elevenlabs":
            text_to_speech_with_elevenlabs(text, str(audio_path))
        else:  # Default to gTTS
            text_to_speech_with_gtts(text, str(audio_path))
        
        return TTSResponse(
            audio_file_path=f"/download-audio/{audio_filename}",
            success=True,
            message="Text converted to speech successfully"
        )
    
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-audio/{filename}")
async def download_audio(filename: str):
    """Download generated audio file"""
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=filename
    )

@app.post("/complete-diagnosis", response_model=CompleteDiagnosisResponse)
async def complete_medical_diagnosis(
    audio_file: UploadFile = File(...),
    image_file: UploadFile = File(...),
    tts_provider: str = Form("elevenlabs")
):
    """Complete medical diagnosis workflow: transcribe audio + analyze image + generate voice response"""
    
    temp_audio_path = None
    temp_image_path = None
    
    try:
        # Validate file formats
        if not audio_file.filename.endswith(('.mp3', '.wav', '.m4a', '.ogg')):
            raise HTTPException(status_code=400, detail="Invalid audio format")
        
        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Save uploaded files temporarily
        temp_audio_filename = f"{uuid.uuid4()}_{audio_file.filename}"
        temp_audio_path = TEMP_DIR / temp_audio_filename
        
        temp_image_filename = f"{uuid.uuid4()}_{image_file.filename}"
        temp_image_path = TEMP_DIR / temp_image_filename
        
        # Save audio file
        with open(temp_audio_path, "wb") as temp_file:
            audio_content = await audio_file.read()
            temp_file.write(audio_content)
        
        # Save image file
        with open(temp_image_path, "wb") as temp_file:
            image_content = await image_file.read()
            temp_file.write(image_content)
        
        # Step 1: Transcribe audio
        transcription = transcribe_with_groq(str(temp_audio_path))
        
        # Step 2: Analyze image with transcribed query
        encoded_image = encode_image(str(temp_image_path))
        full_query = SYSTEM_PROMPT + transcription
        doctor_response = analyze_image_with_query(full_query, encoded_image)
        
        # Step 3: Convert doctor's response to speech
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = TEMP_DIR / audio_filename
        
        if tts_provider.lower() == "elevenlabs":
            text_to_speech_with_elevenlabs(doctor_response, str(audio_path))
        else:
            text_to_speech_with_gtts(doctor_response, str(audio_path))
        
        # Clean up temp files
        os.remove(temp_audio_path)
        os.remove(temp_image_path)
        
        return CompleteDiagnosisResponse(
            transcription=transcription,
            doctor_response=doctor_response,
            audio_file_path=f"/download-audio/{audio_filename}",
            success=True,
            message="Complete diagnosis completed successfully"
        )
    
    except Exception as e:
        logger.error(f"Complete diagnosis error: {e}")
        # Clean up temp files
        if temp_audio_path and temp_audio_path.exists():
            os.remove(temp_audio_path)
        if temp_image_path and temp_image_path.exists():
            os.remove(temp_image_path)
        raise HTTPException(status_code=500, detail=str(e))

# === UTILITY ENDPOINTS ===

@app.delete("/cleanup-temp-files")
async def cleanup_temp_files():
    """Clean up temporary files (for maintenance)"""
    try:
        deleted_count = 0
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                os.remove(file_path)
                deleted_count += 1
        
        return {
            "message": f"Cleaned up {deleted_count} temporary files",
            "success": True
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Detailed health check with API key validation"""
    health_status = {
        "api_status": "healthy",
        "groq_api": "configured" if GROQ_API_KEY else "missing",
        "elevenlabs_api": "configured" if ELEVENLABS_API_KEY else "missing",
        "temp_directory": "available" if TEMP_DIR.exists() else "missing"
    }
    
    return health_status

# Run the app
if __name__ == "__main__":
    # Make sure temp directory exists
    TEMP_DIR.mkdir(exist_ok=True)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )