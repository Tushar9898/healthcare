import os
import subprocess
import platform
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or "YOUR_API_KEY_HERE"

def text_to_speech(input_text, engine="gtts", output_filepath="final.mp3", autoplay=False):
    """
    Convert text to speech using GTTS or ElevenLabs.
    Args:
        input_text (str): Text to convert to speech.
        engine (str): "gtts" or "elevenlabs".
        output_filepath (str): Output file path (default: final.mp3).
        autoplay (bool): Play the audio after saving.
    """
    if engine.lower() == "gtts":
        language = "en"
        audioobj = gTTS(text=input_text, lang=language, slow=False)
        audioobj.save(output_filepath)

    elif engine.lower() == "elevenlabs":
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client.generate(
            text=input_text,
            voice="Aria",
            output_format="mp3_22050_32",
            model="eleven_turbo_v2"
        )
        elevenlabs.save(audio, output_filepath)

    else:
        raise ValueError("Invalid TTS engine. Use 'gtts' or 'elevenlabs'.")

    if autoplay:
        _play_audio(output_filepath)

    return output_filepath


def _play_audio(filepath):
    """Play audio depending on OS."""
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(['afplay', filepath])
        elif os_name == "Windows":
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{filepath}").PlaySync();'])
        elif os_name == "Linux":
            subprocess.run(['aplay', filepath])
    except Exception as e:
        print(f"Could not play audio: {e}")
