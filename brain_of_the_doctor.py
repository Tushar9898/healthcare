# brain_of_the_doctor.py

# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step1: Setup GROQ API key
import os
import base64
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Step2: Convert image to required format
def encode_image(image_path):
    """Encodes an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Step3: Setup Multimodal LLM 
def analyze_image_with_query(query, model, image_path=None, encoded_image=None):
    """
    Analyze an image with a given text query using the specified model.
    You can provide either image_path or encoded_image.
    """
    if not encoded_image and image_path:
        encoded_image = encode_image(image_path)

    # if not encoded_image or not image_path:
    #     raise ValueError("You must provide either image_path or encoded_image.")

    client = Groq(api_key=GROQ_API_KEY)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
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
