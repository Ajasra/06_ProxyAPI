from fastapi import FastAPI, Request, UploadFile, File, Form
from openai import OpenAI
import os
from dotenv import load_dotenv

app = FastAPI()

# get api key from environment variable
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

api_config = {
    'transcribe',
    'completion'
}


def return_error(message):
    return {
        "code": "400",
        "error": message
    }


def return_success(data):
    return {
        "code": "200",
        "data": data
    }


@app.post("/api/completion")
async def openai_completion(request: Request):
    try:
        body = await request.json()
        if body['api_key'] != LOCAL_API_KEY:
            return return_error("Invalid API Key")
        completion = client.chat.completions.create(
            model=body['model'],
            messages=body['messages']
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return return_error(str(e))


# receive multipart form data and transcribe the audio
@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), api_key: str = Form(...)):
    # check if the api_key is valid
    if api_key != LOCAL_API_KEY:
        return return_error("Invalid API Key")

    # check if the file is an audio file
    if file.content_type != "audio/mpeg":
        return return_error("file is not an audio file")

    try:
        contents = await file.read()
        with open("temp/temp.mp3", "wb") as f:
            f.write(contents)
        audio_file = open("temp/temp.mp3", "rb")
        result = client.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file,
            response_format='text'
        )
        return result
    except Exception as e:
        print(e)
        return return_error(str(e))
