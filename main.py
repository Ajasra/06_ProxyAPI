import logging

from fastapi import FastAPI, Request, UploadFile, File, Form
from openai import OpenAI
import os
from dotenv import load_dotenv

app = FastAPI()

debug = True
# Create a custom logger
logger = logging.getLogger(__name__)

# Set level of logger
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# get api key from environment variable
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

api_config = {
    'transcribe',
    'completion'
}


def print_debug(message):
    if debug:
        print(message)


def return_error(message):
    logger.warning(message)
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
        print_debug(body)
        #  print the ip address of the client
        print_debug(request.client.host)
        if body['api_key'] != LOCAL_API_KEY:
            return return_error("Invalid API Key")
        completion = client.chat.completions.create(
            model=body['model'],
            messages=body['messages']
        )
        print_debug(completion.choices[0].message.content)
        return return_success(completion.choices[0].message.content)
    except Exception as e:
        print(e)
        logger.error(e)
        return return_error(str(e))


# receive multipart form data and transcribe the audio
@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), api_key: str = Form(...)):
    # check if the api_key is valid
    if api_key != LOCAL_API_KEY:
        return return_error("Invalid API Key")

    # check if the file is an audio file (can be mp3 or wav)
    if file.content_type not in ["audio/mpeg", "audio/wav"]:
        return return_error("file is not an audio file")

    try:
        contents = await file.read()
        print_debug(file.filename)
        with open("temp/temp.mp3", "wb") as f:
            f.write(contents)
        audio_file = open("temp/temp.mp3", "rb")
        result = client.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file,
            response_format='text'
        )
        print_debug(result)
        return return_success(result)
    except Exception as e:
        print(e)
        logger.error(e)
        return return_error(str(e))
