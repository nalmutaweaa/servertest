import os, base64, requests
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
import os, base64, requests
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()                                   # reads the .env file on my laptop
app = FastAPI()

app.add_middleware(                             # lets the game (github.io) talk to this server
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

KEY = os.getenv("OPENROUTER_KEY")
MODEL = "nvidia/nemotron-3.5-super-vl:free"

load_dotenv()                                   # reads the .env file on my laptop
app = FastAPI()
KEY = os.getenv("OPENROUTER_KEY")
MODEL = "nvidia/nemotron-3.5-super-vl:free"

@app.post("/hunt")
def hunt(word: str = Form(...), photo: UploadFile = File(...)):
    picture = base64.b64encode(photo.file.read()).decode()      # photo as text for the API
    prompt = "Is there a " + word + " in this photo? Answer only by yes or no."
    reply = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": "Bearer " + KEY},
        json={
            "model": MODEL,
            "reasoning": {"enabled": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + picture}},
            ]}],
        },
        timeout=30,
    )
    print(reply.status_code, reply.text[:200])      # shows what OpenRouter answered
    text = reply.json()["choices"][0]["message"]["content"]
    return {"correct": text.strip().lower().startswith("yes")}





@app.post("/speak")
def speak(language: str = Form(...), recording: UploadFile = File(...)):
    reply = requests.post(
        "https://openrouter.ai/api/v1/audio/transcriptions",
        headers={"Authorization": "Bearer " + KEY},
        files={"file": (recording.filename, recording.file, recording.content_type)},
        data={"model": "openai/whisper-large-v3", "language": language},
        timeout=15,
    )
    print(reply.status_code, reply.text[:200])          # shows what OpenRouter answered
    return {"heard": reply.json()["text"]}

@app.get("/")
def home(): 
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))   # shows the page
