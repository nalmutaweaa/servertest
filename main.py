import os, io, ast, base64, requests
import numpy as np
import onnxruntime as ort
from PIL import Image
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


# ---- CARDS: the YOLO model as ONNX (small, runs on the free Render server) ----
card_model = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
card_names = ast.literal_eval(card_model.get_modelmeta().custom_metadata_map["names"])   # {0: 'apple', 1: 'backpack', ...}

def find_cards(image_bytes, conf=0.6):
    "gives the names of the cards the model sees"
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))     # same size as the training pictures
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255            # 1 x 3 x 512 x 512
    if card_model.get_inputs()[0].type == "tensor(float16)":      # the small (half) model wants half numbers
        x = x.astype(np.float16)
    out = card_model.run(None, {card_model.get_inputs()[0].name: x})[0].astype(np.float32)
    if out.shape[-1] == 6:                                  # boxes already cleaned: x1, y1, x2, y2, score, class
        boxes = out[0]
        return sorted({card_names[int(b[5])] for b in boxes if b[4] >= conf})
    scores = out[0][4:].T                                   # older format: 4 box numbers, then one score per card
    best = scores.max(axis=1)
    classes = scores.argmax(axis=1)
    return sorted({card_names[int(c)] for c, s in zip(classes, best) if s >= conf})

@app.post("/card")
async def card(word: str = Form(...), photo: UploadFile = File(...)):
    found = find_cards(await photo.read())
    return {"correct": word in found, "found": found}


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))   # shows the page
