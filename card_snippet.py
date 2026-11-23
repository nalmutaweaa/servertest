# ---- CARDS: the YOLO model as ONNX (small, runs on the free Render server) ----
import io, ast
import numpy as np
import onnxruntime as ort
from PIL import Image

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
