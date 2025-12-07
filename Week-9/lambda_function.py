import base64
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import urllib.request

mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name

def preprocess(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((200,200))
    x = np.array(img).astype("float32") / 255.0
    x = np.transpose(x, (2,0,1))
    x = (x - mean) / std
    x = x.astype(np.float32)
    return x[np.newaxis, :]

def lambda_handler(event, context):
    url = event["url"]

    with urllib.request.urlopen(url) as resp:
        img = Image.open(io.BytesIO(resp.read()))

    x = preprocess(img)
    pred = session.run(None, {input_name: x})[0]

    return {"output": pred.tolist()}
