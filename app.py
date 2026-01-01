
import streamlit as st
import torch
from torch import nn
from PIL import Image
import numpy as np

st.set_page_config(page_title="EF Classifier", layout="centered")
st.title("EF Classification from ECG Image")

MODEL_PATH = "ef_model.pt"

@st.cache_resource
def load_model():
    model = nn.Sequential(
        nn.Linear(64*64, 3)
    )
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

st.write("Upload an ECG image (.jpg/.png). The tool predicts EF group:")
st.markdown("""
1) Group 1: EF < 35%  
2) Group 2: 35–49%  
3) Group 3: ≥ 50%
""")

file = st.file_uploader("Upload ECG image:", type=["jpg","jpeg","png"])

def preprocess(img: Image.Image):
    img = img.convert("L").resize((64,64))
    arr = np.array(img).astype("float32")/255.0
    t = torch.tensor(arr).view(1,-1)
    return t

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded ECG", use_container_width=True)
    x = preprocess(img)
    with torch.no_grad():
        out = model(x)
        pred = int(out.argmax(dim=1)[0]) + 1

    label_map = {
        1: "EF < 35%",
        2: "EF 35%–49%",
        3: "EF ≥ 50%"
    }
    st.success(f"Predicted EF Group: {pred} — {label_map[pred]}")
