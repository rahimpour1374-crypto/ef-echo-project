import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    model = torch.load("ef_model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

st.title("EF Classifier (ECG → EF Group)")

uploaded = st.file_uploader("Upload ECG image", type=["jpg","jpeg","png"])

labels = {
    0: "EF < 35%",
    1: "EF 35–49%",
    2: "EF ≥ 50%"
}

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))     # CHW
    t = torch.tensor(arr).unsqueeze(0)
    return t

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded ECG", width=350)

    x = preprocess(img)

    with torch.no_grad():
        y = model(x)
        pred = int(torch.argmax(y, dim=1)[0])

    st.subheader("Result:")
    st.success(labels[pred])
