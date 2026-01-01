import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# ---------------- MODEL ARCHITECTURE ----------------
class SimpleEF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    model = SimpleEF()
    state = torch.load("ef_model.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

model = load_model()

# ---------------- UI ----------------
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
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded ECG", width=350)

    x = preprocess(img)

    with torch.no_grad():
        y = model(x)
        pred = int(torch.argmax(y, dim=1)[0])

    st.success(f"EF group: {labels[pred]}")
