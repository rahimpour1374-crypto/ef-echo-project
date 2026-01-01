# VERSION: CALIBRATED + STRONGER ECG FILTER
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np


# -------- MODEL ----------
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


# ----- stronger ECG detector -----
def looks_like_ecg(img: Image.Image) -> bool:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape

    score = 0.0

    # 1) افقی بودن
    if w > h * 1.25:
        score += 0.5

    # 2) خطوط (لبه‌ها)
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    edges = np.abs(gx).mean() + np.abs(gy).mean()
    score += min(edges * 2, 0.5)

    # 3) تکرار افقی لیدها
    row_var = gray.var(axis=1)
    periodic = (row_var > row_var.mean()).mean()
    score += min(periodic * 0.6, 0.6)

    return score >= 0.9


if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=350)

    # ECG detection
    if not looks_like_ecg(img):
        st.error("❌ این تصویر شبیه نوار قلب نیست.")
    else:
        x = preprocess(img)

        with torch.no_grad():
            y = model(x)
            probs = torch.softmax(y, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)
            conf = float(conf)
            pred = int(pred)

        # ---- CALIBRATION RULE ----
        # اگر مدل مطمئن نباشه → پیش‌فرض بذار 35–49
        if conf < 0.55:
            pred = 1

        st.subheader("Result:")
        st.success(labels[pred])
        st.caption(f"confidence (raw model) = {conf:.2f}")
