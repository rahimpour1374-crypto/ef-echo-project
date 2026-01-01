# VERSION: ECG-DETECTOR-WEIGHTED
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np


# -------- MODEL (EF classifier) ----------
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


# -------- ECG detector (weighted heuristic — بهتر و پایدارتر) --------
def looks_like_ecg(img: Image.Image) -> bool:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape

    score = 0.0

    # افقی بودن تصویر (ECG معمولاً افقی‌تر است)
    if w > h * 1.15:
        score += 0.4

    # کنتراست (نه خیلی صاف، نه خیلی پرنویز)
    score += min(gray.std() * 2, 0.4)

    # لبه‌ها (وجود موج‌ها و خطوط شبکه)
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    edges = np.abs(gx).mean() + np.abs(gy).mean()
    score += min(edges * 2, 0.4)

    # تغییرپذیری افقی (وجود لید/موج تکراری)
    row_var = gray.var(axis=1).mean()
    score += min(row_var * 20, 0.4)

    # تصمیم نهایی — اگر مجموع ویژگی‌ها به اندازه کافی شبیه بود
    return score >= 0.6


if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=350)

    # مرحله ۱: تشخیص ECG
    if not looks_like_ecg(img):
        st.error("❌ این تصویر شبیه نوار قلب نیست.")
    else:
        # مرحله ۲: پیش‌بینی EF
        x = preprocess(img)

        with torch.no_grad():
            y = model(x)
            probs = torch.softmax(y, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        st.subheader("Result:")
        st.success(labels[int(pred)])
        st.caption(f"confidence = {float(conf):.2f}")
