# VERSION: ECG-BALANCED
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


# ---- ECG detector (balanced — نه خیلی سفت، نه خیلی شُل) ----
def looks_like_ecg(img: Image.Image) -> bool:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape

    # ECG معمولاً افقی‌تر است
    if w < h * 1.25:
        return False

    # کنتراست (ECG متوسط است، نه خیلی کم)
    if gray.std() < 0.07:
        return False

    # گرادیان ساده (لبه‌ها)
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    edges = np.abs(gx).mean() + np.abs(gy).mean()

    # اگر خیلی صاف باشد → ECG نیست
    if edges < 0.07:
        return False

    # بررسی خطوط افقی (شبکه و موج)
    row_var = gray.var(axis=1).mean()
    if row_var < 0.005:
        return False

    return True


if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=350)

    # 1) ECG detector
    if not looks_like_ecg(img):
        st.error("❌ این تصویر شبیه نوار قلب نیست.")
    else:
        # 2) EF prediction
        x = preprocess(img)

        with torch.no_grad():
            y = model(x)
            probs = torch.softmax(y, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        st.subheader("Result:")
        st.success(labels[int(pred)])
        st.caption(f"confidence = {float(conf):.2f}")
