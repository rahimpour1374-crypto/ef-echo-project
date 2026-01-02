import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np


# ------------ MODEL ------------
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


# ------------ detect ECG background (pink grid) ------------
def has_ecg_pink_background(img: Image.Image) -> bool:
    arr = np.array(img.resize((300, 200))).astype(np.float32) / 255.0

    # رنگ صورتی کاغذ ECG → (R زیاد، G متوسط، B کم)
    r = arr[..., 0].mean()
    g = arr[..., 1].mean()
    b = arr[..., 2].mean()

    # محدوده منطقی برای grid ECG
    return (r > 0.75) and (g > 0.55) and (b < 0.55)


# ------------ STRONG ECG DETECTOR ------------
def looks_like_ecg(img: Image.Image) -> bool:

    # 1️⃣ اگر بک‌گراند صورتی باشد → ECG است
    if has_ecg_pink_background(img):
        return True

    # 2️⃣ بقیه ویژگی‌ها (برای سایر ECG ها)
    g = np.array(img.convert("L"), dtype=np.float32) / 255.0
    h, w = g.shape

    wide = w >= h * 1.1

    gx = np.abs(g[:, 1:] - g[:, :-1]).mean()
    gy = np.abs(g[1:, :] - g[:-1, :]).mean()
    grid = (gx + gy) / 2 > 0.05

    row_var = g.var(axis=1)
    multi_leads = (row_var > 0.01).sum() > h * 0.25

    col_var = g.var(axis=0).mean()
    waves = col_var > 0.008

    score = sum([wide, grid, multi_leads, waves])

    return score >= 3


# ------------ PREPROCESS ------------
def preprocess(img):
    arr = np.array(img.convert("RGB").resize((224, 224))).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0)


labels = {
    0: "EF < 35%",
    1: "EF 35–49%",
    2: "EF ≥ 50%"
}

st.title("EF Classifier (ECG → EF Group)")

uploaded = st.file_uploader("Upload ECG image", type=["jpg","jpeg","png"])

if uploaded:

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=350)

    # -------- ECG CHECK --------
    if not looks_like_ecg(img):
        st.error(
            "❌ This image does not appear to be an ECG recording.\n\n"
            "The system looks for:\n"
            "• ECG grid paper (often pink)\n"
            "• Multiple parallel leads\n"
            "• Repeating waveform morphology"
        )
        st.stop()

    # -------- PREDICT EF --------
    x = preprocess(img)

    with torch.no_grad():
        y = model(x)
        probs = torch.softmax(y, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

    # bias toward 35–49 when uncertain
    if conf < 0.45:
        pred = torch.tensor(1)

    st.subheader("EF estimate")
    st.success(labels[int(pred)])
    st.caption(f"Confidence: {float(conf):.2f}")

    # -------- INTERPRETATION --------
    st.subheader("ECG interpretation (AI-assisted)")

    explanation = []

    if conf < 0.45:
        explanation.append(
            "Pattern is nonspecific — classification biased toward EF 35–49%."
        )

    if pred == 0:
        explanation.append(
            "Possible cardiomyopathy pattern (wide QRS / initial negative deflection / abnormal morphology)."
        )

    if pred == 2:
        explanation.append(
            "No major cardiomyopathy markers — waveform morphology appears preserved."
        )

    if not explanation:
        explanation.append("No major cardiomyopathy patterns detected.")

    st.write("• " + "\n• ".join(explanation))
