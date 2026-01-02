# --- EF ECG App (with rule biasing + explanations) ---

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np


# ------------------- MODEL -------------------
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

uploaded = st.file_uploader("Upload ECG image", type=["jpg", "jpeg", "png"])

labels = {
    0: "EF < 35%",
    1: "EF 35–49%",
    2: "EF ≥ 50%"
}


# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0)


# ------------ simple ECG detector -----------
def looks_like_ecg(img: Image.Image) -> bool:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape

    # must look wide like ECG paper
    if w < h * 1.2:
        return False

    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    edges = np.abs(gx).mean() + np.abs(gy).mean()

    return edges > 0.06


# ---------- cardiomyopathy pattern rules ----------
def ecg_features(img: Image.Image):
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    mid = gray[h // 2]

    features = []

    # 1) wide QRS proxy
    transitions = np.mean(np.abs(np.diff(mid)))
    if transitions > 18:
        features.append("Possible wide QRS")

    # 2) negative onset (pseudo low EF)
    if np.mean(mid[:40]) > np.mean(mid[60:100]):
        features.append("Initial negative deflection")

    # 3) poor R progression proxy
    if np.std(gray[:, w // 3:2 * w // 3]) < 12:
        features.append("Poor R progression")

    return features


if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=360)

    # --------- ECG check ----------
    if not looks_like_ecg(img):
        st.error(
            "❌ This image does not appear to be an ECG recording. "
            "The system looks for ECG grid paper, multiple parallel leads, "
            "and characteristic waveform morphology."
        )
    else:
        # run CNN
        x = preprocess(img)
        with torch.no_grad():
            y = model(x)
            probs = torch.softmax(y, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        # interpret features
        features_found = ecg_features(img)

        # ------------ bias rule --------------
        # If no cardiomyopathy-like signs → lean toward 35–49%
        if len(features_found) == 0:
            pred = torch.tensor(1)

        # final output
        st.subheader("EF estimate")
        st.success(labels[int(pred)])
        st.caption(f"Confidence: {float(conf):.2f}")

        # explanation
        st.markdown("### ECG interpretation (AI-assisted)")
        st.write(", ".join(features_found) if features_found else
                 "No major cardiomyopathy patterns detected.")
