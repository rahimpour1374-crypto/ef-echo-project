# ==============================
# EF ECG CLASSIFIER + CARDIO RULES
# ==============================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


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


# --------- UI ----------
st.title("EF Classifier (ECG → EF Group)")

uploaded = st.file_uploader("Upload ECG image", type=["jpg", "jpeg", "png"])

labels = {
    0: "EF < 35%",
    1: "EF 35–49%",
    2: "EF ≥ 50%"
}


# -------- PREPROCESS ----------
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0)


# -------- ADVANCED ECG FEATURES (IMAGE APPROX) --------
def qrs_wide(gray):
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    return gx < 0.028


def fragmented(gray):
    diff = np.abs(gray[:, 1:] - gray[:, :-1])
    return (diff > 0.25).mean() > 0.12


def low_voltage(gray):
    return gray.std() < 0.07


def t_inversion(gray):
    top = gray[: int(gray.shape[0] * 0.35)]
    bottom = gray[int(gray.shape[0] * 0.55):]
    return bottom.mean() < top.mean() - 0.05


def poor_r_progression(gray):
    mid = gray[:, int(gray.shape[1] * 0.45): int(gray.shape[1] * 0.65)]
    return mid.mean() > 0.65


def possible_af(gray):
    col_var = gray.mean(axis=0).std()
    return col_var > 0.18


def ecg_flags(img):
    g = np.array(img.convert("L"), dtype="float32") / 255.0

    flags = {
        "QRS_wide": qrs_wide(g),
        "fragmented_QRS": fragmented(g),
        "low_voltage": low_voltage(g),
        "T_inversion": t_inversion(g),
        "poor_R_progression": poor_r_progression(g),
        "AF_like_pattern": possible_af(g),
    }

    return flags


# -------- PREDICTION PIPELINE --------
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded ECG", width=360)

    x = preprocess(img)

    # base model prediction
    with torch.no_grad():
        y = model(x)
        probs = torch.softmax(y, dim=1)[0].numpy()

    # RULE ENGINE
    flags = ecg_flags(img)
    score_low = sum(flags.values())

    if score_low >= 4:
        probs[0] += 0.30
    elif score_low == 3:
        probs[0] += 0.18
    elif score_low == 2:
        probs[0] += 0.08

    probs = probs / probs.sum()
    pred = int(np.argmax(probs))

    # OUTPUT
    st.subheader("Result")
    st.success(labels[pred])

    st.caption(f"Confidence: {float(probs[pred]):.2f}")

    st.info(
        "ECG findings: "
        + (", ".join([k for k, v in flags.items() if v]) or "none detected")
    )
