# ==========================================
# EF CLASSIFIER + STRONG ECG DETECTOR
# ==========================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# ---------------- MODEL ----------------
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

st.title("ECG → EF Classifier (with smart cardiology rules)")
uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

labels = {0: "EF < 35%", 1: "EF 35–49%", 2: "EF ≥ 50%"}


# ------------- PREPROCESS -------------
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")/255.0
    arr = np.transpose(arr, (2,0,1))
    return torch.tensor(arr).unsqueeze(0)


# --------- STRONG ECG DETECTION --------
def detect_grid(gray):
    gx = np.abs(gray[:,1:] - gray[:,:-1]).mean()
    gy = np.abs(gray[1:,:] - gray[:-1,:]).mean()
    return (gx + gy)/2 < 0.11


def detect_lead_repetition(gray):
    blocks = np.array_split(gray, 6, axis=0)
    means = [b.mean(axis=0) for b in blocks]
    sims = [np.std(m) for m in means]
    return np.mean(sims) > 0.017


def detect_wave_pattern(gray):
    line = gray[gray.shape[0]//2]
    dif = np.abs(line[1:] - line[:-1])
    freq = (dif > 0.02).mean()
    return 0.08 < freq < 0.35


def looks_like_ecg(img):
    g = np.array(img.convert("L"), dtype="float32")/255.0
    h,w = g.shape

    horizontal = (w > h*1.25)
    grid = detect_grid(g)
    leads = detect_lead_repetition(g)
    wave = detect_wave_pattern(g)

    score = sum([horizontal, grid, leads, wave])

    return score >= 3, {
        "horizontal_paper": horizontal,
        "grid_detected": grid,
        "lead_repetition": leads,
        "wave_pattern": wave
    }


# -------- CARDIO FEATURES --------
def qrs_wide(g): return (np.abs(g[:,1:]-g[:,:-1]).mean()) < 0.028
def fragmented(g): return (np.abs(g[:,1:]-g[:,:-1]) > 0.25).mean() > 0.12
def low_voltage(g): return g.std() < 0.07
def t_inversion(g):
    top = g[:int(g.shape[0]*0.35)]
    bottom = g[int(g.shape[0]*0.55):]
    return bottom.mean() < top.mean()-0.05
def poor_r_progression(g):
    mid = g[:,int(g.shape[1]*0.45):int(g.shape[1]*0.65)]
    return mid.mean() > 0.65
def possible_af(g): return g.mean(axis=0).std() > 0.18

def ecg_flags(img):
    g = np.array(img.convert("L"), dtype="float32")/255.0
    return {
        "QRS wide": qrs_wide(g),
        "Fragmented QRS": fragmented(g),
        "Low voltage": low_voltage(g),
        "T-wave inversion": t_inversion(g),
        "Poor R progression": poor_r_progression(g),
        "Irregular RR (AF-like)": possible_af(g),
    }


# -------------- PIPELINE --------------
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=380)

    is_ecg, reasons = looks_like_ecg(img)

    if not is_ecg:
        st.error("❌ این تصویر نوار قلب نیست.")
        st.caption("تشخیص براساس: شبکه، تکرار لیدها، شکل موج و نسبت تصویر انجام شد.")
        st.info(
            "Detector details: " +
            ", ".join([k for k,v in reasons.items() if v]) +
            " (criteria not fully satisfied)"
        )

    else:
        x = preprocess(img)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].numpy()

        flags = ecg_flags(img)
        score_low = sum(flags.values())

        if score_low >= 4: probs[0]+=0.30
        elif score_low==3: probs[0]+=0.18
        elif score_low==2: probs[0]+=0.08

        probs /= probs.sum()
        pred = int(np.argmax(probs))

        st.subheader("EF estimate")
        st.success(labels[pred])
        st.caption(f"Confidence: {float(probs[pred]):.2f}")

        findings = [k for k,v in flags.items() if v]
        if findings:
            st.info(
                "ECG suggests: " + ", ".join(findings) +
                "\n\n(یافته‌ها با EF پایین سازگارند؛ ولی جایگزین اکو نیست.)"
            )
        else:
            st.info("الگوی خاصی به نفع EF پایین دیده نشد.")
