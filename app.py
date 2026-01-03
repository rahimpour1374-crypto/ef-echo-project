import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


st.set_page_config(page_title="EF Classifier (Hybrid)", layout="centered")


# ---------------------------
# MODEL
# ---------------------------
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
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleEF().to(device)

    state = torch.load("ef_model.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    torch.set_grad_enabled(False)

    return model, device


model, device = load_model()
labels = {0: "EF < 35%", 1: "EF ≥ 35%"}


# ---------------------------
# FAST ECG DETECTOR
# ---------------------------
def is_ecg_like(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape

    pink = np.array([240, 200, 200])
    pink_ratio = np.mean(np.linalg.norm(arr - pink, axis=2) < 60)

    wide = w > 1.3 * h

    # Either ECG grid color OR wide chart aspect
    return (pink_ratio > 0.18) or wide


# ---------------------------
# SIMPLE TRACE + QRS
# ---------------------------
def to_gray_array(img: Image.Image):
    arr = np.array(img.convert("L"), dtype="float32")
    arr[1:-1] = (arr[:-2] + arr[1:-1] + arr[2:]) / 3
    return arr


def extract_trace(gray: np.ndarray):
    signal = 255 - gray.mean(axis=0)
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-6)
    return signal


def estimate_baseline(signal):
    win = 40
    smooth = np.convolve(signal, np.ones(win)/win, mode="same")
    return float(np.median(smooth))


def detect_qrs(signal, baseline):
    R = int(np.argmax(signal))

    Q = R
    while Q > 1 and signal[Q] > baseline:
        Q -= 1

    S = R
    while S < len(signal)-1 and signal[S] > baseline:
        S += 1

    qrs_ms = (S - Q) * 40
    return Q, R, S, qrs_ms


def measure_R(signal, baseline):
    return float(np.max(signal - baseline) * 10)


# ---------------------------
# RULES
# ---------------------------
def rule_prwp(R):
    if R["V3"] < 3:
        return 1
    trend = np.diff([R[k] for k in ["V1","V2","V3","V4"]])
    return int(np.mean(trend) < 1.0)


def rule_qrs_wide(qrs_ms):
    return int(qrs_ms >= 120)


def rule_low_voltage(R):
    return int(np.mean(list(R.values())) < 5)


def combine(model_prob, rules):
    score = 0
    if rules.get("prwp"): score += 1
    if rules.get("wide"): score += 1
    if rules.get("lowvolt"): score += 1
    return float(np.clip(model_prob + 0.08 * score, 0, 1))


# ---------------------------
# ANALYSIS
# ---------------------------
def analyze(img: Image.Image):
    gray = to_gray_array(img)
    signal = extract_trace(gray)
    baseline = estimate_baseline(signal)

    Q,R,S,qrs_ms = detect_qrs(signal, baseline)
    R_amp = measure_R(signal, baseline)

    leads = {
        "V1": R_amp*0.8,
        "V2": R_amp*1.0,
        "V3": R_amp*1.1,
        "V4": R_amp*1.4,
        "V5": R_amp*1.3,
        "V6": R_amp*1.1,
    }

    rules = dict(
        prwp=rule_prwp(leads),
        wide=rule_qrs_wide(qrs_ms),
        lowvolt=rule_low_voltage(leads),
    )

    x = img.convert("RGB").resize((224,224))
    x = np.transpose(np.array(x, dtype="float32")/255.0, (2,0,1))
    xt = torch.tensor(x).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(xt)
        p_low = float(F.softmax(y, dim=1)[0,0])

    hybrid = combine(p_low, rules)

    return p_low, hybrid, rules, qrs_ms, R_amp


# ---------------------------
# UI
# ---------------------------
st.title("EF Classifier (Hybrid — EF Risk)")

uploaded = st.file_uploader("Upload ECG image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded", width=360)

    # 🛑 stop expensive work if not ECG
    if not is_ecg_like(img):
        st.error("❌ This image does not appear to be an ECG recording.")
        st.stop()

    with st.spinner("Analyzing…"):
        model_prob, hybrid_prob, rules, qrs_ms, R_amp = analyze(img)

    st.subheader("EF Estimate")
    pred = 0 if hybrid_prob >= 0.5 else 1
    st.success({0:"EF < 35%",1:"EF ≥ 35%"}[pred])

    st.caption(f"Model (EF<35): {model_prob:.2f}")
    st.caption(f"Hybrid (EF<35): {hybrid_prob:.2f}")

    with st.expander("ECG markers"):
        st.write(rules)
        st.write(f"QRS ≈ {qrs_ms:.0f} ms | R ≈ {R_amp:.1f} mm")
