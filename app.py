import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


st.set_page_config(page_title="EF Classifier (Hybrid)", layout="centered")


# ---------------------------
# 0) MODEL
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

    try:
        state = torch.load("ef_model.pt", map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        torch.set_grad_enabled(False)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    return model, device


model, device = load_model()
labels = {0: "EF < 35%", 1: "EF ≥ 35%"}


# ---------------------------
# 1) IMAGE → TRACE (no OpenCV)
# ---------------------------
def to_gray_array(img: Image.Image):
    g = img.convert("L")
    arr = np.array(g, dtype="float32")
    # light smoothing
    arr[1:-1] = (arr[:-2] + arr[1:-1] + arr[2:]) / 3.0
    return arr


def extract_trace(gray: np.ndarray):
    # approximate waveform trace from mean across rows
    signal = 255 - gray.mean(axis=0)
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-6)
    return signal


# ---------------------------
# 2) BASELINE + QRS
# ---------------------------
def estimate_baseline(signal):
    win = 40
    smooth = np.convolve(signal, np.ones(win) / win, mode="same")
    return float(np.median(smooth))


def detect_qrs(signal, baseline):
    R = int(np.argmax(signal))

    Q = R
    while Q > 1 and signal[Q] > baseline:
        Q -= 1

    S = R
    while S < len(signal) - 1 and signal[S] > baseline:
        S += 1

    # assuming 25 mm/s; one small box ≈ 40 ms
    qrs_ms = (S - Q) * 40
    return Q, R, S, qrs_ms


def measure_R(signal, baseline):
    return float(np.max(signal - baseline) * 10)  # convert to pseudo-mm scale


# ---------------------------
# 3) RULES (0/1 flags)
# ---------------------------
def rule_prwp(R):
    # R = dict(V1..V6) in mm (approx)
    if R["V3"] < 3:
        return 1
    trend = np.diff([R[k] for k in ["V1", "V2", "V3", "V4"]])
    return int(np.mean(trend) < 1.0)


def rule_qrs_wide(qrs_ms):
    return int(qrs_ms >= 120)


def rule_low_voltage(R):
    avg = np.mean(list(R.values()))
    return int(avg < 5)


def rule_pathologic_Q(Q_ms, Q_mm, R_mm):
    if R_mm <= 0:
        return 0
    return int(Q_ms >= 40 or (Q_mm / R_mm >= 0.25))


# ---------------------------
# 4) HYBRID COMBINER
# ---------------------------
def combine(model_prob, rules):
    score = 0
    if rules.get("prwp"): score += 1
    if rules.get("wide"): score += 1
    if rules.get("lowvolt"): score += 1
    if rules.get("qpath"): score += 1

    # gentle correction toward EF<35 if risk markers exist
    hybrid = float(np.clip(model_prob + 0.08 * score, 0, 1))
    return hybrid


# ---------------------------
# 5) MAIN ANALYSIS
# ---------------------------
def analyze(img: Image.Image):
    gray = to_gray_array(img)
    signal = extract_trace(gray)
    baseline = estimate_baseline(signal)

    Q, R, S, qrs_ms = detect_qrs(signal, baseline)
    R_amp = measure_R(signal, baseline)

    # approximate chest-lead R heights (proxy — improved later)
    leads = {
        "V1": R_amp * 0.8,
        "V2": R_amp * 1.0,
        "V3": R_amp * 1.1,
        "V4": R_amp * 1.4,
        "V5": R_amp * 1.3,
        "V6": R_amp * 1.1,
    }

    rules = dict(
        prwp=rule_prwp(leads),
        wide=rule_qrs_wide(qrs_ms),
        lowvolt=rule_low_voltage(leads),
        qpath=0,   # placeholder until we refine Q-wave extraction per-lead
    )

    # CNN probability
    x = img.convert("RGB").resize((224, 224))
    x = np.array(x, dtype="float32") / 255.0
    x = np.transpose(x, (2, 0, 1))
    xt = torch.tensor(x).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(xt)
        probs = F.softmax(y, dim=1)[0]
        p_low = float(probs[0])  # EF < 35%

    hybrid = combine(p_low, rules)

    return {
        "model_prob": p_low,
        "hybrid_prob": hybrid,
        "rules": rules,
        "qrs_ms": float(qrs_ms),
        "R_amp": float(R_amp),
    }


# ---------------------------
# UI
# ---------------------------
st.title("EF Classifier (Hybrid — ECG Image → EF Risk)")

uploaded = st.file_uploader("Upload ECG image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded", width=360)

    with st.spinner("Analyzing…"):
        res = analyze(img)

    st.subheader("EF Estimate")
    pred = 0 if res["hybrid_prob"] >= 0.5 else 1
    st.success(labels[pred])

    st.caption(f"Model (EF<35): {res['model_prob']:.2f}")
    st.caption(f"Hybrid (EF<35): {res['hybrid_prob']:.2f}")

    with st.expander("ECG markers (rules)"):
        st.write(res["rules"])
        st.write(f"QRS duration ≈ {res['qrs_ms']:.0f} ms")
        st.write(f"R amplitude (approx) ≈ {res['R_amp']:.1f} mm")

    if res["hybrid_prob"] < 0.55:
        st.info("Uncertainty is moderate — interpret alongside clinical judgment.")
