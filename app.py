import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------
# 1) ابزار عمومی پردازش تصویر
# ---------------------------

def load_gray(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray


def extract_trace(gray):
    # ساده: خط موج را به عنوان مینیمم پیکسل‌ها می‌گیریم
    # برای ECGهای تمیز surprisingly خوب کار می‌کند
    signal = 255 - gray.mean(axis=0)
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-6)
    return signal


# ---------------------------
# 2) تشخیص baseline + QRS
# ---------------------------

def estimate_baseline(signal):
    win = 40
    smooth = np.convolve(signal, np.ones(win)/win, mode="same")
    return np.median(smooth)


def detect_qrs(signal, baseline):
    diff = np.gradient(signal)
    R = int(np.argmax(signal))

    # Q
    Q = R
    while Q > 1 and signal[Q] > baseline:
        Q -= 1

    # S
    S = R
    while S < len(signal)-1 and signal[S] > baseline:
        S += 1

    q_ms = (S - Q) * 40/1000.0   # assuming 25 mm/s grid

    return Q, R, S, q_ms


# ---------------------------
# 3) features لیدهای V1–V6
# ---------------------------

def measure_R(signal, baseline):
    R = np.max(signal - baseline)
    return float(R)


def rule_prwp(R):
    # R: dict(V1..V6)
    if R["V3"] < 3:
        return 1
    trend = np.diff([R[k] for k in ["V1","V2","V3","V4"]])
    return int(np.mean(trend) < 1.0)


def rule_qrs_wide(qrs_ms):
    return int(qrs_ms >= 120)


def rule_low_voltage(R):
    avg = np.mean(list(R.values()))
    return int(avg < 5)


def rule_pathologic_Q(Q_ms, Q_mm, R_mm):
    return int(Q_ms >= 40 or (R_mm and Q_mm/R_mm >= 0.25))


# ---------------------------
# 4) سیستم هیبرید EF
# ---------------------------

def combine(model_prob, rules):
    # rules score between -2 to +2
    score = 0
    if rules.get("prwp"): score += 1
    if rules.get("wide"): score += 1
    if rules.get("lowvolt"): score += 1
    if rules.get("qpath"): score += 1

    p = model_prob

    # confidence blending (linear)
    hybrid = np.clip(p + 0.08 * score, 0, 1)
    return hybrid


# ---------------------------
# 5) wrapper اصلی
# ---------------------------

def analyze_ecg_image(path, model, device="cpu"):

    gray = load_gray(path)
    signal = extract_trace(gray)
    baseline = estimate_baseline(signal)

    Q,R,S,qrs_ms = detect_qrs(signal, baseline)

    R_amp = measure_R(signal, baseline)

    # فقط دموی لیدها
    leads = {"V1":R_amp, "V2":R_amp*1.2, "V3":R_amp*1.3,
             "V4":R_amp*1.5, "V5":R_amp*1.4, "V6":R_amp*1.2}

    rules = dict(
        prwp = rule_prwp(leads),
        wide = rule_qrs_wide(qrs_ms*1000),
        lowvolt = rule_low_voltage(leads),
        qpath = 0
    )

    # مدل CNN
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    x = torch.tensor(img.transpose(2,0,1)).float()/255.
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1)[0,0].item()  # EF<35 prob

    hybrid = combine(prob, rules)

    return {
        "model_prob": prob,
        "hybrid_prob": hybrid,
        "rules": rules,
        "qrs_ms": qrs_ms*1000,
        "R_amp": R_amp
    }
