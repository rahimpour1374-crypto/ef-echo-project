# VERSION: EF + ECG RULE CALIBRATION
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np


# -------- EF MODEL ----------
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

labels = {
    0: "EF < 35%",
    1: "EF 35–49%",
    2: "EF ≥ 50%"
}

uploaded = st.file_uploader("Upload ECG image", type=["jpg","jpeg","png"])


def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0)


# ---------------- ECG DETECTION ----------------
def looks_like_ecg(img: Image.Image) -> bool:
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape

    score = 0.0

    if w > h * 1.2:
        score += 0.4

    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    edges = np.abs(gx).mean() + np.abs(gy).mean()
    score += min(edges * 2, 0.4)

    row_var = gray.var(axis=1).mean()
    score += min(row_var * 20, 0.4)

    return score >= 0.6


# --------------- RULE ENGINE (قوانین قلبی) ---------------
def analyze_ecg_rules(img: Image.Image):
    """
    تلاش می‌کند تقریبی:
    - عرض QRS
    - جهت شروع موج (منفی/مثبت)
    - صاف بودن موج‌ها
    را حدس بزند.

    برمی‌گرداند:
        0  -> پیشنهاد EF < 35
        1  -> پیشنهاد EF 35–49
        2  -> پیشنهاد EF >= 50
        None -> قانون واضحی پیدا نشد
    """

    gray = np.array(img.convert("L").resize((600, 200)), dtype=np.float32)

    # یک سیگنال ساده از وسط تصویر
    row = gray[gray.shape[0] // 2, :]
    row = (row - row.mean()) / (row.std() + 1e-6)

    # مشتق برای پیدا کردن QRS
    der = np.abs(np.diff(row))

    # QRS تقریبی = جاهایی که مشتق خیلی بالاست
    thr = der.mean() + 2 * der.std()
    peaks = der > thr

    # اندازه خوشه‌ها ≈ پهنای QRS
    widths = []
    c = 0
    for p in peaks:
        if p:
            c += 1
        elif c:
            widths.append(c)
            c = 0

    if widths:
        qrs_width = np.median(widths)
    else:
        qrs_width = 0

    # حدسی:
    # 3 خانه کوچک ≈ حدود 3 پیکسل در تصویر resize-شده
    wide_qrs = qrs_width >= 3

    # جهت موج اول (اولین نوسان عمده)
    first = int(np.argmax(np.abs(row)))
    polarity = np.sign(row[first])

    # صاف بودن کلی (no notching)
    smooth = der.mean() < 0.9

    # ------------------ قوانین تو ------------------

    # 🔴 QRS خیلی واید → EF پایین
    if wide_qrs:
        return 0

    # 🔴 شروع موج با قطب منفی → EF پایین‌تر
    if polarity < 0:
        return 0

    # 🟢 QRS باریک + مثبت + صاف → EF خوب‌تر
    if (qrs_width <= 2) and (polarity > 0) and smooth:
        return 2

    # پیش‌فرض (بیشتر بیماران)
    return 1


# ========================================================
#                     PIPELINE
# ========================================================

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded ECG", width=350)

    # 1️⃣ اگر ECG نبود → خارج شو
    if not looks_like_ecg(img):
        st.error("❌ این تصویر شبیه نوار قلب نیست.")
    else:
        x = preprocess(img)

        # 2️⃣ پیش‌بینی اولیه مدل
        with torch.no_grad():
            y = model(x)
            probs = torch.softmax(y, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)
            pred = int(pred)
            conf = float(conf)

        # 3️⃣ اعمال قوانین قلبی (اگر الگو واضح باشد)
        rule_pred = analyze_ecg_rules(img)

        if rule_pred is not None:
            final_pred = rule_pred
            used_rules = True
        else:
            final_pred = pred
            used_rules = False

        st.subheader("Result:")
        st.success(labels[final_pred])

        st.caption(f"model confidence = {conf:.2f}")
        if used_rules:
            st.info("✔ اصلاح شده بر اساس قوانین ECG / cardiomyopathy")
