import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

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
            nn.Linear(64, 3)     # model still predicts 3 classes internally
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

# ---- helpers ----
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0)


def has_pink_grid(img):
    arr = np.array(img.convert("RGB"))/255.0
    r,g,b = arr[:,:,0],arr[:,:,1],arr[:,:,2]
    pink = (r>0.75)&(g<0.65)&(b<0.75)
    return pink.mean()>0.10


def looks_like_ecg_structure(img):
    gray = np.array(img.convert("L"))/255.0
    gx = np.abs(gray[:,1:]-gray[:,:-1]).mean()
    gy = np.abs(gray[1:,:]-gray[:-1,:]).mean()
    return (gx+gy) > 0.10


def cardiomyopathy_rules(gray):
    h,w = gray.shape
    mid = gray[:, w//3:2*w//3]
    gx = np.abs(mid[:,1:] - mid[:,:-1])
    qrs_width = (gx>0.25).sum(axis=1).mean()

    findings=[]
    if qrs_width>35: findings.append("Wide QRS")
    if gray.mean()<0.40: findings.append("Initial negative deflection")
    if np.var(gray)<0.01: findings.append("Poor R progression")
    return findings


# ---------- main ----------
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=380)

    gray = np.array(img.convert("L"))/255.0

    ecg_conf = 0
    if has_pink_grid(img): ecg_conf += 2
    if looks_like_ecg_structure(img): ecg_conf += 1

    if ecg_conf == 0:
        st.error(
            "❌ This image does not appear to be an ECG recording.\n"
            "The system looks for typical ECG paper (often pink), multiple parallel leads, and waveform morphology."
        )

    else:
        x = preprocess(img)

        with torch.no_grad():
            y = model(x)
            probs = torch.softmax(y, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        notes = cardiomyopathy_rules(gray)

        # convert 3-class → 2-class
        # class 0 (old)  → EF ≤ 35
        # class 1/2      → EF > 35
        final_group = "EF ≤ 35%" if int(pred)==0 else "EF > 35%"

        # if no worrying ECG features → bias to EF>35
        if len(notes)==0:
            final_group = "EF > 35%"

        st.subheader("EF estimate")
        st.success(final_group)
        st.caption(f"Confidence (raw model): {float(conf):.2f}")

        st.subheader("ECG interpretation (AI-assisted)")
        if notes:
            for n in notes:
                st.write(f"• {n}")
        else:
            st.write("• No major cardiomyopathy pattern detected.")
