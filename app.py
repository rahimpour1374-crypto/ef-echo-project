import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np


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
            nn.Linear(64, 2)      # ← دو کلاس فقط
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


# ------------- ECG DETECTOR -------------
def is_probable_ecg(img: Image.Image) -> bool:
    arr = np.array(img.convert("RGB"))

    h, w, _ = arr.shape
    if w < h * 1.1:          # بیشتر افقی باشد
        return False

    # تشخیص «پس‌زمینه صورتی ECG»
    mean_color = arr.mean(axis=(0,1))
    if mean_color[0] > 190 and mean_color[1] > 150 and mean_color[2] > 160:
        return True

    # شطرنجی (grid)
    gx = np.abs(arr[:,1:] - arr[:,:-1]).mean()
    gy = np.abs(arr[1:,:] - arr[:-1,:]).mean()

    # اگر خطوط افقی/عمودی زیاد باشد → شبیه نوار
    return (gx + gy) > 18



# ------------- EF PREPROCESS -------------
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img).astype("float32") / 255.0
    x = np.transpose(x, (2,0,1))
    return torch.tensor(x).unsqueeze(0)



# ------------- CARDIOMYOPATHY RULES -------------
def cardiomyopathy_hints(img):
    txt = []

    gray = np.array(img.convert("L"))/255.0
    edges = np.abs(gray[:,1:] - gray[:,:-1]).mean()

    # QRS پهن (خیلی ساده)
    if edges > 0.22:
        txt.append("Possible wide QRS")

    # شروع منفی
    if gray.mean() < 0.35:
        txt.append("Initial negative deflection")

    return txt



# ------------- UI -------------
uploaded = st.file_uploader("Upload ECG image", type=["jpg","jpeg","png"])

labels = {
    0: "EF ≤ 35%",
    1: "EF > 35%"
}

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=380)

    # ECG check
    if not is_probable_ecg(img):
        st.error(
            "❌ This image does not appear to be an ECG recording.\n\n"
            "The system looks for: grid paper (often pink), multiple aligned leads, "
            "and repetitive ECG wave morphology."
        )

    else:
        x = preprocess(img)

        with torch.no_grad():
            y = model(x)
            prob = torch.softmax(y, dim=1)[0]
            pred = int(prob.argmax())

        st.subheader("EF estimate")
        st.success(labels[pred])

        st.caption(f"Confidence: {float(prob[pred]):.2f}")

        # AI-assisted explanation
        hints = cardiomyopathy_hints(img)

        st.subheader("ECG interpretation (AI-assisted)")
        if hints:
            for h in hints:
                st.write("• " + h)
        else:
            st.write("• No major cardiomyopathy pattern detected.")
