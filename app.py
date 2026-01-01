import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

@st.cache_resource
def load_model():
    model = torch.load("ef_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("EF Classifier (ECG → EF Group)")

uploaded = st.file_uploader("Upload ECG image", type=["jpg","jpeg","png"])

labels = {
    0: "EF < 35%",
    1: "EF 35–49%",
    2: "EF ≥ 50%"
}

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded ECG", width=350)

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        y = model(x)
        pred = torch.argmax(y, dim=1).item()

    st.subheader("Result:")
    st.success(labels[pred])
