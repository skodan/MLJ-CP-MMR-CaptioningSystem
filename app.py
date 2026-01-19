# app.py
import streamlit as st
import requests
import subprocess
import time
from PIL import Image
import io
import base64  # For displaying retrieved images if needed

# Start FastAPI server in background
subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"])
time.sleep(2)  # Wait for server to start

API_BASE = "http://localhost:8001"

st.set_page_config(page_title="Multimodal Retrieval & Captioning", layout="wide")

st.title("Multimodal Retrieval & Captioning System")

# Model selection (add more later)
model_name = st.sidebar.selectbox("Select Model", ["resnet_lstm_attention", "vit_lstm_attention", "vit_transformer"], index=0)

# Common inputs
input_method = st.sidebar.radio("Image Input", ["Upload", "Camera"])
image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"]) if input_method == "Upload" else st.camera_input("Capture Image")
text_input = st.text_input("Text Input")
top_k = st.sidebar.slider("Top K", 1, 10, 5)

# Tabs for tasks
tab_caption, tab_text2img, tab_img2text, tab_img2img, tab_text2text = st.tabs([
    "Image → Caption",
    "Text → Image",
    "Image → Text",
    "Image → Image",
    "Text → Text"
])

with tab_caption:
    if image_file and st.button("Generate Caption"):
        files = {"file": image_file.getvalue()}
        data = {"model_name": model_name}
        resp = requests.post(f"{API_BASE}/caption", files=files, data=data)
        if resp.status_code == 200:
            st.write("Caption:", resp.json()["caption"])
        else:
            st.error("Error: " + resp.text)

with tab_text2img:
    if text_input and st.button("Search Images"):
        data = {"model_name": model_name, "query": text_input, "top_k": top_k}
        resp = requests.post(f"{API_BASE}/search/text2img", data=data)
        if resp.status_code == 200:
            results = resp.json()
            for res in results:
                st.image(res["image_path"], caption=f"Score: {res['score']:.3f}")
        else:
            st.error("Error: " + resp.text)

with tab_img2text:
    if image_file and st.button("Retrieve Text"):
        files = {"file": image_file.getvalue()}
        data = {"model_name": model_name, "top_k": top_k}
        resp = requests.post(f"{API_BASE}/search/img2text", files=files, data=data)
        if resp.status_code == 200:
            st.write("Retrieved Texts:", resp.json())
        else:
            st.error("Error: " + resp.text)

with tab_img2img:
    if image_file and st.button("Retrieve Similar Images"):
        files = {"file": image_file.getvalue()}
        data = {"model_name": model_name, "top_k": top_k}
        resp = requests.post(f"{API_BASE}/search/img2img", files=files, data=data)
        if resp.status_code == 200:
            results = resp.json()
            for res in results:
                st.image(res["image_path"], caption=f"Score: {res['score']:.3f}")
        else:
            st.error("Error: " + resp.text)

with tab_text2text:
    st.info("Text → Text not implemented yet. Add to model interface if needed.")