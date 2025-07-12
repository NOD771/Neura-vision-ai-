import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import os
import gdown

# Streamlit Page Settings
st.set_page_config(page_title="Neura AI Image Generator", page_icon="🎨", layout="centered")
st.markdown("## 🎨 Neura AI Image Generator")

# Prompt input
prompt = st.text_input("Enter your image prompt")
generate_button = st.button("Generate")

# Google Drive file ID (replace with your actual model file ID)
FILE_ID = "1ErCyGDdmZl8056BiBsfWbDj02zA_sgC-"  # 🔁 Replace this if needed
MODEL_PATH = "model.safetensors"

# Model Loader
@st.cache_resource(show_spinner="🔄 Loading model, please wait...")
def load_model():
    # Download model from Google Drive if not already present
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    
    # Load model
    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Generate Button Click
if generate_button and prompt:
    st.write(f"🎯 Prompt: `{prompt}`")
    pipe = load_model()

    with st.spinner("🛠️ Generating image..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="🖼️ Generated Image", use_column_width=True)
