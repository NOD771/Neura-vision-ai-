import streamlit as st
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
import torch

# ✅ Authenticate with Hugging Face
login(token=st.secrets["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"])

st.title("🧠 NeuraVision - AI Image Generator")

prompt = st.text_input("Enter your prompt:", "A futuristic cyberpunk robot")

if st.button("Generate"):
    with st.spinner("Generating image..."):
        pipe = StableDiffusionPipeline.from_pretrained(
            "Wiuhh/Neura",  # make sure this model exists!
            torch_dtype=torch.float32  # CPU-compatible
        ).to("cpu")

        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image")Image")
