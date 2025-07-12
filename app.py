import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("🎨 AI Image Generator - NeuraVision")

prompt = st.text_input("Enter your prompt:", "A futuristic cyberpunk city at night")

if st.button("Generate"):
    with st.spinner("Generating image..."):
        pipe = StableDiffusionPipeline.from_pretrained(
            "Wiuhh/Neura", 
            torch_dtype=torch.float16,
            revision="fp16"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
