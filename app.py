import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("🧠 NeuraVision - AI Image Generator")

prompt = st.text_input("Enter your prompt:", "A cyberpunk futuristic city")

if st.button("Generate"):
    with st.spinner("Loading model..."):
        pipe = StableDiffusionPipeline.from_pretrained(
            "Wiuhh/Neura",              # ✅ Correct model name
            torch_dtype=torch.float32   # ❗ use float32 instead of float16 on CPU
        ).to("cpu")                      # ❗ Always use CPU on Streamlit Cloud

        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image")
