import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import io

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/Janus-Pro-7B",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).cuda()
    
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/Janus-Pro-7B")
    return model, tokenizer

def generate_image(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    
    # Extract image from response
    image_bytes = outputs[0].split(b"<|image|>")[-1].split(b"<|endofimage|>")[0]
    image = Image.open(io.BytesIO(image_bytes))
    return image

# Streamlit UI
st.title("Janus-Pro-7B Text-to-Image Demo")
st.write("Basic text-to-image generation using DeepSeek's Janus-Pro-7B")

prompt = st.text_input("Enter your prompt:", "A realistic photo of a baby penguin wearing sunglasses")

if st.button("Generate Image"):
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    
    with st.spinner("Generating image..."):
        try:
            image = generate_image(prompt, model, tokenizer)
            st.image(image, caption=prompt, use_column_width=True)
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
