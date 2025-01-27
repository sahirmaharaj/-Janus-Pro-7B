import streamlit as st
from PIL import Image
import torch
from janus.modeling_janus import JanusForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer

# Configuration
MODEL_NAME = "deepseek-ai/Janus-Pro-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = JanusForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer)
    
    generation_kwargs = dict(
        inputs=inputs.input_ids,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    with torch.inference_mode():
        output = model.generate(**generation_kwargs)
    
    # Extract image from response
    image_start = tokenizer.decode(output[0]).find("<image>")
    image_end = tokenizer.decode(output[0]).find("</image>")
    image_bytes = output[0][image_start+7:image_end]
    
    return Image.open(io.BytesIO(image_bytes))

# Streamlit UI
st.title("Janus-Pro-7B Multimodal Demo")
st.markdown("Official implementation for [Janus-Pro](https://github.com/deepseek-ai/Janus)")

prompt = st.text_area(
    "Enter your multimodal prompt:",
    "Generate an image of a futuristic cityscape with flying vehicles",
    height=100
)

if st.button("Generate"):
    model, tokenizer = load_model()
    
    with st.spinner("Generating response..."):
        try:
            image = generate_response(prompt, model, tokenizer)
            st.image(image, caption=prompt, use_column_width=True)
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            st.stop()

st.markdown("### Model Details")
st.write("""
- **Model**: Janus-Pro-7B
- **Framework**: PyTorch 2.0.1
- **Capabilities**: Unified multimodal understanding & generation
- **License**: DeepSeek Model License
""")
