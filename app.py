import streamlit as st
import torch
from data.dataloader import eng_tokenizer, fr_tokenizer
from train import checkpoint_path, device
from models.transformer import build_model
from inference import translate

@st.cache_resource
def load_model():
    model = build_model(device=device)
    checkpoint = torch.load(checkpoint_path, mmap=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

st.title("English to French Translation")
eng_ip = st.text_area("Enter your english input")
submit = st.button("Translate")

if submit:
    model = load_model()
    fr_op = translate(
        model=model,
        eng_ip=eng_ip,
        eng_tokenizer=eng_tokenizer,
        fr_tokenizer=fr_tokenizer,
        device=device
    )
    
    st.subheader("Translated Output:")
    st.write(fr_op)