# app.py
import streamlit as st
import joblib
import pandas as pd



@st.cache_resource
def load_pipeline(path="text_model.joblib"):
    try:
        pipe = joblib.load(path)
        return pipe
    except Exception as e:
        st.error(f"Cannot load pipeline: {e}")
        return None

pipe = load_pipeline("text_model.joblib")

st.title("Fake News Detector")

st.write("Enter article text below and press **Check**.")

user_text = st.text_area("News text", height=200)

col1, col2 = st.columns([1,3])
with col1:
    btn = st.button("Check")

if btn:
    if pipe is None:
        st.error("Model not loaded. Make sure text_pipe.joblib exists in app folder.")
    elif not user_text or user_text.strip()=="":
        st.warning("Please paste the news text first.")
    else:
        pred = pipe.predict([user_text])[0]
        # Adjust depending on your label mapping (1 = fake, 0 = real)
        if pred == 1:
            st.error("🚨 Predicted: FAKE NEWS")
        else:
            st.success("✅ Predicted: REAL NEWS")

st.markdown("---")
st.write("Try sample news:")
if st.button("Load sample news"):
    samples = [
        "The U.S. Senate voted late Thursday to pass a temporary funding bill, preventing a government shutdown.",
        "NASA confirmed that alien spacecraft were detected entering U.S. airspace near Nevada.",
        "President announced new infrastructure funding to improve highways and airports.",
        "Celebrity XYZ has been cloned and replaced by AI in public appearances, insiders say."
    ]
    for s in samples:
        st.write("—", s)
