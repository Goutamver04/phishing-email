"""Optional deployment - Streamlit UI:  streamlit run app.py"""
import streamlit as st
from src.predict import classify

st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣")
st.title("🎣 AI-Driven Phishing Email Detector")
st.caption("NLP (TF-IDF) + URL/sender metadata features")

sender = st.text_input("Sender domain", "paypal-support.xyz")
subject = st.text_input("Subject", "Account suspended")
body = st.text_area("Email body", "Dear Customer, your account has been suspended. Verify your identity immediately: http://paypal-secure-login.xyz/verify", height=200)

if st.button("Analyze"):
    r = classify(subject, sender, body)
    (st.error if r["label"] == "PHISHING" else st.success)(f"{r['label']}  -  phishing probability {r['phishing_probability']:.1%}")
    st.progress(r["phishing_probability"])
    if r["red_flags"]:
        st.subheader("Red flags"); [st.write("• " + f) for f in r["red_flags"]]
