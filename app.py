import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message to check whether it is **Spam or Not Spam**.")

message = st.text_area("SMS Text", height=150)

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0]

        if prediction == 1:
            st.error(f"🚨 Spam (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"✅ Not Spam (Confidence: {prob[0]*100:.2f}%)")
