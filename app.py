import streamlit as st
import pickle

# 1. Load the saved brains
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vec = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Run train.py first to create the model files!")

# 2. UI Layout
st.title("Sentiment Analysis System")

user_text = st.text_input("Enter your message:", placeholder="Type here...")

if st.button("Analyze"):
    if user_text.strip() != "":
        X_new = vec.transform([user_text])
        prediction = model.predict(X_new)[0]

        st.write("---")
        # 3. Triple check: 1=Pos, 0=Neg, 2=Neutral
        if prediction == 1:
            st.success("Result: POSITIVE 🌟")
        elif prediction == 0:
            st.error("Result: NEGATIVE 🚩")
        elif prediction == 2:
            st.info("Result: NEUTRAL ⚪")
    else:
        st.warning("Please enter some text!")





