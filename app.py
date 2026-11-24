
import streamlit as st
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩"
)

st.title("📩 SMS Spam Classification")
st.write("Machine Learning Mini Project")
st.write("Enter a message below to check whether it is Spam or Ham.")

# Load dataset
@st.cache_resource
def train_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].dropna()

    df["text"] = df["text"].astype(str)
    df["target"] = (df["label"] == "spam").astype(int)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("svm", LinearSVC(C=1, random_state=42))
    ])

    model.fit(df["text"], df["target"])
    return model

try:
    model = train_model()

    message = st.text_area(
        "Enter your SMS message:",
        placeholder="Type your message here..."
    )

    if st.button("Predict"):
        if message.strip():
            prediction = model.predict([message])[0]

            if prediction == 1:
                st.error("🚨 This message is SPAM")
            else:
                st.success("✅ This message is HAM (Not Spam)")
        else:
            st.warning("Please enter a message first.")

    st.divider()
    st.subheader("Project Results - Figures")

    figure_folder = "figures"

    if os.path.exists(figure_folder):
        for filename in sorted(os.listdir(figure_folder)):
            if filename.endswith(".png"):
                st.image(
                    os.path.join(figure_folder, filename),
                    caption=filename
                )
    else:
        st.info("Run your Python project first to create the figures.")

except FileNotFoundError:
    st.error("Please keep spam.csv in the same folder as app.py.")
