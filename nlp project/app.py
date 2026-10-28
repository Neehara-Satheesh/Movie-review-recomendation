"""
Frontend for the Movie Review Sentiment Classifier (Streamlit).
Run:  streamlit run app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from movie_review_classifier import clean_text

DATA_PATH = os.path.join("IMDB Dataset.csv", "IMDB Dataset.csv")
if not os.path.isfile(DATA_PATH):
    DATA_PATH = "IMDB Dataset.csv"
MODEL_PATH = "sentiment_model.joblib"

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Training model on the IMDB dataset (first run only)...")
def load_model():
    """Load saved model, or train one and save it for next time."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH).dropna().drop_duplicates()
    df["clean"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5)),
    ])
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    bundle = {"model": model, "accuracy": acc}
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def explain(model, text, top_n=5):
    """Return the words in the review that pushed the prediction most."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    X = vec.transform([text])
    names = np.array(vec.get_feature_names_out())
    contrib = X.toarray()[0] * clf.coef_[0]
    idx = np.nonzero(X.toarray()[0])[0]
    idx = idx[np.argsort(contrib[idx])]
    neg = [(names[i], contrib[i]) for i in idx[:top_n] if contrib[i] < 0]
    pos = [(names[i], contrib[i]) for i in idx[::-1][:top_n] if contrib[i] > 0]
    return pos, neg


# ----------------------------- UI -----------------------------
st.title("🎬 Movie Review Sentiment Classifier")
st.write("Type or paste a movie review and the model will predict whether it "
         "expresses a **positive** or **negative** opinion.")

try:
    bundle = load_model()
except FileNotFoundError:
    st.error("IMDB dataset not found. Keep `IMDB Dataset.csv` in this folder and restart the app.")
    st.stop()

model = bundle["model"]

with st.sidebar:
    st.header("About")
    st.write("**Model:** TF-IDF + Logistic Regression")
    st.write(f"**Test accuracy:** {bundle['accuracy']*100:.1f}%")
    st.write("**Dataset:** IMDB 50K movie reviews")
    st.divider()
    st.caption("Try an example:")
    examples = {
        "Positive example": "A brilliant film with fantastic acting and a gripping story. I loved it!",
        "Negative example": "A boring, predictable movie. Terrible acting and a complete waste of time.",
        "Mixed example": "Not bad, but not great either. It had some good moments.",
    }
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["review"] = text

review = st.text_area("Your review", key="review", height=160,
                      placeholder="e.g. The movie was amazing, I loved the performances...")

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(review)
        proba = model.predict_proba([cleaned])[0]
        p_pos = float(proba[1])
        label = "POSITIVE" if p_pos >= 0.5 else "NEGATIVE"
        conf = max(p_pos, 1 - p_pos)

        if label == "POSITIVE":
            st.success(f"😊 Positive review  ({conf*100:.1f}% confident)")
        else:
            st.error(f"😞 Negative review  ({conf*100:.1f}% confident)")

        st.progress(p_pos, text=f"Positive probability: {p_pos*100:.1f}%")
        if conf < 0.65:
            st.info("The model is unsure. This review may be mixed or neutral.")

        pos_words, neg_words = explain(model, cleaned)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Words pushing positive**")
            st.write(", ".join(w for w, _ in pos_words) or "-")
        with c2:
            st.markdown("**Words pushing negative**")
            st.write(", ".join(w for w, _ in neg_words) or "-")

st.divider()
st.caption("Mini project: Movie Review Classification | NLP")
