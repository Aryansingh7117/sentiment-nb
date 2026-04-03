import pickle
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"
TFIDF_PATH = BASE_DIR / "artifacts" / "tfidf.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


def predict(review: str) -> dict:
    cleaned = clean_text(review)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    return {
        "sentiment": "positive" if pred == 1 else "negative",
        "confidence": round(float(prob.max()), 4),
        "probabilities": {
            "negative": round(float(prob[0]), 4),
            "positive": round(float(prob[1]), 4),
        }
    }


def predict_batch(reviews: list[str]) -> list[dict]:
    return [predict(review) for review in reviews]