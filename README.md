# 🎬 Movie Sentiment Analysis API

A machine learning API that classifies movie reviews as **positive** or **negative** using Naive Bayes. Trained on the IMDB 50k movie reviews dataset with 88% accuracy.

---

## 📌 Project Overview

This project builds an end-to-end sentiment analysis pipeline:

- Text preprocessing and EDA on 50,000 IMDB reviews
- TF-IDF vectorization with unigrams and bigrams
- MultinomialNB classifier with alpha tuning via cross validation
- FastAPI service with `/predict` and `/batch_predict` endpoints

---

## 📁 Project Structure

```
sentiment-nb/
│
├── app/
│   ├── __init__.py
│   ├── model.py        # loads artifacts, prediction logic
│   └── main.py         # FastAPI routes
│
├── artifacts/
│   ├── model.pkl       # trained MultinomialNB model
│   └── tfidf.pkl       # fitted TF-IDF vectorizer
│
├── data/
│   └── IMDB Dataset.csv
│
├── notebook/
│   └── 01_explore.ipynb   # EDA, preprocessing, training
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Multinomial Naive Bayes |
| Vectorizer | TF-IDF (unigrams + bigrams) |
| Max features | 50,000 |
| Best alpha | tuned via 5-fold cross validation |
| Test accuracy | **88%** |
| Dataset | IMDB 50k Movie Reviews |
| Class balance | 25k positive / 25k negative |

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/Aryansingh7117/sentiment-nb.git
cd sentiment-nb
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the API**
```bash
uvicorn app.main:app --reload
```

**5. Open the interactive docs**
```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### `GET /`
Health check.

```json
{ "message": "Movie Sentiment API is running" }
```

---

### `POST /predict`
Classify a single review.

**Request:**
```json
{
  "review": "This movie was absolutely brilliant, I loved every minute of it!"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.855,
  "probabilities": {
    "negative": 0.145,
    "positive": 0.855
  }
}
```

---

### `POST /batch_predict`
Classify up to 50 reviews at once.

**Request:**
```json
{
  "reviews": [
    "This movie was absolutely brilliant!",
    "Terrible film, complete waste of time.",
    "It was okay, nothing special."
  ]
}
```

**Response:**
```json
{
  "results": [
    { "sentiment": "positive", "confidence": 0.8723, "probabilities": { "negative": 0.1277, "positive": 0.8723 } },
    { "sentiment": "negative", "confidence": 0.9978, "probabilities": { "negative": 0.9978, "positive": 0.0022 } },
    { "sentiment": "negative", "confidence": 0.6868, "probabilities": { "negative": 0.6868, "positive": 0.3132 } }
  ]
}
```

---

## ⚙️ Requirements

```
scikit-learn==1.6.1
pandas==2.2.3
numpy==2.1.3
fastapi==0.115.12
uvicorn==0.34.0
```

---

## 🔍 Known Limitations

- The model treats each word independently (the "naive" assumption) — it struggles with sarcasm and negations like *"not good"*
- Reviews with mixed sentiment (partly positive, partly negative) tend to get lower confidence scores
- Trained only on movie reviews — may not generalize well to other domains

---

## 👤 Author

**Aryan Singh**  
[GitHub](https://github.com/Aryansingh7117)
