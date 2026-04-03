from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict, predict_batch

app = FastAPI(
    title="Movie Sentiment API",
    description="Naive Bayes sentiment classifier trained on IMDB 50k reviews",
    version="1.0.0"
)


class ReviewRequest(BaseModel):
    review: str


class BatchRequest(BaseModel):
    reviews: list[str]


@app.get("/")
def root():
    return {"message": "Movie Sentiment API is running"}


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review cannot be empty")
    return predict(request.review)


@app.post("/batch_predict")
def predict_batch_sentiment(request: BatchRequest):
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty")
    if len(request.reviews) > 50:
        raise HTTPException(status_code=400, detail="Max 50 reviews per batch")
    return {"results": predict_batch(request.reviews)}