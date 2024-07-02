from fastapi import FastAPI
from pydantic import BaseModel
from api.model import vectorizer, model

app = FastAPI()

class Tweet(BaseModel):
    text: str

@app.post("/predict/")
def predict(tweet: Tweet):
    tweet_vector = vectorizer.transform([tweet.text])
    prediction = model.predict(tweet_vector)
    return {"sentiment": prediction[0]}
