import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_sentiment(text):
    tweet_vector = vectorizer.transform([text])
    prediction = model.predict(tweet_vector)
    return prediction[0]
