from flask import Flask, request, jsonify, render_template
import joblib
from utils import clean_tweet

app = Flask(__name__)

# Charger le modèle et le vectorizer
model = joblib.load('src/sentiment_model.pkl')
vectorizer = joblib.load('src/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data['tweet']
    tweet_clean = clean_tweet(tweet)
    tweet_vectorized = vectorizer.transform([tweet_clean])
    prediction = model.predict(tweet_vectorized)
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
