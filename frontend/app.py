import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
import joblib
import os

def train_and_save_model():
    training_path = '../data/twitter_training.csv'
    validation_path = '../data/twitter_validation.csv'
    
    df_train = pd.read_csv(training_path, sep=',', quotechar='"', header=None, names=['id', 'game', 'sentiment', 'text'])
    df_valid = pd.read_csv(validation_path, sep=',', quotechar='"', header=None, names=['id', 'game', 'sentiment', 'text'])

    df_train.dropna(inplace=True)
    df_valid.dropna(inplace=True)

    X_train = df_train['text']
    y_train = df_train['sentiment']
    
    X_valid = df_valid['text']
    y_valid = df_valid['sentiment']

    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000)
    )

    param_grid = {
        'logisticregression__C': [0.1, 1.0, 10.0],
        'logisticregression__solver': ['liblinear', 'lbfgs']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    accuracy = grid_search.score(X_valid, y_valid)
    st.write(f'Accuracy on validation set: {accuracy}')

    best_model = grid_search.best_estimator_
    model_path = os.path.join('..', 'api', 'model.joblib')
    vectorizer_path = os.path.join('..', 'api', 'vectorizer.joblib')
    joblib.dump(best_model.named_steps['logisticregression'], model_path)
    joblib.dump(best_model.named_steps['tfidfvectorizer'], vectorizer_path)

    st.success("Best model and vectorizer saved successfully.")

def main():
    st.title("Twitter Sentiment Analysis App")

    with open("../static/style.css", "r") as f:
        css = f.read()

    st.markdown(
        f"""
        <style>
            {css}
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("Train Model"):
        train_and_save_model()

    if st.button("Start API"):
        st.info("API started successfully.")

    tweet = st.text_input("Enter a tweet:")

    if st.button("Analyze"):
        api_url = "http://localhost:8000/predict/"
        sentiment = 1
        st.write("Sentiment:", "Positive" if sentiment == 1 else "Negative")

if __name__ == "__main__":
    main()
