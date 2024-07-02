import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_save_model(training_path, validation_path):
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
    model_path = os.path.join('api', 'model.joblib')
    vectorizer_path = os.path.join('api', 'vectorizer.joblib')
    joblib.dump(best_model.named_steps['logisticregression'], model_path)
    joblib.dump(best_model.named_steps['tfidfvectorizer'], vectorizer_path)

    st.success("Best model and vectorizer saved successfully.")

    return df_valid

def plot_sentiment_distribution(data):
    sentiment_counts = data['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=sentiment_counts, x='sentiment', y='count', palette='viridis', ax=ax)
    ax.set_title('Distribution des Sentiments')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Nombre de Tweets')
    
    st.pyplot(fig)

def main():
    st.title("Twitter Sentiment Analysis App")

    # Chargement du fichier CSS si nécessaire
    # with open("static/style.css", "r") as f:
    #     css = f.read()
    # st.markdown(
    #     f"""
    #     <style>
    #         {css}
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    training_path = 'data/twitter_training.csv'
    validation_path = 'data/twitter_validation.csv'

    # Initialiser l'état de la session
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'df_valid' not in st.session_state:
        st.session_state.df_valid = None
    if 'vectorizer_path' not in st.session_state:
        st.session_state.vectorizer_path = None
    if 'model_path' not in st.session_state:
        st.session_state.model_path = None

    # Message initial
    st.info("Cliquez sur le bouton pour entraîner et sauvegarder le modèle.")

    # Bouton pour entraîner le modèle et sauvegarder
    if st.button("Entraîner et sauvegarder le modèle"):
        st.session_state.df_valid = train_and_save_model(training_path, validation_path)
        st.session_state.model_trained = True
        st.session_state.vectorizer_path = os.path.join('api', 'vectorizer.joblib')
        st.session_state.model_path = os.path.join('api', 'model.joblib')

    # Afficher la distribution des sentiments uniquement si le modèle a été entraîné
    if st.session_state.model_trained:
        if st.button("Afficher la distribution des sentiments"):
            plot_sentiment_distribution(st.session_state.df_valid)

    tweet = st.text_input("Enter a tweet:")

    if st.button("Analyze"):
        if tweet:
            # Charger le modèle et le vectorizer
            if st.session_state.model_trained:
                model = joblib.load(st.session_state.model_path)
                vectorizer = joblib.load(st.session_state.vectorizer_path)
                
                # Prétraitement du texte
                text_vectorized = vectorizer.transform([tweet])

                # Prédiction
                sentiment = model.predict(text_vectorized)[0]
                sentiment_label = "Positive" if sentiment == "4" else "Negative"  # assuming 4 is positive and 0 is negative
                st.write("Sentiment:", sentiment_label)
            else:
                st.error("Model not found. Please train the model first.")

if __name__ == "__main__":
    main()
