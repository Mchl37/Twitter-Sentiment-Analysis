# Twitter Sentiment Analysis

## Description

Ce projet utilise un jeu de données de tweets pour analyser les sentiments (positif ou négatif) en utilisant des techniques de machine learning.

## Structure du dépôt

- `data/`: Contient les données brutes et nettoyées.
- `api/`: Contient les scripts Python pour l'application web.
- `static/`: Contient le fichier CSS pour styliser l'application web.
- `requirements.txt`: Liste des dépendances Python.
- `README.md`: Documentation du projet.

## Installation

1. Clonez le dépôt :

   ```sh
   git clone https://github.com/Mchl37 Twitter-Sentiment-Analysis.git
   cd Twitter-Sentiment-Analysis
   ```

2. Installez les dépendances directement :

   ```sh
   pip install -r requirements.txt
   ```

   ou avec un environnement virtuel :
   (venv)[https://docs.python.org/3/library/venv.html]

   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Lancez l'application :

   ```sh
   uvicorn api.main:app --reload
   ```

   Dans un autre terminal

   ```sh
   streamlit run frontend/app.py
   ```

## Membres de l'équipe

- GUELIN Michel
- MARIN Johan
- AOULAD Anass
