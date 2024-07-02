# Twitter Sentiment Analysis

## Description

Ce projet utilise un jeu de données de tweets pour analyser les sentiments (positif ou négatif) en utilisant des techniques de machine learning.

## Structure du dépôt

- `data/`: Contient les données brutes et nettoyées.
- `notebooks/`: Contient les notebooks Jupyter pour le nettoyage, l'analyse et la modélisation des données.
- `src/`: Contient les scripts Python pour l'application web.
- `static/`: Contient le fichier CSS pour styliser l'application web.
- `templates/`: Contient le fichier HTML pour l'interface utilisateur.
- `Procfile`: Indique à Heroku comment exécuter l'application.
- `requirements.txt`: Liste des dépendances Python.
- `README.md`: Documentation du projet.
- `LICENSE`: Fichier de licence.

## Installation

1. Clonez le dépôt :

   ```sh
   git clone https://github.com/votre-utilisateur/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Installez les dépendances :

   ```sh
   pip install -r requirements.txt
   ```

3. Lancez l'application :
   ```sh
   python src/app.py
   ```

## Déploiement

Pour déployer l'application sur Heroku, suivez ces étapes :

1. Connectez-vous à Heroku :

   ```sh
   heroku login
   ```

2. Créez une nouvelle application :

   ```sh
   heroku create votre-nom-dapplication
   ```

3. Déployez l'application :
   ```sh
   git add .
   git commit -m "Initial commit"
   heroku git:remote -a votre-nom-dapplication
   git push heroku master
   ```

## Membres de l'équipe

- GUELIN Michel
- MARIN Johan
- AOULAD Anass
