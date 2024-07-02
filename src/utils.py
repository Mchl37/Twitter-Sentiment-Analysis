import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Télécharger les stopwords et lemmatiser
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_tweet(tweet):
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)  # Supprimer les mentions
    tweet = re.sub(r'#', '', tweet)  # Supprimer les hashtags
    tweet = re.sub(r'RT : ', '', tweet)  # Supprimer les retweets
    tweet = re.sub(r'https?://[A-Za-z0-9./]+', '', tweet)  # Supprimer les liens
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Supprimer la ponctuation
    tweet = tweet.lower()  # Convertir en minuscule
    tokens = tweet.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)
