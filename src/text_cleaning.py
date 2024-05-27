# src/text_cleaning.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Asegúrate de tener descargado el corpus necesario
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def limpiar_texto(text):
    # Eliminar caracteres especiales
    text = re.sub(r'\W', ' ', text)
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Tokenizar el texto
    tokens = nltk.word_tokenize(text)
    
    # Eliminar stopwords y lematizar
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Unir tokens en un solo string
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text
