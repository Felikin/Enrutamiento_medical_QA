import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def limpiar_texto(texto):
    """
    Realiza la limpieza de texto eliminando caracteres especiales y convirtiendo el texto a minúsculas.
    """
    texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar caracteres especiales
    texto = texto.lower()  # Convertir el texto a minúsculas
    return texto

def lematizar_texto(texto):
    """
    Realiza la lematización del texto para reducir las palabras a su forma base.
    """
    lemmatizer = WordNetLemmatizer()
    palabras = word_tokenize(texto)
    texto_lematizado = ' '.join([lemmatizer.lemmatize(palabra) for palabra in palabras])
    return texto_lematizado

# Ejemplo de limpieza y lematización de texto
if __name__ == "__main__":
    texto_ejemplo = "¿Cuál es el tratamiento más eficaz para la diabetes?"
    texto_limpio = limpiar_texto(texto_ejemplo)
    texto_lematizado = lematizar_texto(texto_limpio)
    print("Texto limpio:", texto_limpio)
    print("Texto lematizado:", texto_lematizado)
