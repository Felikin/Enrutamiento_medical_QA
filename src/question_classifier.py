from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pickle

def entrenar_clasificador(preguntas, etiquetas):
    """
    Entrena un clasificador para las preguntas médicas.
    """
    X_train, X_test, y_train, y_test = train_test_split(preguntas, etiquetas, test_size=0.2, random_state=42)

    # Pipeline para vectorización y clasificación
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])

    # Entrenamiento del modelo
    pipeline.fit(X_train, y_train)

    # Evaluación del modelo
    precision = pipeline.score(X_test, y_test)
    print(f"Precisión del modelo: {precision}")

    # Guardar el modelo entrenado
    with open('modelos/clasificador_preguntas.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    return pipeline

# Ejemplo de uso
if __name__ == "__main__":
    preguntas = ["¿Cuál es el tratamiento más eficaz para la diabetes?", "¿Qué causa el dolor de cabeza?"]
    etiquetas = ["Tratamiento", "Causa"]
    modelo = entrenar_clasificador(preguntas, etiquetas)
