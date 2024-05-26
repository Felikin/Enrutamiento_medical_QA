import sys
import os
import pandas as pd
import re
# Asegúrate de que el directorio `src` esté en el `PYTHONPATH`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tr_model_training import load_classification_model, train_classification_model, classify_question
from src.fr_model import extract_entities
from src.data_preparation import load_data


def enrutador_de_preguntas(pregunta, tokenizer_classification, model_classification, train_data, label_encoder):
    # Clasificar la pregunta
    categoria = classify_question(pregunta, tokenizer_classification, model_classification, label_encoder)
    
    # Extraer entidades
    enfoque = extract_entities(pregunta)
    enfoque = enfoque[0]["word"]
    enfoque = re.sub(r'^\s', "", enfoque)
    # Filtrar respuestas posibles
    if  enfoque and categoria:
        respuestas_posibles = train_data[train_data["focus"]==enfoque]
        if (len(respuestas_posibles[respuestas_posibles["type"]==categoria[0]])>0):
            respuestas_posibles = respuestas_posibles[respuestas_posibles["type"]==categoria[0]]
    elif categoria and not enfoque:
        respuestas_posibles = train_data[train_data["type"]==categoria[0]]
    elif enfoque and not categoria:
        respuestas_posibles = train_data[train_data["focus"]==enfoque]
    else:
        respuestas_posibles = "no tenemos respuestas para sugerirte"
    return respuestas_posibles["answer"]

def main():
    # Definir la ruta del directorio del modelo clasificador
    class_model = './results/classifier'
    # Cargar los datos de entrenamiento
    data_paths = ['data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-1.xml', 'data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-2.xml']
    train_data = load_data(data_paths)

    # Verificar si el modelo guardado existe
    if not os.path.exists(class_model):
        os.makedirs(class_model)
    
    model_files_exist = (
        os.path.exists(os.path.join(class_model, 'model.safetensors')) and
        os.path.exists(os.path.join(class_model, 'tokenizer.json')) and
        os.path.exists(os.path.join(class_model, 'label_encoder.pkl'))
    )

    if not model_files_exist:
        # Si el modelo no está disponible, entrenar el modelo
        print("Modelo no encontrado. Entrenando el modelo...")
        classification_model, classification_tokenizer, label_encoder = train_classification_model(train_data)
    else:
        # Cargar el modelo guardado
        print("Cargando el modelo guardado...")
        classification_model, classification_tokenizer, label_encoder = load_classification_model(class_model)

    # Ejemplo de uso del enrutador
    pregunta = "how many miligrams of ibuprophen per kilogram should I take for fever?"
    respuesta = enrutador_de_preguntas(pregunta, classification_tokenizer, classification_model, train_data, label_encoder)
    print(f"Pregunta: {pregunta}")
    print(f"Respuestas posibles: {respuesta}")

if __name__ == "__main__":
    main()