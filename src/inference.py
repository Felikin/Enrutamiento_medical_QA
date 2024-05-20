import sys
import os

# Asegúrate de que el directorio `src` esté en el `PYTHONPATH`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import pipeline
from src.tr_model_training import load_classification_model, train_classification_model, classify_question
#from src.fr_model_training import
from src.data_preparation import load_data


def enrutador_de_preguntas(pregunta, tokenizer_classification, model_classification, train_data, label_encoder):
    # Clasificar la pregunta
    categoria = classify_question(pregunta, tokenizer_classification, model_classification, label_encoder)
    
    # Extraer entidades
#    enfoques = extract_entities(pregunta, tokenizer_ner, model_ner)
    
    # Filtrar respuestas posibles
#    respuestas_posibles = train_data[
#        (train_data['type'] == categoria) & 
#        (train_data['focus'].isin(enfoques))
#    ]['answer'].tolist()
    #respuestas_posibles = train_data[train_data["type"]==categoria]

    return categoria

def main():
    # Definir la ruta del directorio del modelo clasificador
    model_dir = './results/classifier'
    # Cargar los datos de entrenamiento
    data_paths = ['data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-1.xml', 'data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-2.xml']
    train_data = load_data(data_paths)



    # Verificar si el modelo guardado existe
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_files_exist = (
        os.path.exists(os.path.join(model_dir, 'model.safetensors')) and
        os.path.exists(os.path.join(model_dir, 'tokenizer.json')) and
        os.path.exists(os.path.join(model_dir, 'label_encoder.pkl'))
    )

    if not model_files_exist:
        # Si el modelo no está disponible, entrenar el modelo
        print("Modelo no encontrado. Entrenando el modelo...")
        classification_model, classification_tokenizer, label_encoder = train_classification_model(train_data)
    else:
        # Cargar el modelo guardado
        print("Cargando el modelo guardado...")
        classification_model, classification_tokenizer, label_encoder = load_classification_model(model_dir)

    # Entrenar el modelo NER
    #ner_model, ner_tokenizer = train_ner(train_data)

    # Ejemplo de uso del enrutador
    pregunta = "What are the treatment options for migraine?"
    print(train_data)
    print(label_encoder)
    respuesta = enrutador_de_preguntas(pregunta, classification_tokenizer, classification_model, train_data, label_encoder)
    print(f"Pregunta: {pregunta}")
    print(f"Respuestas posibles: {respuesta}")

if __name__ == "__main__":
    main()
