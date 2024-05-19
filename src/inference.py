import sys
import os

# Añadir el directorio del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_training import load_biomedbert_model, classify_question, extract_entities
from src.data_preparation import load_data
import pandas as pd

def enrutador_de_preguntas(pregunta, tokenizer, model, data, categorias):
    categoria_id = classify_question(model, tokenizer, pregunta)
    categoria = categorias[categoria_id]

    # Filtrar respuestas relevantes de la base de datos
    respuestas_relevantes = data[data['type'].apply(lambda x: categoria in x)]
    # Extraer entidades (enfoques) de la pregunta
    enfoques = extract_entities(pregunta, tokenizer, model)
    
    # Filtrar aún más las respuestas por enfoque
    respuestas_filtradas = respuestas_relevantes[respuestas_relevantes['focus'].apply(lambda x: any(focus in enfoques for focus in x))]
    
    # Seleccionar la primera respuesta relevante como ejemplo
    if not respuestas_filtradas.empty:
        respuesta = respuestas_filtradas.iloc[0]['answer_text']
    else:
        respuesta = "No se encontró una respuesta adecuada para esta pregunta."
        
    return respuesta

def main():
    data_paths = [
        'data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-1.xml',
        'data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-2.xml'
    ]
    train_data = load_data(data_paths)
    
    # Extraer categorías únicas
    categorias = train_data["type"].unique()
    # Cargar modelo y tokenizer
    tokenizer, model = load_biomedbert_model(num_labels=len(categorias))
    
    # Ejemplo de uso
    pregunta = "What are the symptoms and treatment options for hypothyroidism?"
    respuesta = enrutador_de_preguntas(pregunta, tokenizer, model, train_data, categorias)
    print(respuesta)
if __name__ == "__main__":
    main()
