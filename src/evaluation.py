import os
import sys

# Asegúrate de que el directorio `src` esté en el `PYTHONPATH`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import re
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import pipeline
from sklearn.model_selection import train_test_split

from src.data_preparation import load_data
from src.text_cleaning import limpiar_texto
from src.tr_model_training import load_classification_model, classify_question
from src.fr_model import load_ner_model, extract_entities


def evaluate_model(test_data, classification_model_path):
    # Cargar el modelo de clasificación
    classification_model, classification_tokenizer, label_encoder = load_classification_model(classification_model_path)
    
    # Evaluación de clasificación
    y_true = []
    y_pred = []
    
    for index, row in test_data.iterrows():
        question = row['message']
        true_categories = row['type']
        
        predicted_categories = classify_question(question, classification_tokenizer, classification_model, label_encoder)
        predicted_categories = predicted_categories[0]

        y_true.append(true_categories)
        y_pred.append(predicted_categories)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    classification_report_str = classification_report(y_true, y_pred, output_dict=True)


    # Evaluación de NER
    for index, row in test_data.iterrows():
        question = row['message']
        true_entities = row['focus']
        
        predicted_entities = extract_entities(question)
        if predicted_entities:
            predicted_entities = predicted_entities[0]["word"]
            predicted_entities = re.sub(r'^\s', "", predicted_entities)
        print(f"Question: {question}")
        print(f"True Entities: {true_entities}")
        print(f"Predicted Entities: {predicted_entities}")
        print()
    
    return classification_report_str 
    
def main():
    data_paths = ['data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-1.xml', 'data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-2.xml']
    train_data = load_data(data_paths)
    train_subset, eval_subset = train_test_split(train_data, test_size=0.2, random_state=42)

    classification_model_path = 'results/classifier'
    
    evaluate_model(eval_subset, classification_model_path)

if __name__ == "__main__":
    main()
