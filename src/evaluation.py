import os
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import pipeline

from data_preparation import load_data
from text_cleaning import limpiar_texto
from tr_model_training import load_classification_model, classify_question
from fr_model import load_ner_model, extract_entities

def evaluate_model(test_data_path, classification_model_path):
    # Cargar y preprocesar los datos de test
    test_data = load_data([test_data_path])
    
    # Cargar el modelo de clasificación
    classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_path)
    classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_path)
    
    # Cargar el modelo de NER
    ner_pipeline = load_ner_model()
    
    # Evaluación de clasificación
    y_true = []
    y_pred = []
    
    for index, row in test_data.iterrows():
        question = row['message']
        true_categories = row['type']
        
        predicted_categories = classify_question(question, classification_tokenizer, classification_model)
        
        y_true.append(true_categories)
        y_pred.append(predicted_categories)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Evaluación de NER
    for index, row in test_data.iterrows():
        question = row['message']
        true_entities = row['focus']
        
        predicted_entities = extract_entities(question, ner_pipeline)
        
        print(f"Question: {question}")
        print(f"True Entities: {true_entities}")
        print(f"Predicted Entities: {predicted_entities}")
        print()

def main():
    test_data_path = 'data/TestDataset/TREC-2017-LiveQA-Medical-Test.xml'
    classification_model_path = 'results/classifier'
    
    evaluate_model(test_data_path, classification_model_path)

if __name__ == "__main__":
    main()
