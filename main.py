import os
from src.data_preparation import load_data
from src.tr_model_training import train_classification_model
from src.fr_model import extract_entities
from src.inference import enrutador_de_preguntas
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split


def main():
    # Cargar los datos de entrenamiento
    data_paths = [
        'data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-1.xml',
        'data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-2.xml'
    ]
    train_data = load_data(data_paths)
    train_data = load_data(data_paths)
    train_subset, eval_subset = train_test_split(train_data, test_size=0.2, random_state=42)


    # Entrenar el modelo de clasificación
    classification_model, classification_tokenizer, label_encoder = train_classification_model(train_subset)
    
    # Guardar el modelo de clasificación
    classification_model_path = 'results/classifier'
    classification_model.save_pretrained(classification_model_path)
    classification_tokenizer.save_pretrained(classification_model_path)
        
    # Evaluar el modelo con los datos de test
    classification_report_str = evaluate_model(eval_subset, classification_model_path)

    # Calcular precision, recall y f1-score a partir del reporte de clasificación
    precision = classification_report_str['macro avg']['precision']
    recall = classification_report_str['macro avg']['recall']
    f1 = classification_report_str['macro avg']['f1-score']

    # Monitorizar el rendimiento del modelo
    log_file = 'model_performance.log'
    log_metrics(log_file, precision, recall, f1)

if __name__ == "__main__":
    main()
