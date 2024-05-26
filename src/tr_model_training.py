import os
import joblib
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder


# Función para cargar el modelo guardado
def load_classification_model(model_dir):
    # Cargar el modelo y el tokenizer
    classification_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    classification_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Cargar el label encoder
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    
    return classification_model, classification_tokenizer, label_encoder

def train_classification_model(train_data):
    # Extraer categorías únicas
    categorias = train_data['type'].unique().tolist()
    output_dir ="./results/classifier"

    # Inicializar el tokenizer y el modelo para clasificación
    classification_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    classification_model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_labels=len(categorias))
    
    # Entrenamiento supervisado: convertir etiquetas de categorías a índices
    label_encoder = LabelEncoder()
    label_encoder.fit(categorias)
    
    # Crear dataset para entrenamiento
    train_texts = train_data['subject'].tolist()
    train_labels = label_encoder.transform(train_data['type'].tolist())
    
    # Tokenizar los textos
    train_encodings = classification_tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    
    # Definir dataset de entrenamiento
    class MedicalDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = MedicalDataset(train_encodings, train_labels)

    # Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        save_steps=500,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=0.00005,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=classification_model,
        args=training_args,
        train_dataset=train_dataset
    )

    # Entrenar el modelo
    trainer.train()

    # Guardar el modelo, el tokenizer y el label encoder
    classification_model.save_pretrained(output_dir)
    classification_tokenizer.save_pretrained(output_dir)
    
    # Guardar el label encoder usando joblib
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))

    return classification_model, classification_tokenizer, label_encoder

def classify_question(question, tokenizer, model, label_encoder):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return label_encoder.inverse_transform(predictions.cpu().numpy())
