from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

from transformers.utils import logging
logging.set_verbosity(50)


def load_classification_model(num_labels):
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

def load_ner_model(num_labels):
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def classify_question(model, tokenizer, question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

def extract_entities(question, tokenizer, model):
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    entities = nlp(question)
    return [entity['word'] for entity in entities if entity['entity'].startswith('B')]
