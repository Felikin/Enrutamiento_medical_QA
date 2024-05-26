import pandas as pd
from src.data_preparation import load_data
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def load_ner_model():
  # Carga del tokenizer y el modelo
  model_name = "raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForTokenClassification.from_pretrained(model_name)

  ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
  return ner_pipeline

def extract_entities(pregunta):
  pipeline = load_ner_model()
  # Aplicación del modelo a cada pregunta
  resultados = []
  entities = pipeline(pregunta, aggregation_strategy= "simple")
  resultados.append(entities)

  matches = []
  for resultado in resultados:
    if not resultado:
         return None
      
    best_match = max(resultado, key=lambda x: float(x["score"]))
    matches.append(best_match)

  return matches  