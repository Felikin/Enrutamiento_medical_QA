import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1

def monitor_performance(y_true, y_pred):
    precision, recall, f1 = calculate_metrics(y_true, y_pred)
    print(f"Precisión: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    return precision, recall, f1

def log_metrics(log_file, precision, recall, f1):
    with open(log_file, 'a') as f:
        f.write(f"Precisión: {precision}, Recall: {recall}, F1-Score: {f1}\n")
