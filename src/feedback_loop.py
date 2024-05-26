import json

def collect_feedback(question, predicted_categories, predicted_focus, actual_categories, actual_focus):
    feedback = {
        'question': question,
        'predicted_categories': predicted_categories,
        'predicted_focus': predicted_focus,
        'actual_categories': actual_categories,
        'actual_focus': actual_focus
    }
    with open('feedback.json', 'a') as f:
        f.write(json.dumps(feedback) + '\n')

def process_feedback():
    feedback_data = []
    with open('feedback.json', 'r') as f:
        for line in f:
            feedback_data.append(json.loads(line))
    
    # Proceso para ajustar el modelo basado en la retroalimentación
    # (Ejemplo: re-entrenamiento del modelo con los datos corregidos)

    return feedback_data
