from flask import Blueprint, render_template, request
from src.inference import enrutador_de_preguntas
import time
import pandas as pd

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/ask', methods=['POST'])
def ask():
    pregunta = request.form['pregunta']
    start_time = time.time()
    
    # Obtén la respuesta del enrutador
    respuesta, categoria, enfoque = enrutador_de_preguntas(pregunta)
    
    end_time = time.time()
    tiempo_respuesta = end_time - start_time
    
    # Almacena las métricas
    with open('metrics.log', 'a') as f:
        f.write(f'Pregunta: {pregunta}, Tiempo de Respuesta: {tiempo_respuesta}\n')

    # Si no hay respuesta o si la respuesta es una serie vacía, manejar adecuadamente
    if respuesta is None or (isinstance(respuesta, pd.Series) and respuesta.empty):
        respuesta = "Lo siento, no pude encontrar una respuesta a tu pregunta."
    else:
        # Si hay múltiples respuestas, tomar la primera
        if isinstance(respuesta, pd.Series):
            respuesta = respuesta.iloc[0]
        else:
            respuesta = respuesta[0]

    return render_template('result.html', pregunta=pregunta, respuesta=respuesta, categoria=categoria, enfoque=enfoque, tiempo_respuesta=tiempo_respuesta)

@main.route('/feedback', methods=['POST'])
def feedback():
    satisfaccion = request.form['satisfaccion']
    
    # Almacena la métrica de satisfacción
    with open('metrics.log', 'a') as f:
        f.write(f'Satisfacción del Usuario: {satisfaccion}\n')
    
    return render_template('thanks.html')
