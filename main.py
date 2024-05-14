from src.data_loader import leer_archivo_xml
from src.text_cleaning import limpiar_texto, lematizar_texto
from src.question_classifier import entrenar_clasificador
from src.answer_selector import seleccionar_respuesta

def main():
    # Leer datos de entrenamiento
    archivo_entrenamiento = 'data/TREC-2017-LiveQA-Medical-Train-1.xml'
    preguntas_respuestas = leer_archivo_xml(archivo_entrenamiento)
    preguntas, respuestas = zip(*preguntas_respuestas)

    # Limpiar y lematizar las preguntas
    preguntas_limpiadas = [lematizar_texto(limpiar_texto(pregunta)) for pregunta in preguntas]

    # Asumimos que las etiquetas ya están disponibles para el entrenamiento (esto debe ajustarse)
    etiquetas = ["Tratamiento", "Causa"] * (len(preguntas) // 2)  # Ejemplo de etiquetas, deben ajustarse

    # Entrenar el clasificador
    clasificador = entrenar_clasificador(preguntas_limpiadas, etiquetas)

    # Ejemplo de clasificación y selección de respuesta
    nueva_pregunta = "¿Cuál es el tratamiento más eficaz para la diabetes?"
    nueva_pregunta_limpiada = lematizar_texto(limpiar_texto(nueva_pregunta))
    tipo_pregunta = clasificador.predict([nueva_pregunta_limpiada])[0]
    print(f"Tipo de pregunta: {tipo_pregunta}")

    respuesta = seleccionar_respuesta(tipo_pregunta, respuestas)
    print(f"Respuesta: {respuesta}")

if __name__ == "__main__":
    main()
