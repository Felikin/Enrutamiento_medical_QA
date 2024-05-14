import xml.etree.ElementTree as ET

def leer_archivo_xml(archivo):
    """
    Lee un archivo XML y devuelve una lista de tuplas de preguntas y respuestas.
    """
    preguntas_respuestas = []
    tree = ET.parse(archivo)
    root = tree.getroot()
    for child in root:
        pregunta = child.find('Original-Question/MESSAGE').text.strip()
        respuesta = child.find('ReferenceAnswers/ReferenceAnswer/ANSWER').text.strip()
        preguntas_respuestas.append((pregunta, respuesta))
    return preguntas_respuestas

# Ruta al archivo XML de entrenamiento
archivo_entrenamiento = 'data/TrainingDatasets/TREC-2017-LiveQA-Medical-Train-1.xml'

# Leer el archivo XML y obtener las preguntas y respuestas
if __name__ == "__main__":
    preguntas_respuestas = leer_archivo_xml(archivo_entrenamiento)
    # Imprimir las primeras 5 preguntas y respuestas como ejemplo
    for idx, (pregunta, respuesta) in enumerate(preguntas_respuestas[:5], start=1):
        print(f"Pregunta {idx}: {pregunta}")
        print(f"Respuesta {idx}: {respuesta}")
        print("="*50)
