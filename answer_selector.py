def seleccionar_respuesta(pregunta_clasificada, respuestas):
    """
    Selecciona la respuesta más adecuada basada en la pregunta clasificada.
    """
    # Aquí puedes implementar una lógica para seleccionar la respuesta más relevante.
    # Por simplicidad, retornamos la primera respuesta.
    return respuestas[0] if respuestas else "No se encontró una respuesta adecuada."

# Ejemplo de uso
if __name__ == "__main__":
    pregunta = "¿Cuál es el tratamiento más eficaz para la diabetes?"
    respuestas = ["La insulina es el tratamiento más común.", "Ejercicio regular y dieta balanceada."]
    respuesta = seleccionar_respuesta(pregunta, respuestas)
    print("Respuesta seleccionada:", respuesta)
