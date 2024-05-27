# Proyecto Bootcamp Inteligencia Artificial: Enrutamiento de Preguntas Médicas

Este proyecto es una implementación de un sistema de Preguntas y Respuestas Médicas utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y Aprendizaje Automático. El objetivo es clasificar preguntas médicas y proporcionar respuestas precisas basadas en categorías y enfoques específicos.

## Componentes del Proyecto

### 1. Acceso a Datos
- **Fuente de Datos:** Los datos utilizados fueron proporcionados en archivos XML. Se descargaron y se procesaron para convertirlos en un DataFrame de pandas.
- **Variables Objetivo:** Las principales variables objetivo son la categoría de la pregunta (`type`) y el enfoque de la pregunta (`focus`).

### 2. Análisis Exploratorio de Datos (EDA)
- **Distribución de Preguntas y Respuestas:** Se realizó un análisis descriptivo para explorar la longitud y frecuencia de las preguntas y respuestas.
- **Visualización:** Se crearon nubes de palabras y gráficos para identificar patrones y temas recurrentes en el dataset.

### 3. Preprocesamiento de Datos
- **Limpieza de Texto:** Eliminación de caracteres especiales y HTML, tokenización y lematización.
- **Vectorización:** Conversión del texto en vectores utilizando TF-IDF o embeddings.

### 4. Selección de Modelo
- **Modelos Evaluados:** Se evaluaron diferentes modelos, incluyendo BERT y TF-IDF + SVM.
- **Modelo Seleccionado:** Se utilizó "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" para la clasificación de preguntas y "ner-disease-ncbi-bionlp-bc5cdr-pubmed" para el reconocimiento de entidades (NER).

### 5. Entrenamiento y Evaluación del Modelo
- **Entrenamiento:** Ajuste de hiperparámetros y entrenamiento del modelo con el conjunto de datos preprocesados.
- **Evaluación:** Uso de métricas como F1-score, recall y precision para evaluar el rendimiento del modelo.

### 6. Implementación y Despliegue
- **API:** Creación de endpoints utilizando Flask para permitir el acceso al modelo.
- **Monitorización:** Implementación de métricas de rendimiento y monitorización continua.

### 7. Feedback Loop
- **Interacción con el Usuario:** Diseño de mecanismos para recolectar retroalimentación del usuario para futuras mejoras del modelo.

## Estructura del Proyecto


Aquí tienes el contenido en formato Markdown listo para ser copiado en tu README.md:

markdown
Copiar código
# Proyecto Bootcamp Inteligencia Artificial: Enrutamiento de Preguntas Médicas

Este proyecto es una implementación de un sistema de Preguntas y Respuestas Médicas utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y Aprendizaje Automático. El objetivo es clasificar preguntas médicas y proporcionar respuestas precisas basadas en categorías y enfoques específicos.

## Componentes del Proyecto

### 1. Acceso a Datos
- **Fuente de Datos:** Los datos utilizados fueron proporcionados en archivos XML. Se descargaron y se procesaron para convertirlos en un DataFrame de pandas.
- **Variables Objetivo:** Las principales variables objetivo son la categoría de la pregunta (`type`) y el enfoque de la pregunta (`focus`).

### 2. Análisis Exploratorio de Datos (EDA)
- **Distribución de Preguntas y Respuestas:** Se realizó un análisis descriptivo para explorar la longitud y frecuencia de las preguntas y respuestas.
- **Visualización:** Se crearon nubes de palabras y gráficos para identificar patrones y temas recurrentes en el dataset.

### 3. Preprocesamiento de Datos
- **Limpieza de Texto:** Eliminación de caracteres especiales y HTML, tokenización y lematización.
- **Vectorización:** Conversión del texto en vectores utilizando TF-IDF o embeddings.

### 4. Selección de Modelo
- **Modelos Evaluados:** Se evaluaron diferentes modelos, incluyendo BERT y TF-IDF + SVM.
- **Modelo Seleccionado:** Se utilizó "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" para la clasificación de preguntas y "ner-disease-ncbi-bionlp-bc5cdr-pubmed" para el reconocimiento de entidades (NER).

### 5. Entrenamiento y Evaluación del Modelo
- **Entrenamiento:** Ajuste de hiperparámetros y entrenamiento del modelo con el conjunto de datos preprocesados.
- **Evaluación:** Uso de métricas como F1-score, recall y precision para evaluar el rendimiento del modelo.

### 6. Implementación y Despliegue
- **API:** Creación de endpoints utilizando Flask para permitir el acceso al modelo.
- **Monitorización:** Implementación de métricas de rendimiento y monitorización continua.

### 7. Feedback Loop
- **Interacción con el Usuario:** Diseño de mecanismos para recolectar retroalimentación del usuario para futuras mejoras del modelo.

## Estructura del Proyecto

Enrutamiento_medical_QA/
├── app/
│ ├── init.py
│ ├── routes.py
│ └── templates/
│ ├── index.html
│ ├── result.html
│ └── thanks.html
├── data/
│ └── medical_data.xml
├── results/
│ └── model_files/
├── notebooks/
│ └── explorations.ipynb
├── scripts/
│ ├── data_preparation.py
│ ├── text_cleaning.py
│ ├── tr_model_training.py
│ ├── fr_model.py
│ ├── inference.py
│ ├── evaluation.py
│ ├── monitoring.py
│ └── feedback_loop.py
├── run.py
├── requirements.txt
└── README.md

## Uso del Producto

### Requisitos Previos
1. Python 3.11
2. Bibliotecas listadas en `requirements.txt`

### Instalación
1. Clonar el repositorio:
    ```sh
    git clone https://github.com/Felikin/Enrutamiento_medical_QA.git
    ```
2. Navegar al directorio del proyecto:
    ```sh
    cd Enrutamiento_medical_QA
    ```
3. Instalar las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

### Ejecución de la Aplicación
1. Iniciar la aplicación Flask:
    ```sh
    python run.py
    ```
2. Abrir el navegador y navegar a `http://127.0.0.1:5000`.

### Uso de la API
- **Endpoint `/ask`:** Permite enviar una pregunta médica y recibir una respuesta clasificada.
    - **Método:** POST
    - **Parámetros:**
        - `question` (string): La pregunta médica a ser clasificada.
    - **Respuesta:** JSON con la categoría, enfoque y respuesta predicha.

### Ejemplo de Uso
1. Enviar una pregunta desde la página principal (`index.html`).
2. Ver la respuesta, categoría y enfoque en la página de resultados (`result.html`).
3. Proporcionar retroalimentación sobre la respuesta recibida.

### Monitorización y Feedback
- **Monitorización:** La aplicación registra métricas de rendimiento (precisión, recall, F1-score) y tiempos de respuesta.
- **Feedback Loop:** Se diseñó un módulo para recolectar retroalimentación del usuario, aunque aún no se ha implementado en la versión actual.

## Conclusiones y Futuras Mejoras
El proyecto ha demostrado ser eficaz en la clasificación de preguntas médicas y en la sugerencia de respuestas. Sin embargo, se identificaron áreas para futuras mejoras, como la implementación completa del feedback loop y la optimización de los modelos de NER y clasificación.

Para más detalles sobre el desarrollo y análisis del proyecto, consulte el [informe del proyecto](https://github.com/Felikin/Enrutamiento_medical_QA/blob/main/Proyecto%20Felipe%20Gonz%C3%A1lez.pdf).

**Referencias:**
- [Repositorio de Datos](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017)
- [Modelo NER](https://huggingface.co/raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed)
- [Modelo Clasificador](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)


---

Este README proporciona una visión general del proyecto, sus componentes y su uso. Para cualquier pregunta o problema, por favor, abra un issue en el repositorio de GitHub.
