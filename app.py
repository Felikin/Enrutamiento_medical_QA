import os
import sys
from flask import Flask, render_template, request

# Asegúrate de que el directorio `src` esté en el `PYTHONPATH`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import enrutador_de_preguntas

app = Flask(__name__)

@app.route('/')
def index():
    return '''
        <form action="/inference" method="post">
            <label for="query">Query:</label><br>
            <input type="text" id="query" name="query"><br>
            <input type="submit" value="Submit">
        </form>
    '''

@app.route('/inference', methods=['POST'])
def inference():
    query = request.form['query']

    # Realizar la inferencia del modelo utilizando tu función de inferencia    
    result = enrutador_de_preguntas(query)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
