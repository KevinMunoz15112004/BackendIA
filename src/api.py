from flask import Flask, request, jsonify
import pickle
from utils import limpiar_texto
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, '..', 'models', 'sentiment_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, '..', 'models', 'vectorizer.pkl')

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Modelo o vectorizador no encontrados. Ejecuta train.py primero.")

# Cargar modelo y vectorizador
with open(model_path, 'rb') as f:
    modelo = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    # Limpiamo texto de entrada
    cleaned_text = limpiar_texto(text)
    
    # vectorizar texto
    vect_text = vectorizer.transform([cleaned_text])
    
    # Obtener probabilidad y predicción
    pred_prob = modelo.predict_proba(vect_text)[0][1]
    pred_class = int(pred_prob >= 0.5)
    
    return jsonify({'prediction': pred_class, 'probability': float(pred_prob)})

if __name__ == '__main__':
    app.run(port=5000)
