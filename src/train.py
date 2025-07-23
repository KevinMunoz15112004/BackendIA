import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import limpiar_texto

# CARGA DE DATOS
print("Cargando dataset limpio...")
try:
    df = pd.read_csv('../data/reviews_limpio.csv', encoding='utf-8')
except Exception as e:
    raise FileNotFoundError("Error al cargar el dataset limpio. Verifica la ruta.") from e

# VALIDACIÓN DE VALORES NULOS
if df.isnull().values.any():
    raise ValueError("El dataset contiene valores nulos. Revisa los datos antes de entrenar.")

# PREPROCESAMIENTO
X = df['clean_text'].astype(str)
y = df['label'].values

# DIVISIÓN DE DATOS
print("Dividiendo los datos en entrenamiento y prueba")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# VECTORIZACIÓN TF-IDF
print("Vectorizando texto con TF-IDF")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ENTRENAMIENTO DEL MODELO
print("Entrenando modelo con Logistic Regression")
modelo = LogisticRegression(max_iter=200, solver='liblinear')
modelo.fit(X_train_vec, y_train)

# EVALUACIÓN
print("Evaluando modelo")
y_pred = modelo.predict(X_test_vec)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred, digits=4))

# GUARDAR MODELO Y VECTORIZADOR
print("Guardando modelo y vectorizador")
with open('../models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(modelo, f)

with open('../models/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# PRUEBA MANUAL CON FRASES
print("\nPrueba con frases de ejemplo:")
test_phrases = [
    "El producto es excelente y me encantó",
    "Muy decepcionado, no funciona como esperaba",
    "La calidad es pésima, no lo recomiendo",
    "Servicio al cliente fantástico y rápido",
    "No me gustó, llegó dañado y mal embalado",
    "Estoy muy satisfecho con la compra, la recomiendo",
    "Terrible experiencia, no volveré a comprar",
    "Muy buena calidad, vale la pena",
    "El peor producto que he comprado",
    "Cumple con lo que promete, muy bien",
    "Demasiado bueno para ser cierto, ¡me encanta!",
    "Una estafa completa, no lo compren"
]

# LIMPIAMOS EL TEXTO PARA LAS FRASES NUEVAS
test_phrases_clean = [limpiar_texto(p) for p in test_phrases]
test_vec = tfidf.transform(test_phrases_clean)
preds = modelo.predict_proba(test_vec)[:, 1]

for phrase, pred in zip(test_phrases, preds):
    resultado = "Positivo" if pred >= 0.5 else "Negativo"
    print(f"  '{phrase}': {resultado}")

print("\nEntrenamiento y evaluación completados con éxito.")
