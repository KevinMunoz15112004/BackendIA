import pandas as pd
from utils import limpiar_batch

df = pd.read_csv("../data/IMDB Dataset SPANISH.csv", encoding='utf-8')

# 1. se verifica las primeras filas
print(df[['review_es', 'sentimiento']].head())

# 2. filtramos solo para obtener la reseña en español y el sentimiento en español
df = df[['review_es', 'sentimiento']].copy()

# 3. se mapean los sentimientos a etiquetas numéricas
df = df[df['sentimiento'].isin(['positivo', 'negativo'])]
df['label'] = df['sentimiento'].map({'negativo': 0, 'positivo': 1})

# 4. Aplicamos la limpieza
df['clean_text'] = limpiar_batch(df['review_es'])

# 5. se guarda el dataset ya limpio
df[['label', 'clean_text']].to_csv('../data/reviews_limpio.csv', index=False)


