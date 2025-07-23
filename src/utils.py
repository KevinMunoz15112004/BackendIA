import re
import spacy
from tqdm import tqdm

nlp = spacy.load('es_core_news_sm')

NEGACIONES = {"no", "nunca", "jamás", "nadie", "ninguno", "ni", "sin"}

def preprocesar_texto(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def limpiar_texto(texto):
    texto = preprocesar_texto(texto)
    doc = nlp(texto)
    lemas = [
        token.lemma_
        for token in doc
        if (not token.is_stop or token.text in NEGACIONES)
        and token.lemma_ != '-PRON-'
    ]
    return " ".join(lemas)

def limpiar_batch(textos, batch_size=32):
    textos = [preprocesar_texto(t) for t in textos]

    resultados = []
    for doc in tqdm(nlp.pipe(textos, batch_size=batch_size), desc="Limpieza de textos", total=len(textos)):
        lemas = [
            token.lemma_
            for token in doc
            if (not token.is_stop or token.text in NEGACIONES)
            and token.lemma_ != '-PRON-'
        ]
        resultados.append(" ".join(lemas))

    return resultados
