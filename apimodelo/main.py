from fastapi import FastAPI
from typing import Union
from transformers import pipeline


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/consultar")
def consultar():
    return {"service_name": "Servicio"}


@app.get("/reconocer")
def face_reckonigtion():
    #aqui debo poner la logica para reconocer rostros
    return {"service_name": "Reconocimiento de rostros"}


classifier = pipeline("sentiment-analysis")

@app.get("/analyze_sentiment")
async def analyze_sentiment(text: str):
    result = classifier(text)
    return {"sentiment": result[0]["label"], "score": result[0]["score"]}