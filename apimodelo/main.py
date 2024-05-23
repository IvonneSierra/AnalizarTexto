from fastapi import FastAPI, Request, Form
from starlette.templating import Jinja2Templates
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from pysentimiento import create_analyzer
from data import comments,context, comments2
from fastapi.staticfiles import StaticFiles
from typing import List


model_name = "pysentimiento/robertuito-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

app = FastAPI()
templates = Jinja2Templates(directory="../apimodelo")
app.mount("/static", StaticFiles(directory="static"), name="static")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


from typing import List

@app.post("/analyze_sentiment")
async def analyze_sentiment(request: Request, text: str = Form(...)):
    # Dividir el texto en chunks
    chunk_size = 130  # Ajusta este valor según el límite de tokens de tu modelo
    chunks: List[str] = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Analizar cada chunk por separado
    chunk_results = []
    for chunk in chunks:
        result = classifier(chunk)
        sentiment = result[0]["label"]
        score = result[0]["score"]
        chunk_results.append({"sentiment": sentiment, "score": score, "text": chunk})

    # Combinar los resultados de los chunks
    overall_sentiment = max(chunk_results, key=lambda x: x["score"])["sentiment"]
    overall_score = max(chunk_results, key=lambda x: x["score"])["score"]

    return templates.TemplateResponse("results.html", {
        "request": request,
        "sentiment": overall_sentiment,
        "score": overall_score,
        "text": text,
        "chunk_results": chunk_results
    })

@app.post("/analyze_sentiment_batch")
async def analyze_sentiment_batch(request: Request):
    batch_results = []
    analyzer = create_analyzer(task="context_hate_speech", lang="es")

    for comment, context_text in zip(comments, context):
        result = analyzer.predict(comment, context=context_text)
        sentiment = result.output
        probas = result.probas

        batch_results.append({
            "labels": sentiment,
            "probas": probas,
            "context": context_text,
            "text": comment
        })

    return templates.TemplateResponse("/batch_results.html", {
        "request": request,
        "batch_results": batch_results
    })


@app.post("/analyze_sentiment_comments")
async def analyze_sentiment_batch(request: Request):
    batch_results = []
    analyzer = create_analyzer(task="sentiment", lang="es")

    for comment in comments2:
        result = analyzer.predict(comment)
        sentiment = result.output
        score = result.probas[sentiment]
        batch_results.append({"sentiment": sentiment, "score": score, "text": comment})

    return templates.TemplateResponse("/results_comments.html", {
        "request": request,
        "batch_results": batch_results
    })	

