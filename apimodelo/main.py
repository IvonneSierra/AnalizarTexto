from fastapi import FastAPI, Request, Form
from starlette.templating import Jinja2Templates
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from pysentimiento import create_analyzer
from typing import List,Dict
from data import comments,context



model_name = "pysentimiento/robertuito-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

app = FastAPI()
templates = Jinja2Templates(directory="../apimodelo")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze_sentiment")
async def analyze_sentiment(request: Request, text: str = Form(...)):
    result = classifier(text)
    sentiment = result[0]["label"]
    score = result[0]["score"]
    return templates.TemplateResponse("results.html", {
        "request": request,
        "sentiment": sentiment,
        "score": score,
        "text": text
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

    for comment in comments:
        result = analyzer.predict(comment)
        sentiment = result.output
        score = result.probas[sentiment]
        batch_results.append({"sentiment": sentiment, "score": score, "text": comment})

    return templates.TemplateResponse("/results_comments.html", {
        "request": request,
        "batch_results": batch_results
    })	

