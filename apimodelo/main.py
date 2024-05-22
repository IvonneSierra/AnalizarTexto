from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from pysentimiento import create_analyzer
from data import context, comments
from typing import List


model_name = "pysentimiento/robertuito-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

app = FastAPI()
templates = Jinja2Templates(directory="../apimodelo")

########

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
async def analyze_sentiment_batch(request: Request, comments: List[str], context: List[str]):
    batch_results = []
    analyzer = create_analyzer(task="sentiment", lang="es")  # Modifica estos valores según tus necesidades
    for comment, context_text in zip(comments, context):
        result = analyzer.predict(comment, context=context_text)
        sentiment = result.output
        score = result.probas[sentiment]
        batch_results.append({"sentiment": sentiment, "score": score, "text": comment})
    return Jinja2Templates("/batch_results.html", {
        "request": request,
        "batch_results": batch_results
    })