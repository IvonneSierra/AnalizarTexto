from fastapi import FastAPI, Request, Form
from starlette.templating import Jinja2Templates
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from pysentimiento import create_analyzer
from data import TEXTS, CONTEXTS

model_name = "pysentimiento/robertuito-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

app = FastAPI()
templates = Jinja2Templates(directory="../apimodelo")

analyzer = create_analyzer("context_hate_speech", lang="es")

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


@app.get("/detect_hate_speech")
async def detect_hate_speech(request: Request):
    analyzer = create_analyzer("context_hate_speech", lang="es")
    results = []
    for text, context in zip(TEXTS, CONTEXTS):
        comments = [text]
        contexts = [context]
        predictions = analyzer.predict(comments, context=contexts)[0]
        
        hate_speech_detected = False
        for pred in predictions['predictions']:
            if pred['label'] == 'hate_speech':
                hate_speech_detected = True
                break
        
        results.append({
            "text": text,
            "context": context,
            "hate_speech_detected": hate_speech_detected
        })
    return templates.TemplateResponse("hate_speech_result.html", {
        "request": request,
        "results": results
    })
