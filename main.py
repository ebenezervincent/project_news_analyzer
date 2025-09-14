# app/main.py
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from newspaper import Article
from app.topic_model import extract_topics
from app.sentiment_analysis import analyze_sentiment
from app.bias_classifier import classify_political_bias
from app.api_fetcher import fetch_articles

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleRequest(BaseModel):
    text: str | None = None
    url: str | None = None

def _scrape_with_newspaper(url: str, timeout: int = 7) -> str:
    art = Article(url)
    # reduce default timeouts
    art.config.request_timeout = timeout
    art.download()
    art.parse()
    return (art.text or "").strip()

def _normalize_text(s: str) -> str:
    # collapse whitespace and remove very long runs
    return " ".join((s or "").split())

@app.post("/process/")
async def process_article(req: ArticleRequest):
    # 1) pick source
    if req.text and req.text.strip():
        raw_text = req.text.strip()
    elif req.url and req.url.strip():
        try:
            raw_text = _scrape_with_newspaper(req.url.strip())
        except Exception as e:
            return {"error": f"Failed to extract article from URL: {e}"}
    else:
        return {"error": "Please provide either text or a valid URL."}

    text = _normalize_text(raw_text)

    if len(text) < 200:
        return {
            "keywords": ["news", "world", "article"],
            "sentiment": analyze_sentiment(text),
            "bias": "center",
            "related_articles": [],
            "note": "Input is quite short; topic extraction is less reliable on short texts.",
        }

    # 2) run independent tasks concurrently
    sentiment_task = asyncio.to_thread(analyze_sentiment, text)
    bias_task = asyncio.to_thread(classify_political_bias, text)
    topics_task = asyncio.to_thread(extract_topics, text)

    sentiment, bias, keywords = await asyncio.gather(sentiment_task, bias_task, topics_task)

    # 3) fetch related articles (cached + bounded)
    related = await asyncio.to_thread(fetch_articles, keywords)

    return {
        "keywords": keywords,
        "sentiment": sentiment,
        "bias": bias,
        "related_articles": related[:7],  # trim for UI
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
