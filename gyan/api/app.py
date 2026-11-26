"""HTTP API (FastAPI). Run: uvicorn gyan.api.app:app --reload"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..app import engine
from ..models import Card

app = FastAPI(title="Gyan", version="0.1.0")
STATIC = Path(__file__).parent / "static"


class Grade(BaseModel):
    card: Card
    grade: int


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/import")
async def import_export(file: UploadFile, source: str | None = None):
    suffix = Path(file.filename or "x.json").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
    return engine().ingest(tmp.name, source)


@app.get("/ask")
def ask(q: str, k: int = 5):
    return [asdict(h) for h in engine().ask(q, k)]


@app.get("/daily")
def daily(n: int = 5):
    return asdict(engine().daily(n))


@app.post("/review")
def review(g: Grade):
    return engine().review(g.card, g.grade)


@app.get("/quiz")
def quiz(topic: str):
    return engine().quiz(topic)


@app.get("/graph")
def graph(limit: int = 300):
    st = engine().store
    return {"nodes": st.concepts(limit), "edges": st.relations(limit * 3)}


@app.get("/stats")
def stats():
    return engine().stats()


@app.get("/wrapped")
def wrapped(days: int = 7):
    return engine().wrapped(days)
