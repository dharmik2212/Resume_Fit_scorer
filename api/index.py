"""Vercel serverless entry point."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

app = FastAPI(title="Resume Fit Scorer", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api", tags=["Resume scoring"])

# Serve frontend files (copied to public/ during build)
public = Path(__file__).resolve().parent.parent / "public"
if public.exists():
    app.mount("/", StaticFiles(directory=str(public), html=True), name="frontend")
