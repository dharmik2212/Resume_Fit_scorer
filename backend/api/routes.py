"""HTTP endpoints for scoring and in-memory follow-up questions."""
import logging
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from backend.config import HF_API_TOKEN, MAX_FILE_BYTES, MAX_RESUMES, USE_TRANSFORMER_EMBEDDINGS
from backend.core.chat import answer_question
from backend.core.pipeline import run_pipeline
from backend.models.schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.get("/health")
async def health_check():
    return {"status": "ok", "semantic_mode": "sentence-transformer" if USE_TRANSFORMER_EMBEDDINGS else "offline-tfidf", "llm_configured": bool(HF_API_TOKEN)}


@router.post("/score")
async def score_resumes(jd_text: str = Form(...), resumes: list[UploadFile] = File(...)):
    jd_text = jd_text.strip()
    if len(jd_text) < 40:
        raise HTTPException(400, "Please provide a job description of at least 40 characters.")
    if len(jd_text) > 50000:
        raise HTTPException(400, "Job description is too long (maximum 50,000 characters).")
    if not resumes:
        raise HTTPException(400, "Upload at least one resume.")
    if len(resumes) > MAX_RESUMES:
        raise HTTPException(400, f"Maximum {MAX_RESUMES} resumes per analysis.")

    files = []
    for upload in resumes:
        filename = Path(upload.filename or "resume").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type for {filename}. Use PDF, DOCX, or TXT.")
        content = await upload.read(MAX_FILE_BYTES + 1)
        if not content:
            raise HTTPException(400, f"{filename} is empty.")
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{filename} exceeds the 10 MB file limit.")
        files.append((filename, content))

    try:
        result = await run_pipeline(jd_text, files)
        if "error" in result:
            raise HTTPException(422, result["error"])
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Scoring pipeline failed")
        raise HTTPException(500, "The analysis failed unexpectedly. Please verify the files and try again.") from exc


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        return answer_question(request.session_id, request.question)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
