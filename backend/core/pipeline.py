"""Orchestrate parsing, cheap filtering, semantic ranking, and deep scoring."""
import logging
import time
import uuid
from pathlib import Path
from backend.config import WEIGHT_BM25, WEIGHT_SEMANTIC, WEIGHT_LLM
from backend.core.chat import store_session
from backend.core.parser import extract_candidate_name, extract_text, parse_resume_sections
from backend.core.stage1_bm25 import score_resumes_bm25
from backend.core.stage2_embeddings import score_resumes_semantic
from backend.core.stage3_llm import _local_score, estimate_cost_usd, score_resumes_llm

logger = logging.getLogger(__name__)


def _compute_final_score(resume: dict) -> float:
    values = [(WEIGHT_BM25, resume.get("bm25_score"))]
    if resume.get("semantic_score") is not None:
        values.append((WEIGHT_SEMANTIC, resume["semantic_score"] * 100))
    if resume.get("llm_score") is not None:
        values.append((WEIGHT_LLM, resume["llm_score"]))
    available = [(weight, value) for weight, value in values if value is not None]
    denominator = sum(weight for weight, _ in available) or 1
    return round(sum(weight * value for weight, value in available) / denominator, 2)


def _enrich_without_deep_score(jd_text: str, resume: dict, stage: int) -> dict:
    enriched = _local_score(jd_text, resume)
    enriched["llm_score"] = None
    enriched["stage_reached"] = stage
    enriched["scoring_source"] = "local evidence"
    return enriched


async def run_pipeline(jd_text: str, resume_files: list[tuple[str, bytes]]) -> dict:
    started = time.perf_counter()
    parsed = []
    parse_warnings = []
    for filename, content in resume_files:
        text = extract_text(content, filename)
        if not text:
            parse_warnings.append(f"Could not extract readable text from {filename}.")
            continue
        name = extract_candidate_name(text)
        if name == "Unknown Candidate":
            name = Path(filename).stem.replace("_", " ").replace("-", " ").strip().title() or name
        sections = parse_resume_sections(text)
        parsed.append({"filename": filename, "text": text, "name": name, "stage_reached": 1, "sections": sections})
    if not parsed:
        return {"error": "No readable text was found. Upload a text-based PDF, DOCX, or TXT resume."}

    stage1 = score_resumes_bm25(jd_text, parsed)
    if not stage1:
        stage1 = [max(parsed, key=lambda item: item.get("bm25_score", 0))]
    stage1_names = {id(item) for item in stage1}
    filtered_stage1 = [item for item in parsed if id(item) not in stage1_names]

    stage2 = score_resumes_semantic(jd_text, stage1)
    if not stage2:
        stage2 = [max(stage1, key=lambda item: (item.get("semantic_score", 0), item.get("bm25_score", 0)))]
    stage2_names = {id(item) for item in stage2}
    filtered_stage2 = [item for item in stage1 if id(item) not in stage2_names]

    deep_scored = await score_resumes_llm(jd_text, stage2)
    for item in deep_scored:
        item["stage_reached"] = 3
    all_results = list(deep_scored)
    all_results += [_enrich_without_deep_score(jd_text, item, 2) for item in filtered_stage2]
    all_results += [_enrich_without_deep_score(jd_text, item, 1) for item in filtered_stage1]

    for item in all_results:
        item["final_score"] = _compute_final_score(item)
    all_results.sort(key=lambda item: item["final_score"], reverse=True)

    session_id = uuid.uuid4().hex
    ranked = []
    chat_candidates = []
    for rank, item in enumerate(all_results, 1):
        public = {
            "rank": rank, "filename": item["filename"], "candidate_name": item.get("name", "Unknown Candidate"),
            "final_score": item["final_score"], "bm25_score": item.get("bm25_score", 0),
            "semantic_score": item.get("semantic_score"), "llm_score": item.get("llm_score"),
            "matched_skills": item.get("matched_skills", []), "missing_skills": item.get("missing_skills", []),
            "experience_relevance": item.get("experience_relevance", 0), "breakdown": item.get("breakdown"),
            "reason": item.get("reason", ""), "stage_reached": item.get("stage_reached", 1),
            "scoring_source": item.get("scoring_source", "local"),
            "role_axes": item.get("role_axes"),
            "education_label": item.get("education_label", "Not specified"),
            "education_detail": item.get("education_detail", ""),
            "companies": item.get("companies", []),
            "company_count": item.get("company_count", 0),
        }
        ranked.append(public)
        chat_candidates.append({**public, "text": item["text"], "sections": item.get("sections", {})})
    store_session(session_id, jd_text, chat_candidates)

    elapsed = round(time.perf_counter() - started, 2)
    return {
        "session_id": session_id,
        "jd_summary": " ".join(jd_text.split())[:240],
        "total_resumes": len(resume_files),
        "processed_time_seconds": elapsed,
        "cost_estimate_usd": estimate_cost_usd(len(deep_scored)),
        "stage_summary": {"input": len(resume_files), "parsed": len(parsed), "after_stage1_bm25": len(stage1), "after_stage2_semantic": len(stage2), "after_stage3_llm": len(deep_scored)},
        "warnings": parse_warnings,
        "results": ranked,
    }
