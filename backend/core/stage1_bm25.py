"""Stage 1: fast keyword/BM25-style filtering with absolute evidence scores."""
import logging
import math
import re
from rank_bm25 import BM25Okapi
from backend.config import BM25_THRESHOLD, STAGE1_KEEP_PERCENT, USE_DYNAMIC_THRESHOLDS
from backend.core.skills import extract_skills

logger = logging.getLogger(__name__)
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "of", "on", "or", "our", "that", "the", "this", "to", "we", "will", "with", "you", "your", "role", "work", "team", "job", "required", "preferred", "skills", "experience"}


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\b[a-z][a-z0-9+#.]*\b", text.lower()) if token not in STOPWORDS and len(token) > 1]


def score_resumes_bm25(jd_text: str, resumes: list[dict]) -> list[dict]:
    if not resumes:
        return []
    corpus = [tokenize(resume["text"]) for resume in resumes]
    jd_tokens = tokenize(jd_text)
    raw = BM25Okapi(corpus).get_scores(jd_tokens) if jd_tokens else [0.0] * len(resumes)
    positive_max = max(max(raw), 0.0) or 1.0
    jd_terms = set(jd_tokens)
    required_skills = set(extract_skills(jd_text))

    for index, resume in enumerate(resumes):
        resume_terms = set(corpus[index])
        lexical = 100.0 * len(jd_terms & resume_terms) / max(len(jd_terms), 1)
        resume_skills = set(extract_skills(resume["text"]))
        skill_match = 100.0 * len(required_skills & resume_skills) / max(len(required_skills), 1)
        bm25_relative = max(float(raw[index]), 0.0) / positive_max * 100.0
        if required_skills:
            score = 0.65 * skill_match + 0.25 * lexical + 0.10 * bm25_relative
        else:
            score = 0.70 * lexical + 0.30 * bm25_relative
        resume["bm25_score"] = round(min(max(score, 0.0), 100.0), 2)
    scored = sorted(resumes, key=lambda item: item["bm25_score"], reverse=True)

    if USE_DYNAMIC_THRESHOLDS and len(scored) > 1:
        cutoff = max(1, math.ceil(len(scored) * STAGE1_KEEP_PERCENT / 100))
        passed = scored[:cutoff]
        for item in scored[cutoff:]:
            logger.info("[Stage1] dynamic filter %s at %.1f", item["filename"], item["bm25_score"])
    else:
        passed = []
        for item in scored:
            if item["bm25_score"] >= BM25_THRESHOLD:
                passed.append(item)
            else:
                logger.info("[Stage1] filtered %s at %.1f", item["filename"], item["bm25_score"])
    return passed
