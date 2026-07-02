"""Stage 2: semantic ranking with an offline TF-IDF fallback."""
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.config import EMBEDDING_MODEL, SEMANTIC_THRESHOLD, STAGE2_KEEP_COUNT, TOP_N_FOR_LLM, USE_DYNAMIC_THRESHOLDS, USE_TRANSFORMER_EMBEDDINGS

logger = logging.getLogger(__name__)
_model = None
_model_attempted = False


def get_model():
    global _model, _model_attempted
    if not USE_TRANSFORMER_EMBEDDINGS:
        return None
    if not _model_attempted:
        _model_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception as exc:
            logger.warning("Local embedding model unavailable; using TF-IDF: %s", exc)
    return _model


def _similarities(jd_text: str, texts: list[str]) -> np.ndarray:
    model = get_model()
    if model is not None:
        vectors = model.encode([jd_text, *texts], show_progress_bar=False, normalize_embeddings=True)
        return cosine_similarity(vectors[:1], vectors[1:])[0]
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit_transform([jd_text, *texts])
        return cosine_similarity(matrix[:1], matrix[1:])[0]
    except ValueError:
        return np.zeros(len(texts))


def score_resumes_semantic(jd_text: str, resumes: list[dict]) -> list[dict]:
    if not resumes:
        return []
    similarities = _similarities(jd_text, [resume["text"] for resume in resumes])
    for resume, similarity in zip(resumes, similarities):
        resume["semantic_score"] = round(min(max(float(similarity), 0.0), 1.0), 4)
    scored = sorted(resumes, key=lambda item: item["semantic_score"], reverse=True)

    if USE_DYNAMIC_THRESHOLDS:
        keep = min(len(scored), STAGE2_KEEP_COUNT)
        passed = scored[:keep]
        for item in scored[keep:]:
            logger.info("[Stage2] dynamic filter %s at %.3f", item["filename"], item["semantic_score"])
    else:
        passed = []
        for item in scored:
            if item["semantic_score"] >= SEMANTIC_THRESHOLD:
                passed.append(item)
            else:
                logger.info("[Stage2] filtered %s at %.3f", item["filename"], item["semantic_score"])
    return passed[:TOP_N_FOR_LLM]
