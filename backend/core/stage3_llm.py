"""Stage 3: deep evidence scoring, optionally refined by one LLM call per finalist."""
import asyncio
import json
import logging
import re
import aiohttp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.config import HF_API_PROXY, HF_API_TOKEN, HF_API_URL, HF_MODEL
from backend.core.skills import SKILL_ALIASES, experience_years, skill_evidence, extract_skills

logger = logging.getLogger(__name__)
HF_DEFAULT_URL = "https://api-inference.huggingface.co/models/" + HF_MODEL + "/v1/chat/completions"


def _clamp(value) -> float:
    return round(min(max(float(value), 0.0), 100.0), 2)


def _compute_role_axes(resume_text: str, matched: list[str]) -> dict:
    text_lower = resume_text.lower()
    backend_set = {"python", "java", "node.js", "fastapi", "django", "flask", "spring", "sql", "postgresql", "mysql", "mongodb", "redis", "rest api", "docker", "kubernetes", "aws", "azure", "gcp", "microservices"}
    frontend_set = {"react", "angular", "vue", "javascript", "typescript", "html", "css", "figma", "ui/ux", "frontend"}
    data_set = {"machine learning", "deep learning", "nlp", "data analysis", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "power bi", "tableau", "excel", "statistics"}
    leadership_set = {"lead", "mentor", "manager", "stakeholder", "agile", "scrum", "team lead", "head of", "director", "vp "}

    def _depth(term_set: set) -> int:
        score = sum(1 for term in matched if term.lower() in term_set)
        score += sum(2 for term in term_set if term in text_lower)
        return min(score, 20)

    return {
        "backend_depth": _depth(backend_set),
        "frontend_depth": _depth(frontend_set),
        "data_depth": _depth(data_set),
        "leadership_depth": _depth(leadership_set),
    }


def _detect_companies(text: str) -> list[str]:
    lines = text.splitlines()
    companies = []
    for line in lines:
        line = line.strip()
        for m in re.finditer(r'\b(?:at|with|and)\s+([A-Z][A-Za-z0-9.]*(?:\s+[A-Z][A-Za-z0-9.]*){0,3})', line):
            name = m.group(1).strip().rstrip(".,;:()")
            if len(name) >= 2 and name[-1].isalpha() and name.lower() not in ('the', 'a', 'an', 'this'):
                companies.append(name)
        for m in re.finditer(r'[–—]\s*([A-Z][A-Za-z0-9.]*(?:\s+[A-Z][A-Za-z0-9.]*){0,3})', line):
            name = m.group(1).strip().rstrip(".,;:()")
            if len(name) >= 2 and name[-1].isalpha() and name.lower() not in ('the', 'a', 'an'):
                companies.append(name)
    skip_words = {'year', 'yr', 'experience', 'b.tech', 'btech', 'm.tech', 'phd', 'master',
                  'python', 'fastapi', 'docker', 'sql', 'aws', 'javascript', 'react', 'html',
                  'css', 'java', 'node', 'typescript', 'angular', 'vue', 'kubernetes', 'redis'}
    filtered = []
    seen = set()
    for c in companies:
        cl = c.lower()
        clean = c.split(".")[0].strip()
        cl = clean.lower()
        if any(w in cl for w in skip_words):
            continue
        if cl.startswith("and ") or cl.startswith("with "):
            clean = clean.split(" ", 1)[1] if " " in clean else clean
            cl = clean.lower()
        if clean not in seen and len(clean) > 1 and not cl.endswith("inc"):
            seen.add(clean)
            filtered.append(clean)
    return filtered


def _detect_education_detail(text: str) -> tuple[str, str, int]:
    lower = text.lower()

    def _has_term(terms: list[str]) -> bool:
        for t in terms:
            if re.search(r'\b' + re.escape(t) + r'\b', lower):
                return True
        return False

    # Find the degree line — extract just the degree portion, not the whole line
    degree_part = ""
    for line in text.split("\n"):
        line = line.strip()
        match = re.search(r'\b((?:B\.Tech|Btech|B\.Sc|B\.E|Bachelor|Master\'?s?|M\.Tech|MBA|PhD|Ph\.D|M\.Sc|Diploma|Associate)[^,.]*(?:,\s*[A-Z][A-Za-z\s]+)?)', line, re.I)
        if match:
            degree_part = match.group(1).strip()
            break

    # Fallback: education section
    if not degree_part:
        sections = None
        if "education" in lower:
            parts = lower.split("education")
            if len(parts) > 1:
                after = parts[1]
                for header in ["experience", "skills", "projects", "certifications", "summary"]:
                    if header in after:
                        sections = after.split(header)[0]
                        break
                if not sections:
                    sections = after[:200]
        if sections:
            lines = [l.strip() for l in sections.split("\n") if l.strip()][:2]
            degree_part = " | ".join(lines)[:90]

    education_map = [
        ("PhD", 100, ["phd", "ph.d", "doctorate", "doctor of"]),
        ("Masters", 80, ["master's", "masters", "m.tech", "mtech", "m.sc", "msc", "mba", "m.s.", "post graduate", "postgraduate"]),
        ("Bachelors", 60, ["bachelor's", "bachelors", "b.tech", "btech", "b.sc", "bsc", "b.e", "be ", "b.s.", "undergraduate"]),
        ("Diploma", 40, ["diploma"]),
        ("Associate", 20, ["associate"]),
    ]
    for label, score, terms in education_map:
        if _has_term(terms):
            return label, degree_part, score
    return "Not specified", degree_part, 0


def _build_skill_importance(jd_text: str) -> dict[str, float]:
    """Compute TF-IDF importance weight for each canonical skill in the JD.
    
    A skill mentioned frequently in the JD gets a higher weight, meaning missing it
    hurts more than missing a skill mentioned only once.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=500)
        matrix = vectorizer.fit_transform([jd_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        word_weight = dict(zip(feature_names, scores))
    except Exception:
        return {}

    importance = {}
    for skill, aliases in SKILL_ALIASES.items():
        max_w = 0.0
        for alias in aliases:
            if alias in word_weight:
                max_w = max(max_w, word_weight[alias])
            else:
                tokens = alias.split()
                if len(tokens) > 1:
                    w = sum(word_weight.get(t, 0) for t in tokens) / max(len(tokens), 1)
                    max_w = max(max_w, w)
        if max_w > 0:
            importance[skill] = max_w
    # Normalize to [0.5, 2.0] range so missing a rare skill still matters
    if importance:
        values = list(importance.values())
        min_v, max_v = min(values), max(values)
        if max_v > min_v:
            for k in importance:
                importance[k] = 0.5 + 1.5 * (importance[k] - min_v) / (max_v - min_v)
        else:
            for k in importance:
                importance[k] = 1.0
    return importance


def _compute_lexical_tfidf_overlap(jd_text: str, resume_text: str) -> float:
    """Compute TF-IDF weighted lexical overlap between JD and resume."""
    try:
        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=500)
        matrix = vectorizer.fit_transform([jd_text, resume_text])
        feature_names = vectorizer.get_feature_names_out()
        jd_vec = matrix[0].toarray()[0]
        resume_vec = matrix[1].toarray()[0]
        # Sum of min weights for terms that appear in both
        overlap = sum(min(jd_vec[i], resume_vec[i]) for i in range(len(jd_vec)) if jd_vec[i] > 0 and resume_vec[i] > 0)
        total = sum(jd_vec)
        return overlap / max(total, 0.01)
    except Exception:
        return 0.0


def _local_score(jd_text: str, resume: dict) -> dict:
    matched, missing = skill_evidence(jd_text, resume["text"])

    # --- Skills (50%): TF-IDF weighted skill importance ---
    skill_importance = _build_skill_importance(jd_text)
    total_required = len(matched) + len(missing)

    if total_required > 0 and skill_importance:
        # Weighted by TF-IDF importance from JD
        total_weight = sum(skill_importance.get(skill, 1.0) for skill in matched + missing)
        matched_weight = sum(skill_importance.get(skill, 1.0) for skill in matched)
        importance_ratio = matched_weight / max(total_weight, 0.01)
        # TF-IDF lexical overlap as bonus
        lexical_overlap = _compute_lexical_tfidf_overlap(jd_text, resume["text"])
        skill_score = min(importance_ratio * 80 + lexical_overlap * 20, 100)
    elif total_required > 0:
        # Fallback: flat coverage
        coverage = len(matched) / total_required
        lexical_overlap = _compute_lexical_tfidf_overlap(jd_text, resume["text"])
        skill_score = min(coverage * 80 + lexical_overlap * 20, 100)
    else:
        # No skills in JD at all — use TF-IDF overlap directly
        lexical_overlap = _compute_lexical_tfidf_overlap(jd_text, resume["text"])
        skill_score = min(lexical_overlap * 100, 100)

    # --- Experience (30%) ---
    requested_years = experience_years(jd_text)
    candidate_years = experience_years(resume["text"])
    companies = _detect_companies(resume["text"])
    company_count = len(companies)

    if requested_years > 0:
        experience = min((candidate_years / requested_years) * 100, 100)
    else:
        experience = min((candidate_years / 8) * 100, 100)

    if company_count >= 3:
        experience = min(experience + 5, 100)
    elif company_count >= 2:
        experience = min(experience + 2, 100)

    # --- Education (20%) ---
    cand_level, cand_detail, cand_score = _detect_education_detail(resume["text"])
    jd_level, _, jd_score = _detect_education_detail(jd_text)

    if jd_score > 0:
        if cand_score >= jd_score:
            education = 100
        elif cand_score > 0:
            education = (cand_score / jd_score) * 60
        else:
            education = 10
    else:
        education = cand_score

    # --- Final composite ---
    llm_score = 0.50 * skill_score + 0.30 * experience + 0.20 * education

    # --- Reason builder ---
    reason_parts = []
    if matched:
        reason_parts.append(f"Skills matched: {', '.join(matched[:5])}")
    if missing:
        reason_parts.append(f"Missing: {', '.join(missing[:5])}")
    if not matched and not missing:
        reason_parts.append("No canonical skills detected in JD")
        tfidf_overlap = _compute_lexical_tfidf_overlap(jd_text, resume['text'])
        reason_parts.append(f"TF-IDF text overlap: {tfidf_overlap:.0%}")

    if candidate_years > 0:
        if requested_years > 0:
            reason_parts.append(f"{candidate_years:.0f}yr exp ({requested_years:.0f}yr req)")
        else:
            reason_parts.append(f"{candidate_years:.0f}yr exp shown")
    else:
        reason_parts.append("Experience not clearly evidenced")

    if companies:
        reason_parts.append(f"Companies: {', '.join(companies[:3])}")
    else:
        reason_parts.append("Companies: not parsed")

    if cand_score > 0:
        reason_parts.append(f"Education: {cand_detail[:80] if cand_detail else cand_level}")
    else:
        reason_parts.append("Education not found")

    reason = ". ".join(reason_parts) + "."

    return {
        **resume,
        "matched_skills": matched,
        "missing_skills": missing,
        "experience_relevance": _clamp(experience),
        "llm_score": _clamp(llm_score),
        "breakdown": {"skills_match": _clamp(skill_score), "experience": _clamp(experience), "education": _clamp(education)},
        "education_label": cand_level,
        "education_detail": cand_detail,
        "companies": companies,
        "company_count": company_count,
        "reason": reason,
        "scoring_source": "local",
        "role_axes": _compute_role_axes(resume["text"], matched),
    }


def _parse_json(text: str) -> dict:
    text = re.sub(r"\x60{3}(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group() if match else text)


def _get_api_url() -> str:
    """Return configured API URL or default."""
    return HF_API_URL or HF_DEFAULT_URL


def _get_connector():
    """Return a proxy connector if configured, else None."""
    if HF_API_PROXY:
        from aiohttp import TCPConnector
        return TCPConnector()
    return None


async def _refine(session, jd_text: str, resume: dict, semaphore: asyncio.Semaphore) -> dict:
    local = _local_score(jd_text, resume)
    if not HF_API_TOKEN:
        return local
    prompt = f"""Score this resume for the job. Return JSON only with skills_match, experience, education, culture_fit, reason. Use 0-100 scores and cite only evidence present.\nJOB:\n{jd_text[:5000]}\nRESUME:\n{resume['text'][:5000]}"""
    payload = {"model": HF_MODEL, "messages": [{"role": "system", "content": "You are a careful recruiting analyst. Output valid JSON only."}, {"role": "user", "content": prompt}], "max_tokens": 450, "temperature": 0.1}
    url = _get_api_url()
    connector = _get_connector()
    try:
        kwargs = {"timeout": aiohttp.ClientTimeout(total=90)}
        if connector:
            kwargs["connector"] = connector
        async with semaphore, session.post(url, json=payload, headers={"Authorization": f"Bearer {HF_API_TOKEN}"}, **kwargs) as response:
            if response.status != 200:
                raise RuntimeError(f"provider returned {response.status}")
            body = await response.json()
            data = _parse_json(body["choices"][0]["message"]["content"])
            breakdown = {"skills_match": _clamp(data["skills_match"]), "experience": _clamp(data["experience"]), "education": _clamp(data["education"]), "culture_fit": _clamp(data["culture_fit"])}
            local["breakdown"] = breakdown
            local["experience_relevance"] = breakdown["experience"]
            local["llm_score"] = _clamp(0.40 * breakdown["skills_match"] + 0.30 * breakdown["experience"] + 0.15 * breakdown["education"] + 0.15 * breakdown["culture_fit"])
            local["reason"] = str(data.get("reason") or local["reason"])
            local["scoring_source"] = "huggingface"
    except Exception as exc:
        logger.warning("LLM refinement failed for %s; local score retained: %s", resume["filename"], exc)
    return local


async def score_resumes_llm(jd_text: str, resumes: list[dict]) -> list[dict]:
    if not resumes:
        return []
    if not HF_API_TOKEN:
        return [_local_score(jd_text, resume) for resume in resumes]
    semaphore = asyncio.Semaphore(3)
    async with aiohttp.ClientSession() as session:
        return list(await asyncio.gather(*[_refine(session, jd_text, resume, semaphore) for resume in resumes]))


def estimate_cost_usd(num_resumes: int) -> float:
    return round(num_resumes * 0.000315, 6) if HF_API_TOKEN else 0.0
