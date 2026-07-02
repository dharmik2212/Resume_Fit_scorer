"""In-memory, no-API follow-up Q&A over the current scoring evidence."""
import re
import time
from typing import Any

_SESSIONS: dict[str, dict[str, Any]] = {}
_TTL_SECONDS = 3600

_AXIS_ALIASES = {
    "backend": "backend_depth", "server": "backend_depth", "api": "backend_depth",
    "frontend": "frontend_depth", "front-end": "frontend_depth", "ui": "frontend_depth",
    "data": "data_depth", "ml": "data_depth", "machine learning": "data_depth",
    "leadership": "leadership_depth", "lead": "leadership_depth", "manager": "leadership_depth",
}


def store_session(session_id: str, jd_text: str, candidates: list[dict]) -> None:
    now = time.time()
    for key in [key for key, value in _SESSIONS.items() if now - value["created_at"] > _TTL_SECONDS]:
        _SESSIONS.pop(key, None)
    _SESSIONS[session_id] = {"created_at": now, "jd_text": jd_text, "candidates": candidates}


def _get_section(candidate: dict, section: str) -> str:
    """Get a specific resume section text, or empty string."""
    sections = candidate.get("sections") or {}
    # Try: exact, singular (strip trailing s), then add s
    for key in [section, section.rstrip("s"), section + "s"]:
        if key in sections:
            return sections[key]
    return ""


def _summary(candidate: dict) -> str:
    matched = ", ".join(candidate.get("matched_skills", [])[:5]) or "none"
    missing = ", ".join(candidate.get("missing_skills", [])[:4]) or "none"
    edu = candidate.get("education_label", "Not specified")
    companies = candidate.get("companies", [])
    comp_str = f" at {', '.join(companies[:2])}" if companies else ""
    return f"#{candidate['rank']} {candidate['candidate_name']} — Score: {candidate['final_score']:.0f}/100. Skills: {matched}. Gaps: {missing}. Education: {edu}.{comp_str}"


def _matched_candidate(query: str, candidates: list[dict]) -> dict | None:
    """Find a specific candidate by rank or name."""
    rank_match = re.search(r"(?:#|candidate|rank)\s*(\d+)", query)
    name_match = next((c for c in candidates if c["candidate_name"].lower() in query), None)
    if rank_match:
        return next((c for c in candidates if c["rank"] == int(rank_match.group(1))), name_match)
    return name_match


def answer_question(session_id: str, question: str) -> dict:
    session = _SESSIONS.get(session_id)
    if not session:
        raise KeyError("Analysis session not found or expired. Please run the analysis again.")
    candidates = session["candidates"]
    query = question.lower().strip()
    if not query:
        raise ValueError("Question cannot be empty.")

    selected = _matched_candidate(query, candidates)

    # --- Education queries ---
    if selected and any(w in query for w in ["education", "degree", "qualification", "college", "university", "study", "studied"]):
        edu = selected.get("education_label", "Not specified")
        detail = selected.get("education_detail", "")
        detail_str = f"\nDetail: {detail}" if detail else ""
        return {"answer": f"{selected['candidate_name']} — Education: {edu}.{detail_str}", "candidate_ranks": [selected["rank"]]}

    # --- Company / employer queries ---
    if selected and any(w in query for w in ["company", "employer", "workplace", "organization", "firm", "worked at"]):
        companies = selected.get("companies", [])
        comp_str = ", ".join(companies) if companies else "not clearly parsed in resume"
        return {"answer": f"{selected['candidate_name']} worked at: {comp_str}.", "candidate_ranks": [selected["rank"]]}

    # --- Experience / work history ---
        exp_text = _get_section(selected, "experience")
        if exp_text:
            lines = exp_text.split("\n")[:6]
            detail = "\n".join(lines)
            return {"answer": f"Work experience for {selected['candidate_name']}:\n{detail}", "candidate_ranks": [selected["rank"]]}
        return {"answer": f"{selected['candidate_name']} — Experience relevance: {selected.get('experience_relevance', 0):.0f}/100. No detailed experience section found in resume.", "candidate_ranks": [selected["rank"]]}

    if selected and any(w in query for w in ["skill", "technical", "project", "built", "develop"]):
        skills_text = _get_section(selected, "skills") or _get_section(selected, "technical skills")
        if skills_text:
            lines = skills_text.split("\n")[:8]
            return {"answer": f"Skills section for {selected['candidate_name']}:\n" + "\n".join(lines), "candidate_ranks": [selected["rank"]]}
        matched = ", ".join(selected.get("matched_skills", [])) or "none"
        return {"answer": f"{selected['candidate_name']} matched skills: {matched}. Missing: {', '.join(selected.get('missing_skills', [])) or 'none'}.", "candidate_ranks": [selected["rank"]]}

    # --- "Summarize #N" or "Candidate N" ---
    if selected:
        return {"answer": _summary(selected) + "\n" + selected.get("reason", ""), "candidate_ranks": [selected["rank"]]}

    # --- "Compare N and M" ---
    compare = re.findall(r"#(\d+)", query)
    if len(compare) >= 2 and ("compare" in query or "versus" in query or "vs" in query or "difference" in query):
        a = next(c for c in candidates if c["rank"] == int(compare[0]))
        b = next(c for c in candidates if c["rank"] == int(compare[1]))
        return {"answer": f"Comparison:\n{_summary(a)}\nvs\n{_summary(b)}", "candidate_ranks": [a["rank"], b["rank"]]}

    # --- Role-axis queries (backend, frontend, data, leadership) ---
    axis_key = None
    for phrase, key in _AXIS_ALIASES.items():
        if phrase in query:
            axis_key = key
            break
    if axis_key is not None:
        def _axis_score(c):
            axes = c.get("role_axes") or {}
            return axes.get(axis_key, 0)
        ranked = sorted(candidates, key=lambda c: (_axis_score(c), c.get("experience_relevance", 0), c["final_score"]), reverse=True)
        best = ranked[0]
        axis_name = axis_key.replace("_depth", "")
        score = _axis_score(best)
        return {"answer": f"{best['candidate_name']} has the strongest {axis_name} profile (depth={score}).\n{_summary(best)}", "candidate_ranks": [best["rank"]]}

    # --- Experience / seniority ---
    if any(w in query for w in ["senior", "years", "yrs", "experienced"]):
        ranked = sorted(candidates, key=lambda c: c.get("experience_relevance", 0), reverse=True)
        lines = [f"#{c['rank']} {c['candidate_name']} — {c['experience_relevance']:.0f}/100" for c in ranked]
        return {"answer": "Candidates by experience relevance:\n" + "\n".join(lines), "candidate_ranks": [c["rank"] for c in ranked]}

    # --- Education query across all candidates ---
    if any(w in query for w in ["education", "degree", "qualification"]) and not selected:
        lines = [f"#{c['rank']} {c['candidate_name']} — {c.get('education_label', 'Not specified')}" for c in candidates]
        return {"answer": "Education levels:\n" + "\n".join(lines), "candidate_ranks": [c["rank"] for c in candidates]}

    # --- Gap / missing skills ---
    if "missing" in query or "gap" in query or "lack" in query:
        pool = [c for c in candidates if c.get("missing_skills")]
        if not pool:
            return {"answer": "No explicit named skill gaps detected.", "candidate_ranks": []}
        if "closest" in query or "high" in query:
            best = max(pool, key=lambda c: c["final_score"])
        else:
            best = max(pool, key=lambda c: (len(c["missing_skills"]), c["final_score"]))
        return {"answer": f"{best['candidate_name']} — Missing: {', '.join(best['missing_skills'])}. Score: {best['final_score']:.0f}/100.", "candidate_ranks": [best["rank"]]}

    # --- Best / top / recommend ---
    if any(w in query for w in ["best", "top", "recommend", "strongest", "highest"]):
        best = candidates[0]
        return {"answer": "Top recommendation:\n" + _summary(best), "candidate_ranks": [best["rank"]]}

    # --- Count ---
    if "how many" in query or "count" in query:
        return {"answer": f"There are {len(candidates)} candidates in the ranked shortlist.", "candidate_ranks": [c["rank"] for c in candidates]}

    # --- List all ---
    if "list all" in query or "show all" in query:
        lines = [f"#{c['rank']} {c['candidate_name']} — {c['final_score']:.0f}/100 — {c.get('education_label', 'N/A')} — {c.get('experience_relevance', 0):.0f}/100 exp" for c in candidates]
        return {"answer": "Ranked candidates:\n" + "\n".join(lines), "candidate_ranks": [c["rank"] for c in candidates]}

    # --- Keyword fallback using sections ---
    terms = set(re.findall(r"[a-z][a-z0-9+#.]{2,}", query)) - {"which", "what", "who", "candidate", "about", "with", "from", "these", "their", "does", "have", "the", "for", "that", "can", "not", "all", "any", "but"}
    scored = []
    for candidate in candidates:
        # Search across full text + matched skills + sections
        sections_text = " ".join(candidate.get("sections", {}).values())
        haystack = (candidate.get("text", "") + " " + sections_text + " " + " ".join(candidate.get("matched_skills", []))).lower()
        scored.append((sum(term in haystack for term in terms), candidate.get("experience_relevance", 0), candidate["final_score"], candidate))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best = scored[0][3]
    prefix = "Based on the stored resume evidence, " if scored[0][0] else "I could not find an exact match; the closest ranked candidate is "
    return {"answer": prefix + _summary(best), "candidate_ranks": [best["rank"]]}
