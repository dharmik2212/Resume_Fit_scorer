
#  Resume Fit Scorer

Compare multiple resumes against a job description and get a ranked shortlist with
skill match, experience, and education scores — all fully local, no API keys needed.



##  Scoring Approach & Why

**Fully local 3-stage funnel.** No LLM is called per-resume. This was a deliberate choice:

1. **Zero cost at any scale** — 1 resume or 1000, the price is the same (free)
2. **Deterministic & auditable** — every score has traceable evidence, not a black box
3. **No API failures** — no rate limits, no downtime, no token limits to manage

### Stage 1 — BM25 Keyword Filter (free, instant)
Tokenizes JD and resumes, scores by skill taxonomy overlap (65%) + lexical overlap (25%) +
BM25 relative score (10%). Uses dynamic percentile threshold (default: keep top 60%).
Filters ~60% of clear mismatches.

### Stage 2 — Semantic Ranking (free, offline)
TF-IDF vectorizer with bigrams + cosine similarity between JD and each resume. Filters by
dynamic count (default: keep top 15). No model download required. Sentence Transformer can
be swapped in if cached locally.

### Stage 3 — Deep Evidence Scoring (free, deterministic)
Core logic in `backend/core/stage3_llm.py`. Three dimensions:

- **Skills (50% of deep score):** Uses TF-IDF importance weights from the JD. A skill
  mentioned 6 times (e.g. "Python") gets higher weight than one mentioned once. Weighted
  coverage + TF-IDF lexical overlap bonus.
- **Experience (30%):** Parses year ranges ("2020-2024", "2020 - Present"), explicit years
  ("5 years of experience"), and computes career span across all roles. Compares against JD
  requirement. Bonus for multiple companies.
- **Education (20%):** Detects degree level and exact degree text from resume line-by-line
  (B.Tech, MBA, PhD, etc.). Compares against JD requirement if specified.

### Final Score
```
Final = 0.20 × BM25 + 0.30 × Semantic + 0.50 × Deep Score
```
For candidates filtered before later stages, remaining weights are normalized.

### Chat (Bonus Feature)
Follow-up Q&A uses only pre-computed data stored at scoring time (skills, education detail,
companies, experience, resume sections). Zero API calls per message.

---


## Run Locally

Requires Python 3.10+ and pip.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://localhost:8000**. Paste a job description, upload resumes (PDF/DOCX/TXT,
max 10 MB each, up to 50 files), click Analyze.

API docs at **http://localhost:8000/docs** (FastAPI Swagger UI).

No API key is required. Optional settings in `.env` (copy from `.env.example`).

---





## API Reference

| Endpoint | Method     | Description                |
| :-------- | :------- | :------------------------- |
| `/api/health`| GET|Active modes (semantic, LLM configured) | 
| `/api/score`| POST|Multipart: `jd_text` + `resumes` (multiple files) |
| `/api/chat`| POST|JSON: `session_id` + `question` |

---

## Tests

```bash
python -m unittest discover -s backend/tests -v
```

---

## Dead Ends & Pivots

**1. Role/culture scoring removed.** Initially scored "culture fit" (leadership, mentoring,
collaboration signals) at 15% weight. The heuristics were unreliable — "team" appeared in
every resume but with very different meaning. Without an LLM to judge context, the signal
was noise. Removed entirely and redistributed weight to skills and education.

**2. Education level false positives.** "mba" matched inside "Bombay" (IIT Bombay). Fixed
with word-boundary regex (`\bmba\b` instead of substring match). Similarly "master" in
"master's" vs "Mastercard" ambiguities — the skill taxonomy learned to prioritize specific
degree terms over generic matches.

**3. Company detection over-match.** "at" regex captured "at Google" but also "at IIT" and
"at Python". Fix: filter skill keywords, degree terms, and common stopwords from company
candidates, plus only allow proper-capitalized sequences of 2+ words.

**4. Experience from year ranges.** Early version only read explicit "X years" text. Many
resumes use only date ranges ("2020 - 2024", "Jan 2020 - Present"). Added career-span
computation from all detected date ranges to capture total professional history.

**5. Education detail extraction.** Initially returned raw scores (60/100) without
showing the actual degree text. Switched to line-level regex that captures "B.Tech Computer
Science, IIT Bombay" as a display string alongside the score.

**6. TF-IDF for skill importance (not just Stage 2).** Originally Stage 3 used flat
coverage ratios (matched/total × 80). Moved to TF-IDF weighted skill importance so
skills mentioned more frequently in the JD get proportionally higher weight. Same TF-IDF
mechanism used for Stage 2 semantic similarity is reused here for a different purpose.

---

## What I Would Improve With More Time

- **Replace the 50-skill hardcoded taxonomy** with NLP phrase extraction (KeyBERT or
  similar) so the system catches domain-specific skills not in the pre-built list.
- **OCR for scanned PDFs** — current pipeline only extracts text from digital PDFs.
- **Experience month precision** — currently year-level. Months would give finer resolution.
- **Redis for sessions** instead of in-memory dict (lost on server restart).
- **Task queue** (Celery + Redis) for processing 1000+ resumes asynchronously.
- **Threshold calibration** against recruiter-labelled datasets for objective accuracy.
- **Authentication and audit logs** before any production deployment with candidate data.
- **Model refinement evaluation** — the optional Hugging Face path exists but hasn't been
  systematically compared against local scoring for accuracy gains.





