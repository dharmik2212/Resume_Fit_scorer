# Resume Fit Scorer — Implementation Plan

## Assignment Context

Built for Saral's Talent Hunt assignment: a **Resume Fit Scorer** that accepts a JD + resumes
and returns ranked scores with breakdowns. Stack: Python + FastAPI backend, vanilla HTML/JS
frontend, no paid APIs required, no database.

---

## Scoring Approach & Why

**Fully local, no LLM per-resume.** The assignment says "we care about your reasoning, not
which approach." Fully local scoring wins on three grounds: zero cost, deterministic
evidence trails, and recruiter-friendly consistency. An LLM is used only optionally for
JD-level skill extraction, never per-candidate.

---

## Complete Logic Map

### 1. Parse & Extract (backend/core/parser.py)

```
PDF / DOCX / TXT bytes
    │
    ▼
extract_text() → clean_text()
    │
    ├── Candidate name: heuristic over first 8 lines
    │   (capital-letter name, blocked keywords: "resume", "summary"...)
    │
    ├── Resume sections: parse_resume_sections()
    │   Splits text by known headers (Experience, Education, Skills...)
    │   Used later by Chat for section-specific answers
    │
    └── For each file → {"filename", "text", "name", "sections", "stage_reached": 1}
```

### 2. Stage 1 — BM25 Keyword Filter (backend/core/stage1_bm25.py)

**Purpose:** Remove clear mismatches fast (free, instant).

```
JD text + N resumes
    │
    ├── Tokenize (lowercase, alphanumeric, stopword removal)
    ├── BM25Okapi(vectorized resumes).get_scores(JD tokens)
    ├── For each resume:
    │   ├── skill_match = % of JD's canonical skills present in resume
    │   ├── lexical = % of JD's words overlapping with resume
    │   ├── bm25_relative = (BM25 score / max_score) * 100
    │   └── composite = 0.65*skill + 0.25*lexical + 0.10*bm25
    │       (70/30 split if no skills detected in JD)
    │
    └── Filtering (two modes):
        ├── Dynamic (default): keep top STAGE1_KEEP_PERCENT% (60%)
        └── Absolute: keep ≥ BM25_THRESHOLD (25/100)
```

### 3. Stage 2 — Semantic Ranking (backend/core/stage2_embeddings.py)

**Purpose:** Rank by topical relevance (free, offline).

```
JD text + passed resumes
    │
    ├── TF-IDF Vectorizer(ngram_range=(1,2), stop_words="english")
    │   fit_transform([JD, resume1, resume2, ...])
    │
    ├── Cosine similarity between JD vector and each resume vector
    │   (Optional: Sentence Transformer if cached locally)
    │
    └── Filtering (two modes):
        ├── Dynamic (default): keep top STAGE2_KEEP_COUNT (15)
        └── Absolute: keep ≥ SEMANTIC_THRESHOLD (0.12)
        Both capped at TOP_N_FOR_LLM (10) → Stage 3 input
```

### 4. Stage 3 — Deep Evidence Scoring (backend/core/stage3_llm.py)

**This is the core scoring engine.** Everything here is deterministic.

```
JD text + top ~10 candidates
    │
    ▼
    _local_score(jd_text, resume) → dict with scores, evidence, metadata
```

#### 4a. Skills Score (50% of final)

```
Skill importance weights from JD (TF-IDF):
    │
    ├── TfidfVectorizer(stop_words="english") fit on JD text
    ├── For each canonical skill → max TF-IDF weight across its aliases
    ├── Normalize weights to [0.5, 2.0] range
    │   (a skill mentioned 5x gets ~2.0, mentioned once gets ~0.5)
    │
    ├── matched_weight = sum(importance of matched skills)
    ├── total_weight   = sum(importance of all required skills)
    ├── importance_ratio = matched_weight / total_weight
    │
    ├── lexical_overlap = TF-IDF cosine similarity between JD & resume
    │   (captures keyword overlap beyond canonical skills)
    │
    └── skill_score = importance_ratio * 80 + lexical_overlap * 20
        (capped 100, fallback to flat coverage if no TF-IDF)
```

**Why TF-IDF weights matter:** If JD says "Python" 6 times and "SQL" once, matching
Python should count more. Missing SQL shouldn't tank the score. TF-IDF captures this
automatically — "Python" gets weight ~2.0, "SQL" gets ~0.5.

#### 4b. Experience Score (30% of final)

```
experience_years(resume_text):
    ├── Regex for: "5 years of experience", "5+ years", "over 10 years"
    ├── Year ranges: "2020-2024" → 4yr, "2020 - Present" → 6yr (2026)
    ├── Month ranges: "Jan 2020 - Present", "Mar 2020 - Mar 2024"
    └── Career span: min(start_year) to max(end_year) across all roles
        e.g. "Amazon (2018-2020), Google (2020-Present)" → 8yr

_detect_companies(resume_text):
    ├── Regex: "at Google", "and Amazon", "— Microsoft"
    ├── Filters skill keywords & degree terms (false positives)
    └── Used for display + experience bonus

experience_score:
    ├── If JD has year requirement: min(candidate / required, 1.0) * 100
    │   e.g. 8yr candidate / 3yr required → 100
    ├── If no JD requirement: min(candidate / 8, 1.0) * 100
    │   e.g. 2yr → 25, 8yr → 100
    └── Company diversity bonus: 2+ companies → +2, 3+ → +5
```

#### 4c. Education Score (20% of final)

```
_detect_education_detail(resume_text):
    ├── Scan each line for degree keywords with word boundaries
    │   (B.Tech, Bachelors, MBA, PhD, Diploma...)  
    ├── Extract the exact degree text from the matching line
    │   e.g. "B.Tech Computer Science, IIT Bombay"
    ├── Fallback: "Education" section text if no direct match
    └── Score map: PhD=100, Masters=80, Bachelors=60, Diploma=40, None=0

education_score:
    ├── If JD specifies a degree requirement:
    │   ├── Candidate meets or exceeds → 100
    │   ├── Candidate has lower degree → partial credit
    │   └── No education in resume → 10
    └── If JD doesn't require specific degree:
        └── Candidate's education level directly (0-100)
```

#### 4d. Final Composite

```
llm_score = 0.50 * skill_score + 0.30 * experience_score + 0.20 * education_score
```
(No "culture/role" component — removed as it was unreliable without an LLM.)

#### 4e. Reason Text Builder

Built from actual data, not templates:
```
"Skills matched: Python, FastAPI, Docker. Missing: SQL. 8yr exp (3yr req).
 Companies: Google, Amazon. Education: B.Tech Computer Science, IIT Bombay."
```

Each segment is conditional — only present if the data exists.

---

### 5. Pipeline Orchestration (backend/core/pipeline.py)

```
run_pipeline(jd_text, resume_files)
    │
    ├── Parse all files → [{text, name, sections}]
    ├── Stage 1 (BM25) → scored + filtered
    ├── Stage 2 (Semantic) → scored + filtered
    ├── Stage 3 (Deep Score) → scored (local)
    ├── Enrich filtered candidates with partial scores
    ├── Compute final = 0.20*BM25 + 0.30*Semantic + 0.50*Deep
    │   (weights normalized if stages missing)
    ├── Sort by final_score descending
    └── Return {session_id, results[], stage_summary, cost, time}
```

**Why BM25 and Semantic still contribute:** Stage 3 only gets top ~10. For filtered
candidates, BM25 and Semantic signals are the only scores — they come from different
dimensions (keyword hit vs. topical relevance) so both add useful information.

---

### 6. Chat Engine (backend/core/chat.py)

**No external API calls.** Answers from pre-computed data only.

```
Stored per candidate at pipeline time:
  ├── matched_skills, missing_skills
  ├── experience_relevance, education_label, education_detail
  ├── companies[], role_axes{}
  ├── sections{experience, education, skills, ...}
  └── full resume text (for keyword fallback)

Question routing:
  ├── "Education #1" / "Degree" / "College" → education_label + education_detail
  ├── "Company #1" / "Employer" → companies[]
  ├── "Experience #1" / "Work" → sections["experience"] or experience_relevance
  ├── "Skills #1" / "Technical" → matched/missing skills
  ├── "Compare #1 and #2" → side-by-side _summary()
  ├── "Best backend" / "backend_depth" → sort by role_axes.backend_depth
  ├── "Missing" / "Gap" → sort candidates by missing_skills count
  ├── "List all" → all candidates with rank + score + education
  ├── "Years" / "Senior" → sort by experience_relevance
  └── Keyword fallback → search across all stored fields
```

---

### 7. Session Lifecycle

```
POST /api/score → pipeline → {session_id, results}
    │
    ├── session_id (UUID hex) stored in memory dict
    ├── TTL = 3600s (1 hour)
    └── POST /api/chat {session_id, question} → lookup → routed answer
```

---

## File Structure

```
Resume_Fit_Scorer/
├── main.py                       # FastAPI entry point + frontend mount
├── requirements.txt
├── .env / .env.example           # Configuration (thresholds, API keys)
├── implementation_plan.md        # This file
├── README.md
├── backend/
│   ├── config.py                 # Env-based config with defaults
│   ├── api/
│   │   └── routes.py             # /health, /score, /chat endpoints + validation
│   ├── core/
│   │   ├── parser.py             # PDF/DOCX/TXT extraction, name + section detection
│   │   ├── stage1_bm25.py        # BM25 keyword filter (adaptive threshold)
│   │   ├── stage2_embeddings.py  # TF-IDF / Sentence-Transformer semantic rank
│   │   ├── stage3_llm.py         # Deep scoring: TF-IDF skills, experience, education
│   │   ├── pipeline.py           # 3-stage orchestrator + enrichment
│   │   ├── skills.py             # 50+ skill taxonomy with aliases + evidence
│   │   └── chat.py               # Section-aware Q&A (no API calls)
│   ├── models/
│   │   └── schemas.py            # Pydantic models for API contracts
│   └── tests/
│       └── test_core.py          # 7 unit tests (parser, scoring, chat)
└── frontend/
    ├── index.html                # Single-page app (textarea + upload + results + chat)
    ├── style.css                 # Minimal clean styles
    └── app.js                    # Client logic: form, rendering, chat
```

---

## Weights & Thresholds Summary

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| WEIGHT_BM25 | 0.20 | Final score contribution from Stage 1 |
| WEIGHT_SEMANTIC | 0.30 | Final score contribution from Stage 2 |
| WEIGHT_LLM | 0.50 | Final score contribution from Stage 3 |
| SKILLS_WEIGHT | 0.50 | Within Stage 3: skills component |
| EXP_WEIGHT | 0.30 | Within Stage 3: experience component |
| EDU_WEIGHT | 0.20 | Within Stage 3: education component |
| BM25_THRESHOLD | 25 | Absolute cutoff for Stage 1 (when dynamic off) |
| SEMANTIC_THRESHOLD | 0.12 | Absolute cutoff for Stage 2 (when dynamic off) |
| STAGE1_KEEP_PERCENT | 60% | Dynamic: keep top 60% in Stage 1 |
| STAGE2_KEEP_COUNT | 15 | Dynamic: keep top 15 in Stage 2 |
| TOP_N_FOR_LLM | 10 | Max candidates to pass to Stage 3 |

---

## Future Improvements

- **NLP skill extraction** — replace hand-maintained taxonomy with phrase embeddings
- **OCR** — handle image-only PDFs (scanned resumes)
- **Threshold calibration** — tune against recruiter-labelled data
- **Redis sessions** — persist across server restarts
- **Task queue** — Celery for batch 1000+ resume processing
- **Authentication** — multi-tenant with audit logs for production use
