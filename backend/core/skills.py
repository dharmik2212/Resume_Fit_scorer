"""Transparent skill taxonomy and evidence extraction."""
import re
import time

SKILL_ALIASES = {
    "Python": ("python",), "Java": ("java",), "JavaScript": ("javascript", "js"),
    "TypeScript": ("typescript",), "React": ("react", "reactjs", "react.js"),
    "Angular": ("angular",), "Vue": ("vue", "vuejs", "vue.js"),
    "Node.js": ("node", "nodejs", "node.js"), "FastAPI": ("fastapi",),
    "Django": ("django",), "Flask": ("flask",), "Spring": ("spring boot", "spring"),
    "SQL": ("sql",), "PostgreSQL": ("postgresql", "postgres"), "MySQL": ("mysql",),
    "MongoDB": ("mongodb", "mongo"), "Redis": ("redis",),
    "AWS": ("aws", "amazon web services"), "Azure": ("azure",), "GCP": ("gcp", "google cloud"),
    "Docker": ("docker",), "Kubernetes": ("kubernetes", "k8s"), "Git": ("git", "github", "gitlab"),
    "REST APIs": ("rest api", "restful", "rest"), "GraphQL": ("graphql",),
    "Microservices": ("microservices", "microservice"),
    "CI/CD": ("ci/cd", "continuous integration", "continuous deployment"), "Linux": ("linux", "unix"),
    "Machine Learning": ("machine learning", "ml"), "Deep Learning": ("deep learning",),
    "NLP": ("natural language processing", "nlp"), "Data Analysis": ("data analysis", "data analytics"),
    "Pandas": ("pandas",), "NumPy": ("numpy",), "Scikit-learn": ("scikit-learn", "sklearn"),
    "TensorFlow": ("tensorflow",), "PyTorch": ("pytorch",), "Power BI": ("power bi", "powerbi"),
    "Tableau": ("tableau",), "Excel": ("excel",), "Agile": ("agile", "scrum"),
    "Leadership": ("leadership", "led a team", "team lead"),
    "Communication": ("communication", "stakeholder management"),
    "Product Management": ("product management", "product manager"),
    "Project Management": ("project management", "project manager"),
    "Figma": ("figma",), "UI/UX": ("ui/ux", "user experience", "user interface"),
}


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])", text.lower()) is not None


def extract_skills(text: str) -> list[str]:
    return [name for name, aliases in SKILL_ALIASES.items() if any(_contains(text, alias) for alias in aliases)]


def skill_evidence(jd_text: str, resume_text: str) -> tuple[list[str], list[str]]:
    required = extract_skills(jd_text)
    possessed = set(extract_skills(resume_text))
    return [skill for skill in required if skill in possessed], [skill for skill in required if skill not in possessed]


def experience_years(text: str) -> float:
    lower = text.lower()
    values = []
    year_ranges_data = []
    current_year = time.localtime().tm_year

    # "X years of experience in Y"
    values.extend(float(v) for v in re.findall(r"(\d{1,2}(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience", lower, re.I))
    # "X+ years" standalone + "over X years"
    values.extend(float(v) for v in re.findall(r"\b(\d{1,2}(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b", lower, re.I))
    values.extend(float(v) for v in re.findall(r"over\s+(\d{1,2})\s*(?:years?|yrs?)", lower, re.I))
    # "experience of X years"
    values.extend(float(v) for v in re.findall(r"experience\s+(?:of\s+)?(\d{1,2}(?:\.\d+)?)\s*(?:years?|yrs?)", lower, re.I))

    # Year range: "2020-2024", "2020 to present"
    for m in re.finditer(r"\b(20\d{2})\s*[-–to]+\s*(20\d{2}|present|current)", lower, re.I):
        start = int(m.group(1))
        end_str = m.group(2)
        end = current_year if end_str in ("present", "current") else int(end_str)
        year_ranges_data.append((start, end))

    # Month/year date ranges: "Jan 2020 - Mar 2024"
    for m in re.finditer(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*(20\d{2})\s*[-–]\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*(20\d{2}|present|current)", lower):
        start = int(m.group(1))
        end_str = m.group(2)
        end = current_year if end_str in ("present", "current") else int(end_str)
        year_ranges_data.append((start, end))

    # Compute career span from earliest start to latest end
    if year_ranges_data:
        min_start = min(s for s, e in year_ranges_data)
        max_end = max(e for s, e in year_ranges_data)
        values.append(max_end - min_start)

    if not values:
        return 0.0
    return min(max(values), 40.0)
