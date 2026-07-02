from typing import Optional
from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    skills_match: float
    experience: float
    education: float


class RoleAxes(BaseModel):
    backend_depth: int = 0
    frontend_depth: int = 0
    data_depth: int = 0
    leadership_depth: int = 0


class ResumeResult(BaseModel):
    rank: int
    filename: str
    candidate_name: str
    final_score: float
    bm25_score: float
    semantic_score: Optional[float] = None
    llm_score: Optional[float] = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience_relevance: float
    breakdown: Optional[ScoreBreakdown] = None
    reason: str
    stage_reached: int
    scoring_source: str
    role_axes: Optional[RoleAxes] = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    question: str = Field(min_length=1, max_length=1000)
