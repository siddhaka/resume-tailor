from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Change(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    section: str
    action: str
    content: str
    reason: str


class AnalysisResult(BaseModel):
    """Internal — produced by the analyzer node, consumed by the tailor node."""

    model_config = ConfigDict(populate_by_name=True)

    keywords_in_jd_not_in_resume: list[str]
    relevant_sections: list[str]
    experience_level_signal: str
    tone_signal: str
    priority_changes: list[str]


class ValidationResult(BaseModel):
    """Internal — produced by the validator node; controls the retry edge."""

    model_config = ConfigDict(populate_by_name=True)

    passed: bool
    fabrication_detected: bool
    latex_structure_intact: bool
    changes_verified: bool
    feedback: str | None = None


class ScoringResult(BaseModel):
    """Internal — produced by the scorer node; surfaced in TailorResult."""

    model_config = ConfigDict(populate_by_name=True)

    ats_score: int = Field(ge=0, le=100)
    ats_score_delta: int
    confidence: float
    remaining_gaps: list[str]


class TailorResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    modified_latex: str
    changes: list[Change]
    ats_keywords_added: list[str]
    ats_keywords_missing: list[str]


class CreateJobResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str
    status: str
    message: str


class JobResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str
    status: str
    result: TailorResult | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
