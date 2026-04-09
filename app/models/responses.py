from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class Change(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    section: str
    action: str
    content: str
    reason: str


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
    result: Optional[TailorResult] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
