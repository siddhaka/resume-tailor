from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class TailorRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "resume_latex": (
                    r"\documentclass[11pt]{article}"
                    r"\begin{document}"
                    r"\section*{Skills}"
                    r"Python, FastAPI, PostgreSQL"
                    r"\end{document}"
                ),
                "job_description": (
                    "We are looking for a backend engineer with strong Python skills "
                    "to build and maintain scalable REST APIs. You will work closely "
                    "with the data team to integrate ML pipelines into production "
                    "services. Experience with FastAPI, PostgreSQL, and Docker is "
                    "required. Familiarity with Kubernetes and CI/CD pipelines is a plus."
                ),
            }
        },
    )

    resume_latex: str
    job_description: str

    @field_validator("resume_latex")
    @classmethod
    def validate_resume_latex(cls, v: str) -> str:
        if len(v) < 100:
            raise ValueError("Resume LaTeX appears too short to be a complete document")
        if not v.lstrip().startswith(r"\documentclass"):
            raise ValueError(
                r"resume_latex must be a valid LaTeX document starting with \documentclass"
            )
        return v

    @field_validator("job_description")
    @classmethod
    def validate_job_description(cls, v: str) -> str:
        if len(v.split()) < 20:
            raise ValueError(
                "job_description must be at least 20 words to provide meaningful context"
            )
        return v
