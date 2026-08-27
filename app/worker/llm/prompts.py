"""Shared JSON schema constants for the resume-tailor LLM pipeline.

Agent-specific system prompts and user message templates live in
app/worker/llm/graph/nodes.py alongside the node functions that use them.
Keeping prompts co-located with their nodes makes it easy to read a node
and immediately see what it asks the LLM to do, without cross-file jumping.

This module retains only the schema strings that are referenced in more than
one place, or that benefit from being inspectable outside the graph context
(e.g., documentation generation, prompt testing scripts).
"""

from __future__ import annotations

ANALYSIS_RESULT_SCHEMA = """{
  "keywords_in_jd_not_in_resume": ["list of specific keyword strings"],
  "relevant_sections": ["experience", "skills", "summary", "projects"],
  "experience_level_signal": "senior | mid-level | entry-level",
  "tone_signal": "e.g. technical and precise | startup casual | corporate formal",
  "priority_changes": ["ordered list from highest to lowest impact"]
}"""

VALIDATION_RESULT_SCHEMA = """{
  "passed": true | false,
  "fabrication_detected": true | false,
  "latex_structure_intact": true | false,
  "changes_verified": true | false,
  "feedback": "null if passed=true; otherwise specific instructions on what to fix"
}"""

SCORING_RESULT_SCHEMA = """{
  "ats_score": 0-100,
  "ats_score_delta": "integer improvement over original (can be negative)",
  "confidence": 0.0-1.0,
  "remaining_gaps": ["keywords the JD requires that could not be honestly added"]
}"""

TAILOR_OUTPUT_SCHEMA = """{
  "modified_latex": "complete LaTeX source from \\\\documentclass to \\\\end{document}",
  "changes": [
    {
      "section": "which resume section was changed",
      "action": "added | modified | reordered | removed",
      "content": "what specifically changed",
      "reason": "why this change improves alignment with the JD"
    }
  ]
}"""
