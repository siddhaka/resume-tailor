# Gradio client — thin HTTP wrapper around the FastAPI backend.
#
# This file contains zero business logic, zero database access, zero Redis
# connections, and zero LLM calls. It only speaks HTTP to the API.
#
# That separation is intentional: the Gradio interface is fully replaceable.
# You could swap it for a CLI (`python cli.py --resume resume.tex --jd jd.txt`),
# a VS Code extension, a GitHub Action, or a mobile app without changing a
# single line of the backend. The API is the product; this file is one
# particular way to interact with it.

from __future__ import annotations

import os
import time
from collections.abc import Generator

import gradio as gr
import httpx

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — all values from environment, never hardcoded
# ─────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 120


# ─────────────────────────────────────────────────────────────────────────────
# Formatting — presentation concern, belongs here not in the API
# ─────────────────────────────────────────────────────────────────────────────


def format_changes(result: dict) -> str:
    """Format structured API result into human-readable text.

    Formatting is a presentation concern that belongs in the client layer, not
    in the API. The API returns clean, structured JSON; a CLI client, a mobile
    app, or a VS Code extension would format that same JSON completely
    differently. Keeping formatting here means the API stays reusable.
    """
    lines: list[str] = []

    lines.append("CHANGES MADE")
    lines.append("─" * 40)
    lines.append("")

    changes = result.get("changes", [])
    if changes:
        for change in changes:
            section = change.get("section", "unknown").upper()
            action = change.get("action", "modified")
            content = change.get("content", "")
            reason = change.get("reason", "")
            lines.append(f"[{section}] {action}")
            lines.append(f"  {content}")
            lines.append(f"  Why: {reason}")
            lines.append("")
    else:
        lines.append("No changes recorded.")
        lines.append("")

    added = result.get("ats_keywords_added", [])
    lines.append("KEYWORDS ADDED")
    lines.append("─" * 40)
    lines.append(", ".join(added) if added else "None added")

    missing = result.get("ats_keywords_missing", [])
    lines.append("")
    lines.append("KEYWORDS STILL MISSING")
    lines.append("─" * 40)
    lines.append(", ".join(missing) if missing else "None — great fit!")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core function — submit job and stream status updates
# ─────────────────────────────────────────────────────────────────────────────


def submit_and_poll(
    resume_latex: str,
    job_description: str,
    api_key: str,
) -> Generator[tuple[str, str, str], None, None]:
    """Submit a tailoring job and yield (latex, changes, status) as it progresses.

    This is a generator so Gradio can stream intermediate status messages to
    the UI while the LLM processes. Without this, the UI would appear frozen
    for the 10–30 seconds the LLM takes, giving users no feedback.

    Yields: (modified_latex, changes_text, status_message)
    """
    # ── 1. Validate inputs ────────────────────────────────────────────────────
    if not resume_latex.strip() or not job_description.strip() or not api_key.strip():
        yield ("", "", "Error: all fields are required.")
        return

    # ── 2. Submit job ─────────────────────────────────────────────────────────
    yield ("", "", "Submitting job to API...")

    try:
        response = httpx.post(
            f"{API_BASE_URL}/v1/jobs",
            headers={"X-API-Key": api_key},
            json={"resume_latex": resume_latex, "job_description": job_description},
            timeout=30,
        )
    except httpx.RequestError:
        yield (
            "",
            "",
            "Error: Could not connect to the API. Is the service running?",
        )
        return

    if response.status_code == 401:
        yield ("", "", "Error: Invalid API key. Check your key and try again.")
        return

    if response.status_code == 422:
        try:
            detail = response.json().get("detail", "unknown validation error")
            if isinstance(detail, list):
                detail = "; ".join(
                    d.get("msg", str(d)) for d in detail
                )
        except Exception:
            detail = response.text[:200]
        yield ("", "", f"Error: Invalid input — {detail}")
        return

    if response.status_code == 429:
        yield (
            "",
            "",
            "Error: Rate limit exceeded. Wait 60 seconds and try again.",
        )
        return

    if response.status_code != 202:
        yield ("", "", f"Error: Unexpected API response {response.status_code}.")
        return

    job_id: str = response.json()["job_id"]
    yield (
        "",
        "",
        f"Job submitted (ID: {job_id[:8]}...). Waiting for result...",
    )

    # ── 3. Poll for result ────────────────────────────────────────────────────
    start_time = time.time()
    attempt = 0

    while True:
        if time.time() - start_time > POLL_TIMEOUT_SECONDS:
            yield (
                "",
                "",
                "Timeout: job took over 2 minutes. The service may be overloaded.",
            )
            return

        time.sleep(POLL_INTERVAL_SECONDS)
        attempt += 1

        try:
            poll_response = httpx.get(
                f"{API_BASE_URL}/v1/jobs/{job_id}",
                headers={"X-API-Key": api_key},
                timeout=15,
            )
            data = poll_response.json()
        except httpx.RequestError:
            yield (
                "",
                "",
                f"Attempt {attempt}: connection issue, retrying...",
            )
            continue
        except Exception:
            yield ("", "", f"Attempt {attempt}: unexpected error, retrying...")
            continue

        status = data.get("status", "unknown")

        if status == "queued":
            yield ("", "", f"Attempt {attempt}: job is queued, waiting for a worker...")
            continue

        if status == "processing":
            yield (
                "",
                "",
                f"Attempt {attempt}: LLM is processing your resume... (10–20 s typical)",
            )
            continue

        if status == "failed":
            error = data.get("error", "unknown error")
            yield ("", "", f"Error: job failed — {error}")
            return

        if status == "complete":
            result = data.get("result", {})
            modified_latex = result.get("modified_latex", "")
            changes_text = format_changes(result)
            yield (
                modified_latex,
                changes_text,
                "Complete! Your tailored resume is ready.",
            )
            return

        yield ("", "", f"Unknown status '{status}'. Retrying...")


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Resume Tailor") as demo:
    gr.Markdown("# Resume Tailor")
    gr.Markdown(
        "Paste your LaTeX resume and a job description. "
        "The system will tailor your resume using a multi-agent LLM pipeline "
        "and return modified LaTeX with a full audit trail of every change made."
    )

    with gr.Row():
        with gr.Column(scale=1):
            resume_input = gr.Textbox(
                lines=25,
                label="Resume (LaTeX source)",
                placeholder=r"\documentclass{article}" + "\n...",
            )
            jd_input = gr.Textbox(
                lines=12,
                label="Job Description",
                placeholder="We are looking for...",
            )
            api_key_input = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="Your X-API-Key",
            )
            submit_btn = gr.Button(
                "Tailor My Resume",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            status_output = gr.Textbox(
                label="Status",
                lines=2,
                interactive=False,
            )
            latex_output = gr.Textbox(
                label="Modified LaTeX",
                lines=20,
                interactive=False,
            )
            changes_output = gr.Textbox(
                label="Changes Made",
                lines=15,
                interactive=False,
            )

    submit_btn.click(
        fn=submit_and_poll,
        inputs=[resume_input, jd_input, api_key_input],
        outputs=[latex_output, changes_output, status_output],
    )

if __name__ == "__main__":
    # server_name="0.0.0.0" — by default Gradio listens only on 127.0.0.1,
    # which is not accessible from outside the container. Binding to 0.0.0.0
    # means the server listens on all network interfaces, making it reachable
    # via the container's exposed port 7860 on the host machine.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
