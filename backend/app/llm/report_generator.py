"""
LLM Report Generator — Uses Groq API (GPT OSS 120B / 20B) to synthesize
the DL model's output, clinical criteria, and retrieved cases into
a clear, clinician-readable report.

The LLM does NOT override or alter the underlying model's diagnosis.
It only explains and contextualizes.
"""

import logging

from groq import Groq

from app.config import Settings
from app.llm.prompts import SYSTEM_PROMPT, build_report_prompt

logger = logging.getLogger("edith.report_generator")


def remove_markdown_tables(text: str) -> str:
    """
    Ensure the report contains only narrative paragraphs and no markdown tables.
    Dissolves any table rows into natural sentences.
    """
    lines = text.split("\n")
    cleaned_lines = []
    accumulated_cells = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1:
            # Skip table separator line like |---|---|
            if set(stripped.replace("|", "").strip()).issubset({"-", ":", " "}):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1] if c.strip()]
            if cells:
                accumulated_cells.append(" — ".join(cells))
        else:
            if accumulated_cells:
                cleaned_lines.append(". ".join(accumulated_cells) + ".")
                accumulated_cells = []
            cleaned_lines.append(line)

    if accumulated_cells:
        cleaned_lines.append(". ".join(accumulated_cells) + ".")

    return "\n".join(cleaned_lines)


def generate_report(
    classification: dict,
    quality: dict,
    similar_cases: list[dict],
    settings: Settings,
) -> str:
    """
    Generate a clinician-readable report using the Groq API.
    Guaranteed to return text in paragraph format with no tables.
    """
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
        return (
            "⚠️ Report generation unavailable — please configure a valid "
            "GROQ_API_KEY in the .env file."
        )

    # Build the prompt
    user_prompt = build_report_prompt(classification, quality, similar_cases)

    logger.info(
        f"Generating report for DR grade {classification['grade']} "
        f"using {settings.GROQ_MODEL}..."
    )

    # Call Groq API
    client = Groq(api_key=settings.GROQ_API_KEY)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
            top_p=0.9,
            stream=False,
        )

        report = chat_completion.choices[0].message.content

        if not report:
            logger.warning("LLM returned empty report")
            return "Report generation returned empty content. Please try again."

        # Enforce paragraph-only output (strip any tables if generated)
        report = remove_markdown_tables(report)

        logger.info(f"Report generated successfully ({len(report)} chars)")
        return report

    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise RuntimeError(f"Failed to generate report via Groq API: {e}") from e

