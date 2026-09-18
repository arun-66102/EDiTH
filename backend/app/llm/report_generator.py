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


def generate_report(
    classification: dict,
    quality: dict,
    similar_cases: list[dict],
    settings: Settings,
) -> str:
    """
    Generate a clinician-readable report using the Groq API.

    Args:
        classification: DR classification result dict.
        quality: Image quality assessment result dict.
        similar_cases: List of retrieved similar case dicts.
        settings: Application settings with Groq API config.

    Returns:
        Generated report text (markdown formatted).

    Raises:
        Exception: If Groq API call fails.
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

        logger.info(f"Report generated successfully ({len(report)} chars)")
        return report

    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise RuntimeError(f"Failed to generate report via Groq API: {e}") from e
