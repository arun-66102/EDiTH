"""
LLM Prompt Templates for Clinical Report Generation.

Contains the system prompt with clinical safety instructions and
the report generation prompt template.

CRITICAL DESIGN CONSTRAINT:
    The LLM NEVER overrides the DL model's diagnosis. It only explains,
    contextualizes, and generates a readable report based on the model's
    output. The DR grade is treated as ground truth within the report.
"""

SYSTEM_PROMPT = """You are EDiTH, an AI clinical assistant specialized in Diabetic Retinopathy (DR) screening. Your role is to generate clear, clinician-readable reports based on the analysis results provided by the DR classification model.

CRITICAL RULES:
1. NEVER override, contradict, or second-guess the DR classification model's diagnosis. The model's DR grade is the definitive finding — your job is to explain it, not change it.
2. ALWAYS include a disclaimer that this is AI-assisted screening and must be reviewed by a qualified ophthalmologist.
3. Use clear, professional medical language appropriate for a clinician audience.
4. Structure the report with clear sections for easy scanning.
5. When referencing similar cases, note that they are de-identified historical cases for context only.
6. Include actionable next steps based on the DR severity grade and established clinical guidelines.
7. Do NOT fabricate clinical findings that are not supported by the model's output.
8. Do NOT include patient identifying information.
"""

REPORT_TEMPLATE = """Generate a concise clinician-readable report based on the following DR screening results:

## Analysis Input

**DR Classification Result:**
- Grade: {grade} ({label})
- Model Confidence: {confidence:.1%}
- Class Probabilities: {probabilities}

**Image Quality:**
- Gradable: {quality_gradable}
- Quality Confidence: {quality_confidence:.1%}
- Issues: {quality_issues}

**Similar Historical Cases Retrieved:**
{similar_cases_text}

## Report Requirements

Generate the report with these exact sections:

### 1. Summary
A 2–3 sentence overview of the screening finding.

### 2. Classification Details
Explain the DR grade, what it means clinically, and the model's confidence level.

### 3. Detected Features
Based on the DR grade, describe the typical lesion patterns expected at this severity level. Reference the Grad-CAM heatmap that highlights the relevant regions.

### 4. Similar Cases
Briefly reference the retrieved historical cases for clinical context. Note these are de-identified.

### 5. Recommended Actions
Based on established clinical guidelines (ICO/AAO), provide specific next steps appropriate for this DR grade — including follow-up timeline, referral urgency, and management considerations.

### 6. Disclaimer
Include a standard AI screening disclaimer.

Keep the report professional, concise (400–600 words), and actionable.
"""


def format_similar_cases(similar_cases: list[dict]) -> str:
    """Format similar cases for inclusion in the prompt."""
    if not similar_cases:
        return "No similar historical cases available."

    lines = []
    for i, case in enumerate(similar_cases, 1):
        lines.append(
            f"Case {i} (ID: {case.get('case_id', 'N/A')}, "
            f"Grade: {case.get('dr_grade', 'N/A')}, "
            f"Similarity: {case.get('similarity', 0):.1%}):\n"
            f"  {case.get('text', 'No details available.')}"
        )

    return "\n\n".join(lines)


def build_report_prompt(
    classification: dict,
    quality: dict,
    similar_cases: list[dict],
) -> str:
    """
    Build the complete report generation prompt from analysis results.

    Args:
        classification: Dict with grade, label, confidence, probabilities.
        quality: Dict with gradable, confidence, issues.
        similar_cases: List of retrieved similar case dicts.

    Returns:
        Formatted prompt string for the LLM.
    """
    probs_str = ", ".join(
        f"Grade {i}: {p:.1%}"
        for i, p in enumerate(classification.get("probabilities", []))
    )

    similar_text = format_similar_cases(similar_cases)

    return REPORT_TEMPLATE.format(
        grade=classification["grade"],
        label=classification["label"],
        confidence=classification["confidence"],
        probabilities=probs_str or "Not available",
        quality_gradable=quality.get("gradable", True),
        quality_confidence=quality.get("confidence", 1.0),
        quality_issues=", ".join(quality.get("issues", [])) or "None",
        similar_cases_text=similar_text,
    )
