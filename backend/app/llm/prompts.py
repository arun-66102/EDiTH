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
4. STRICT FORMATTING RULE: The output MUST consist ONLY of narrative paragraphs. Absolutely DO NOT include ANY tables, markdown tables (| ... |), HTML tables, grids, or tabular columns under any circumstances. All descriptions, metrics, and recommendations must be written entirely as natural text paragraphs.
5. When referencing similar cases, note that they are de-identified historical cases for context only.
6. Include actionable next steps based on the DR severity grade and established clinical guidelines, written strictly in paragraph form.
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

## Formatting Requirement
CRITICAL: The output MUST consist ONLY of narrative paragraphs. Absolutely NO tables (no markdown tables with pipes, no HTML tables, no grids, no columns) are permitted anywhere in the response.

## Report Sections
Generate the report with these exact section headings, writing each section strictly as natural prose paragraphs:

### 1. Summary
A concise paragraph overview of the screening finding.

### 2. Classification Details
A paragraph explaining the DR grade, clinical meaning, and model confidence level in sentence form.

### 3. Detected Features
A paragraph describing typical lesion patterns expected at this severity level, referencing the Grad-CAM heatmap highlighting the relevant regions.

### 4. Similar Cases
A paragraph referencing the retrieved historical cases for clinical context.

### 5. Recommended Actions
A paragraph outlining recommended clinical next steps, follow-up timeline, and management considerations based on established guidelines.

### 6. Disclaimer
A paragraph providing the standard AI screening disclaimer.

Remember: Write strictly in continuous paragraphs. Tables must NOT be present anywhere in the output.
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
