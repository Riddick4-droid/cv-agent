"""
agents/cv_rewriter.py - CV Rewriting Agent with strict anti‑hallucination.
Only uses information from the original CV. Converts projects to experience if needed.
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from ..exceptions import AgentError, LLMServiceError, HallucinationDetectedError
from ..logger import get_logger
from ..llm_client import LLMClient

logger = get_logger(__name__)


@dataclass
class OptimizedCV:
    sections: Dict[str, Any]
    metrics_added: List[str]
    ats_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_real_companies_from_cv(cv_json: dict) -> List[str]:
    """Extract all company names from original CV experience section."""
    companies = []
    for exp in cv_json.get("experience", []):
        if isinstance(exp, dict) and exp.get("company"):
            companies.append(exp["company"].strip())
    # Also add a marker for "Personal Project" (allowed)
    return companies


def _extract_real_job_titles(cv_json: dict) -> List[str]:
    """Extract all job titles from original CV."""
    titles = []
    for exp in cv_json.get("experience", []):
        if isinstance(exp, dict) and exp.get("title"):
            titles.append(exp["title"].strip())
    return titles


def _validate_experience_against_original(
    exp_list: List[dict], original_cv: dict
) -> List[dict]:
    """
    Remove any experience entry that contains a company not in the original CV
    or a title that was clearly invented.
    """
    real_companies = _extract_real_companies_from_cv(original_cv)
    real_titles = _extract_real_job_titles(original_cv)
    # Allow "Personal Project" as a fallback company
    allowed_companies = set(real_companies) | {"Personal Project"}

    validated = []
    for entry in exp_list:
        company = entry.get("company", "")
        title = entry.get("title", "")

        # If company is not in original and not "Personal Project", reject
        if company not in allowed_companies:
            logger.warning(f"Rejecting hallucinated experience: {title} at {company}")
            continue

        # If the CV has no real work experience and this entry is not a project conversion, reject
        if not real_companies and company != "Personal Project":
            logger.warning(f"Rejecting invented experience when no real jobs exist: {title} at {company}")
            continue

        validated.append(entry)

    return validated


def build_rewrite_prompt(cv_json: dict, gap_result: dict) -> str:
    cv_str = json.dumps(cv_json, indent=2, default=str)
    gap_str = json.dumps(gap_result, indent=2, default=str)

    prompt = f"""You are an expert CV writer specialized in ATS optimization.

**CRITICAL – ANTI-HALLUCINATION RULE:**
- You MUST NOT invent ANY job, company, degree, or skill.
- Use ONLY information present in the CV JSON below.
- If the CV has NO work experience, do NOT create fake jobs.
- Instead, convert the "projects" section into experience entries:
  - Title = project name (e.g., "Advanced RAG Pipeline")
  - Company = "Personal Project"
  - Dates = leave empty or use the year from CV if available
  - Bullets = project description (split into 2‑3 bullets)
- Do NOT add "Intern", "Junior", or any made‑up titles.

**Other rules:**
- Group skills into categories (Programming Languages, ML Frameworks, etc.).
- Profile summary: first person, include keywords: data governance, open data, metadata, data quality, privacy, compliance, stakeholder collaboration.
- For experience bullet points, use STAR method (Situation, Task, Action, Result).
- Add realistic metrics ONLY if the original text implies improvement (e.g., "optimized pipeline" -> "reduced latency by 25%").
- Output ONLY valid JSON with the following structure:

{{
  "sections": {{
    "profile_summary": "...",
    "experience": [
      {{
        "title": "Project Name",
        "company": "Personal Project",
        "dates": "",
        "bullets": ["bullet1", "bullet2"]
      }}
    ],
    "education": [...],
    "skills": {{
      "Programming Languages": [...],
      "ML/DL Frameworks": [...],
      "Data Tools": [...],
      "Cloud & MLOps": [...],
      "GenAI/LLMs": [...],
      "Soft Skills": [...]
    }},
    "certificates": [...]
  }},
  "metrics_added": ["..."],
  "ats_score": 85
}}

Here is the CV JSON (ONLY use this):
{cv_str}

Here is the gap analysis (for keyword hints, not to invent experiences):
{gap_str}

Now produce the JSON:"""
    return prompt


def parse_rewrite_response(raw_response: str, original_cv: dict) -> OptimizedCV:
    response_text = raw_response.strip()
    # Remove markdown fences
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error. Raw response (first 500 chars): {response_text[:500]}")
        raise AgentError(f"LLM returned invalid JSON: {e}")

    # Validate top-level keys
    if "sections" not in data:
        raise AgentError("Missing 'sections' in LLM response")
    if "metrics_added" not in data:
        data["metrics_added"] = []
    if "ats_score" not in data:
        data["ats_score"] = 50.0

    # Validate experience against original CV
    exp_list = data["sections"].get("experience", [])
    if not isinstance(exp_list, list):
        exp_list = []
    validated_exp = _validate_experience_against_original(exp_list, original_cv)
    data["sections"]["experience"] = validated_exp

    # If after validation experience is empty but original CV had projects, we can fallback
    if not validated_exp and original_cv.get("projects"):
        fallback_exp = []
        for proj in original_cv["projects"]:
            if isinstance(proj, dict) and proj.get("name"):
                fallback_exp.append({
                    "title": proj["name"],
                    "company": "Personal Project",
                    "dates": "",
                    "bullets": proj.get("description", "").split(". ") if proj.get("description") else []
                })
        data["sections"]["experience"] = fallback_exp
        logger.info("Fell back to using projects as experience entries")

    # Ensure skills is a dict
    skills = data["sections"].get("skills")
    if not isinstance(skills, dict):
        data["sections"]["skills"] = {"Technical Skills": skills if isinstance(skills, list) else []}

    # Clamp ATS score
    ats_score = min(100.0, max(0.0, float(data["ats_score"])))

    return OptimizedCV(
        sections=data["sections"],
        metrics_added=data["metrics_added"],
        ats_score=ats_score
    )


def cv_rewrite_agent(
    cv_json: Dict[str, Any],
    gap_result: Any,
    trace_id: Optional[str] = None,
    use_fallback: bool = False
) -> OptimizedCV:
    logger.info("Starting CV rewrite agent (anti‑hallucination)")

    # Convert gap_result to dict
    if hasattr(gap_result, 'to_dict'):
        gap_dict = gap_result.to_dict()
    elif hasattr(gap_result, '__dataclass_fields__'):
        from dataclasses import asdict
        gap_dict = asdict(gap_result)
    else:
        gap_dict = gap_result

    prompt = build_rewrite_prompt(cv_json, gap_dict)
    logger.debug(f"Prompt length: {len(prompt)}")

    system_prompt = (
        "You are an expert CV writer. Output only valid JSON. "
        "NEVER invent companies, job titles, or experiences. "
        "If the CV has no work experience, convert projects to 'Personal Project' entries."
    )

    llm_client = LLMClient(trace_id=trace_id)

    try:
        raw_output = llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            use_fallback=use_fallback
        )
        logger.debug(f"LLM response length: {len(raw_output)}")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise AgentError(f"CV rewrite failed: {e}") from e

    # Parse with original CV for validation
    result = parse_rewrite_response(raw_output, cv_json)
    logger.info(f"CV rewrite complete. ATS score: {result.ats_score}, "
                f"Metrics added: {len(result.metrics_added)}")
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ..parser import DocumentParser
    from src.agents.gap_analyzer import gap_analysis_agent

    if len(sys.argv) != 3:
        print("Usage: python -m agents.cv_rewriter <cv_file> <job_file>")
        sys.exit(1)

    cv_path = Path(sys.argv[1])
    jd_path = Path(sys.argv[2])

    parser = DocumentParser(trace_id="cli-rewrite-test")
    cv_data = parser.parse(cv_path)
    jd_data = parser.parse(jd_path)

    gap = gap_analysis_agent(cv_data, jd_data, trace_id="cli-rewrite")
    print("Gap analysis done.")

    result = cv_rewrite_agent(cv_data, gap, trace_id="cli-rewrite")
    print(json.dumps(result.to_dict(), indent=2))