"""
agents/gap_analyzer.py - Gap Analysis Agent.
Compares CV and job description JSONs to identify:
- Missing skills
- Weak sections
- Keyword suggestions
- ATS score (0-100)
- Recommendations
Uses the unified LLM client with OpenAI primary and local fallback.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

#project imports
from ..exceptions import AgentError, LLMServiceError
from ..logger import get_logger
from ..llm_client import LLMClient

logger = get_logger(__name__)

@dataclass
class GapAnalyzerResult:
    """Structured output from gap analysis agent."""
    missing_skills: List[str]
    weak_sections: List[str]
    keyword_suggestions: List[str]
    likely_ats_score: float
    recommendations: str

    def to_dict(self)->Dict[str,Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
def build_gap_analysis_prompt(cv_json:Dict[str,Any], job_json: Dict[str,Any])->str:
    """
    Build a detailed prompt for the LLM to extract gap analysis.
    Enforces no hallucination: only use information present in the JSONs.
    """
    cv_str = json.dumps(cv_json, indent=2, default=str)
    job_str = json.dumps(job_json, indent=2, default=str)

    prompt = f"""
You are an expert ATS (Applicant Tracking System) consultant and career coach.
Your task is to compare a candidate's CV with a job description.

RULES:
- Output ONLY valid JSON with exactly these keys: missing_skills, weak_sections, keyword_suggestions, likely_ats_score, recommendations.
- Do NOT add extra text, explanations, or markdown.
- Do NOT invent skills or experiences. Only use what is present in the CV and JD.
- missing_skills: list of skills/technologies explicitly mentioned in the job description but NOT found in the CV.
- weak_sections: list of CV sections (e.g., "experience", "education", "skills", "certificates", "profile_summary") that lack sufficient detail or relevance.
- keyword_suggestions: important keywords from the JD that should be naturally incorporated into the CV (if truthful).
- likely_ats_score: number from 0 to 100 estimating CV-JD match. Be realistic based on keyword overlap and relevance.
- recommendations: 1-3 sentences of specific, actionable advice.

Here is the CV JSON:
{cv_str}

Here is the Job Description JSON:
{job_str}

Output the JSON analysis:
"""
    return prompt

def parse_gap_analysis_response(raw_response:str)->GapAnalyzerResult:
    """
    Parse LLM response into GapAnalysisResult.
    Handles possible markdown code fences or extra whitespace.
    """
    #clean the reponse from the llm
    response_text = raw_response.strip()

    #remove markdown code fences if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    #parse json
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {response_text[:500]}")
        raise AgentError(f"LLM returned invalid JSON: {e}\nResponse: {response_text[:200]}")
    
    #validate required fields
    required = ["missing_skills", "weak_sections", "keyword_suggestions", "likely_ats_score", "recommendations"]

    for field in required:
        if field not in data:
            raise AgentError(f"LLM response missing required field: {field}")
    #ensurinfg correct types
    if not isinstance(data["missing_skills"], list):
        data["missing_skills"] = []
    if not isinstance(data["weak_sections"], list):
        data["weak_sections"] = []
    if not isinstance(data["keyword_suggestions"], list):
        data["keyword_suggestions"] = []
    if not isinstance(data["likely_ats_score"], (int, float)):
        data["likely_ats_score"] = 50.0
    else:
        data["likely_ats_score"] = float(data["likely_ats_score"])
    if not isinstance(data["recommendations"], str):
        data["recommendations"] = str(data["recommendations"])
    
    # Clamp ATS score to 0-100
    data["likely_ats_score"] = max(0.0, min(100.0, data["likely_ats_score"]))
    
    # Limit lengths-for now display all
    data["missing_skills"] = data["missing_skills"]
    data["weak_sections"] = data["weak_sections"]
    data["keyword_suggestions"] = data["keyword_suggestions"]

    return GapAnalyzerResult(**data)

def gap_analysis_agent(cv_json: Dict[str,Any], job_json: Dict[str,Any],trace_id: Optional[str]=None, use_fallback: bool = False)->GapAnalyzerResult:
    """
    Perform gap analysis between CV and job description.
    
    Args:
        cv_json: Structured CV (from DocumentParser)
        job_json: Structured job description (from DocumentParser)
        trace_id: Optional trace ID for logging
        use_fallback: If True, force local model instead of OpenAI
    
    Returns:
        GapAnalysisResult dataclass with analysis fields.
    
    Raises:
        AgentError: If analysis fails (LLM error or parsing error).
        LLMServiceError: If LLM call fails after retries.
    """
    logger.info("Starting gap analysis....")

    #build prompt
    prompt = build_gap_analysis_prompt(cv_json=cv_json, job_json=job_json)
    logger.debug(f"Prompt length: {len(prompt)} characters")

    #system prompt
    system_prompt = (
        "You are an expert ATS consultant. Output only valid JSON. "
        "Never hallucinate. Only use information explicitly present in the input."
    )

    #initialize LLmM client
    llm_client = LLMClient(trace_id=trace_id)

    #call llm
    try:
        raw_output = llm_client.generate(
            prompt = prompt,
            system_prompt=system_prompt,
            use_fallback=use_fallback
        )
        logger.debug(f"LLM response received (length: {len(raw_output)})")
    except LLMServiceError as e:
        logger.error(f"LLM service error during gap analysis: {e}")
        raise AgentError(f"Gap analysis failed: LLM error - {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during gap analysis: {e}")
        raise AgentError(f"Gap analysis failed: {e}") from e
    
    #parsing the reponse
    try:
        result = parse_gap_analysis_response(raw_response=raw_output)
    except AgentError as e:
        # If parsing fails, log raw output for debugging and re-raise
        logger.error(f"Parsing failed. Raw output: {raw_output[:500]}")
        raise 
    logger.info(
        f"Gap analysis complete. Likely ATS score: {result.likely_ats_score}, "
        f"Missing skills: {len(result.missing_skills)}, "
        f"Weak sections: {len(result.weak_sections)}"
    )
    return result

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from ..parser import DocumentParser
    from ..config import get_config
    
    if len(sys.argv) != 3:
        print("Usage: python -m agents.gap_analyzer <cv_file> <job_file>")
        print("Example: python -m agents.gap_analyzer ./data/cv.pdf ./data/job.pdf")
        sys.exit(1)
    
    cv_path = Path(sys.argv[1])
    jd_path = Path(sys.argv[2])
    
    # Parse documents
    parser = DocumentParser(trace_id="cli-gap-test")
    cv_data = parser.parse(cv_path)
    jd_data = parser.parse(jd_path)
    
    # Run gap analysis
    try:
        result = gap_analysis_agent(cv_data, jd_data, trace_id="cli-test")
        print(json.dumps(result.to_dict(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)