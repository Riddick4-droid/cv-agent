"""
agents/cover_letter_writer.py - Cover letter generator.
Outputs only the body of the letter (no salutation or closing).
LaTeX template adds the rest.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from ..exceptions import AgentError, LLMServiceError
from ..logger import get_logger
from ..llm_client import LLMClient

logger = get_logger(__name__)


@dataclass
class CoverLetter:
    content: str
    word_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_letter_prompt(cv_json: dict, job_json: dict, gap_result: dict) -> str:
    # Extract job title and company from job_json (fallback)
    job_title = "the position"
    company = "your organization"
    profile = job_json.get("profile_summary", "")
    if profile:
        import re
        # Try to extract title from first line
        first_line = profile.split('\n')[0]
        # Look for patterns like "Role at Company"
        match = re.search(r'^(.*?)\s+(?:at|for|–|-)', first_line)
        if match:
            job_title = match.group(1).strip()
        # Look for company name
        match_company = re.search(r'(?:at|for|with)\s+([A-Z][a-zA-Z0-9\s&]+)', profile)
        if match_company:
            company = match_company.group(1).strip()
    
    cv_str = json.dumps(cv_json, indent=2, default=str)
    job_str = json.dumps(job_json, indent=2, default=str)
    gap_str = json.dumps(gap_result, indent=2, default=str)
    
    prompt = f"""You are an expert cover letter writer. Write ONLY the body paragraphs of a cover letter (2-3 paragraphs, 250-350 words). Do NOT include any salutation (like "Dear") or closing (like "Sincerely"). The letter will be inserted into a template that already has the address, date, and closing.

Specifically, write 3 paragraphs:
1. Opening: Express strong interest in the {job_title} role at {company}. Mention why you are excited about this company (use any info from job description).
2. Body: Connect 2-3 specific experiences from the CV to the job requirements. Use the gap analysis to highlight relevant strengths. Write in first person.
3. Closing: Enthusiastically restate your interest and a call to action (e.g., "I look forward to discussing how I can contribute").

Use the candidate's real CV data (below). Never invent information. Include metrics if available in the CV.

CV JSON:
{cv_str}

Job Description JSON:
{job_str}

Gap Analysis (key points):
{gap_str}

Now output only the letter body (no salutation, no closing):"""
    return prompt


def parse_letter_response(raw_response: str) -> CoverLetter:
    response_text = raw_response.strip()
    # Remove any markdown
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines).strip()
    word_count = len(response_text.split())
    if word_count < 50:
        logger.warning(f"Very short letter: {word_count} words")
    return CoverLetter(content=response_text, word_count=word_count)


def cover_letter_agent(
    cv_json: dict,
    job_json: dict,
    gap_result: Any,
    trace_id: Optional[str] = None,
    use_fallback: bool = False
) -> CoverLetter:
    logger.info("Starting cover letter generation")

    if hasattr(gap_result, 'to_dict'):
        gap_dict = gap_result.to_dict()
    elif hasattr(gap_result, '__dataclass_fields__'):
        from dataclasses import asdict
        gap_dict = asdict(gap_result)
    else:
        gap_dict = gap_result

    prompt = build_letter_prompt(cv_json, job_json, gap_dict)
    logger.debug(f"Prompt length: {len(prompt)}")

    system_prompt = "You are an expert cover letter writer. Output only the body of the letter (2-3 paragraphs). Never include salutation or closing. Use first person."

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
        raise AgentError(f"Cover letter generation failed: {e}") from e

    result = parse_letter_response(raw_output)
    logger.info(f"Cover letter generated. Word count: {result.word_count}")
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from ..parser import DocumentParser
    from src.agents.gap_analyzer import gap_analysis_agent

    if len(sys.argv) != 3:
        print("Usage: python -m agents.cover_letter_writer <cv_file> <job_file>")
        sys.exit(1)

    cv_path = Path(sys.argv[1])
    jd_path = Path(sys.argv[2])

    parser = DocumentParser(trace_id="cli-cover-test")
    cv_data = parser.parse(cv_path)
    jd_data = parser.parse(jd_path)

    gap = gap_analysis_agent(cv_data, jd_data, trace_id="cli-cover")
    result = cover_letter_agent(cv_data, jd_data, gap, trace_id="cli-cover")
    print("\n" + "="*60)
    print("COVER LETTER BODY")
    print("="*60)
    print(result.content)
    print(f"\n[Word count: {result.word_count}]")