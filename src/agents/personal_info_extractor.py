
"""
agents/personal_info_extractor.py - Extracts personal information from CV raw text.
Uses LLM with regex fallback to extract name, email, phone, location, LinkedIn, GitHub.
"""

import re
import json
from typing import Optional
from dataclasses import dataclass, asdict

from ..exceptions import AgentError
from ..logger import get_logger
from ..llm_client import LLMClient

logger = get_logger(__name__)


@dataclass
class PersonalInfo:
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    x: str = ""          # Twitter/X
    instagram: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def is_empty(self) -> bool:
        return not any([self.name, self.email, self.phone, self.location, self.linkedin, self.github])



# Regex fallback patterns (in case LLM misses)
def extract_email_fallback(text: str) -> str:
    """Extract first email address using regex."""
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else ""

def extract_phone_fallback(text: str) -> str:
    """Extract phone number (supports international, spaces, dashes)."""
    patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}',  # generic
        r'\+44\s?\d{2,4}\s?\d{6,8}',  # UK
        r'0\d{9,10}',  # UK landline/mobile without country code
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""

def extract_linkedin_fallback(text: str) -> str:
    """Extract LinkedIn profile URL."""
    patterns = [
        r'linkedin\.com/in/[\w-]+',
        r'linkedin\.com/pub/[\w-]+',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return ""

def extract_github_fallback(text: str) -> str:
    """Extract GitHub profile URL or username."""
    patterns = [
        r'github\.com/[\w-]+',
        r'github:\s*([\w-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0) if 'github.com' in match.group(0) else f"github.com/{match.group(1)}"
    return ""

def extract_location_fallback(text: str) -> str:
    """
    Extract location (city, state, country) using common patterns.
    Looks for patterns like "City, Country" or "City, State".
    """
    patterns = [
        r'Location:\s*([^,\n]+(?:,\s*[A-Za-z\s]+)?)',
        r'Based in\s+([^,\n]+(?:,\s*[A-Za-z\s]+)?)',
        r'\b([A-Z][a-z]+(?:,\s*[A-Z][a-z]+(?:\s+[A-Za-z]+)?)?)\s+(?:UK|USA|Canada|Germany|France|Spain|Italy|Netherlands|Sweden|Norway|Denmark|Australia|New Zealand|India|Singapore|Japan|China|Brazil|Mexico|South Africa|Nigeria|Kenya|Ghana)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""

def extract_name_fallback(text: str) -> str:
    """
    Extract name from the first few lines (often ALL CAPS or Title Case).
    """
    lines = text.split('\n')[:20]  # first 20 lines
    for line in lines:
        line = line.strip()
        # Skip lines that look like contact info
        if re.search(r'@|github|linkedin|phone|Location', line, re.IGNORECASE):
            continue
        # If line is all caps and length 5-40 chars, likely name
        if line.isupper() and 5 <= len(line) <= 40:
            return line.title()
        # If line is title case with at least two words
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line):
            return line
    return ""


# LLM prompt (neutral, without location bias)
def build_extraction_prompt(raw_text: str) -> str:
    truncated = raw_text[:3000]
    prompt = f"""Extract personal contact information from the CV text below.

Output ONLY a valid JSON object with these exact keys: name, email, phone, location, linkedin, github.
- For linkedin and github, provide the FULL URL (e.g., "https://linkedin.com/in/username").
- For location, use format "City, Country" (e.g., "Accra, Ghana" or "London, United Kingdom").
- If a field is not found, use an empty string ("").

Example output:
{{"name": "Jane Smith", "email": "jane.smith@example.com", "phone": "+44 20 7946 0123", "location": "Manchester, United Kingdom", "linkedin": "https://linkedin.com/in/janesmith", "github": "https://github.com/janesmith"}}

CV text:
{truncated}

JSON output:"""
    return prompt


def extract_personal_info(raw_text: str, trace_id: Optional[str] = None) -> PersonalInfo:
    logger.info("Extracting personal information from CV")
    
    if not raw_text or not raw_text.strip():
        logger.warning("Empty raw text provided")
        return PersonalInfo()

    # Try LLM extraction first
    info = PersonalInfo()
    try:
        prompt = build_extraction_prompt(raw_text)
        system_prompt = "You are a data extraction assistant. Output only valid JSON. Always include full URLs for linkedin and github."
        llm_client = LLMClient(trace_id=trace_id)
        response = llm_client.generate(prompt=prompt, system_prompt=system_prompt, use_fallback=False)
        
        # Clean response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        data = json.loads(response)
        info = PersonalInfo(
            name=data.get("name", "").strip(),
            email=data.get("email", "").strip(),
            phone=data.get("phone", "").strip(),
            location=data.get("location", "").strip(),
            linkedin=data.get("linkedin", "").strip(),
            github=data.get("github", "").strip()
        )
        logger.info(f"LLM extracted: name={info.name}, email={info.email}, location={info.location}")
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}, using fallback regex")

    # Fallback: fill missing fields with regex
    if not info.email:
        info.email = extract_email_fallback(raw_text)
    if not info.phone:
        info.phone = extract_phone_fallback(raw_text)
    if not info.linkedin:
        linkedin = extract_linkedin_fallback(raw_text)
        if linkedin and not linkedin.startswith("http"):
            linkedin = "https://" + linkedin
        info.linkedin = linkedin
    if not info.github:
        github = extract_github_fallback(raw_text)
        if github and not github.startswith("http"):
            github = "https://" + github
        info.github = github
    if not info.location:
        info.location = extract_location_fallback(raw_text)
    if not info.name:
        info.name = extract_name_fallback(raw_text)

    logger.info(f"Final extracted: name={info.name}, email={info.email}, linkedin={info.linkedin[:30] if info.linkedin else ''}")
    return info


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from ..parser import DocumentParser
    
    if len(sys.argv) != 2:
        print("Usage: python -m agents.personal_info_extractor <cv_file>")
        sys.exit(1)
    
    cv_path = Path(sys.argv[1])
    parser = DocumentParser(trace_id="cli-personal-info")
    cv_data = parser.parse(cv_path)
    raw_text = cv_data.get("raw_text", "")
    info = extract_personal_info(raw_text, trace_id="cli-test")
    print(json.dumps(info.to_dict(), indent=2))