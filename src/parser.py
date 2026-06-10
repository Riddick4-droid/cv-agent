"""
parser.py - Document parser for CVs and job descriptions.
Extracts text from PDF, DOCX, TXT, MD files and structures into JSON.
Now with improved section detection, education splitting, skills cleaning, and leadership grouping.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Optional imports
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    from docx import Document
except ImportError:
    Document = None
try:
    import markdown
    from bs4 import BeautifulSoup
except ImportError:
    markdown = None
    BeautifulSoup = None

from .exceptions import (
    ParsingError,
    TextExtractionError,
    UnsupportedFileTypeError,
)
from .logger import get_logger, set_trace_id, _get_trace_id
from .config import get_config

logger = get_logger(__name__)

# Section heading keywords (expanded for common CV formats)

COMMON_SECTION_HEADINGS = {
    'profile': ['profile', 'summary', 'professional summary', 'personal statement', 'about me', 'bio'],
    'education': ['education', 'academic background', 'qualifications', 'degrees', 'university', 'education background'],
    'experience': [
        'experience', 'work experience', 'professional experience', 'work history',
        'relevant work experience', 'current work experience', 'employment history'
    ],
    'projects': ['projects', 'key projects', 'portfolio', 'project experience'],
    'skills': ['skills', 'technical skills', 'core competencies', 'expertise', 'technologies', 'skill proficiency'],
    'certificates': ['certifications', 'certificates', 'licenses', 'professional certifications'],
    'leadership': ['leadership', 'leadership experience', 'positions held', 'leadership roles', 'volunteer leadership'],
    'soft_skills': ['soft skills', 'interpersonal skills', 'personal attributes', 'communication skills', 'soft competencies'],
    'references': ['references', 'referees', 'available upon request', 'reference list']
}

# Text extraction functions (unchanged)
def extract_text_from_pdf(file_path: Path) -> str:
    logger.debug(f"Extracting text from PDF: {file_path}")
    if pdfplumber is None:
        raise TextExtractionError("pdfplumber not installed. Run: pip install pdfplumber")
    try:
        with pdfplumber.open(file_path) as pdf:
            text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        if not text.strip():
            raise TextExtractionError(f"PDF file {file_path} contains no extractable text.")
        logger.debug(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise TextExtractionError(f"Failed to extract text from PDF: {e}") from e

def extract_text_from_docx(file_path: Path) -> str:
    logger.debug(f"Extracting text from DOCX: {file_path}")
    if Document is None:
        raise TextExtractionError("python-docx not installed. Run: pip install python-docx")
    try:
        doc = Document(file_path)
        text = '\n'.join(para.text for para in doc.paragraphs)
        if not text.strip():
            raise TextExtractionError(f"DOCX file {file_path} contains no text.")
        logger.debug(f"Extracted {len(text)} characters from DOCX")
        return text
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise TextExtractionError(f"Failed to extract text from DOCX: {e}") from e

def extract_text_from_txt(file_path: Path) -> str:
    logger.debug(f"Extracting text from TXT: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            raise TextExtractionError(f"TXT file {file_path} is empty.")
        return text
    except UnicodeDecodeError:
        logger.debug("UTF-8 failed, trying latin-1")
        with open(file_path, 'r', encoding='latin-1') as f:
            text = f.read()
        return text
    except Exception as e:
        logger.error(f"TXT extraction failed: {e}")
        raise TextExtractionError(f"Failed to read TXT file: {e}") from e

def extract_text_from_md(file_path: Path) -> str:
    logger.debug(f"Extracting text from Markdown: {file_path}")
    if markdown is None or BeautifulSoup is None:
        raise TextExtractionError("markdown and beautifulsoup4 required. Run: pip install markdown beautifulsoup4")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        if not text.strip():
            raise TextExtractionError(f"Markdown file {file_path} has no text after conversion.")
        return text
    except Exception as e:
        logger.error(f"Markdown extraction failed: {e}")
        raise TextExtractionError(f"Failed to extract text from Markdown: {e}") from e

def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    logger.info(f"Extracting text from {file_path} (type: {suffix})")
    if suffix == '.pdf':
        return extract_text_from_pdf(file_path)
    elif suffix == '.docx':
        return extract_text_from_docx(file_path)
    elif suffix == '.txt':
        return extract_text_from_txt(file_path)
    elif suffix == '.md':
        return extract_text_from_md(file_path)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .txt, .md")


# Section splitting (improved)
def split_into_sections(text: str) -> Dict[str, str]:
    lines = text.split('\n')
    sections = {}
    current_section = 'header'
    current_content = []
    
    heading_patterns = {}
    for section, keywords in COMMON_SECTION_HEADINGS.items():
        for kw in keywords:
            pattern = re.compile(r'^\s*' + re.escape(kw) + r'\s*:?\s*$', re.IGNORECASE)
            heading_patterns[pattern] = section
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            current_content.append(line)
            continue
        
        matched = False
        for pattern, section_name in heading_patterns.items():
            if pattern.match(line_stripped):
                if current_section != 'header' and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = section_name
                current_content = []
                matched = True
                break
        if not matched:
            current_content.append(line)
    
    if current_section != 'header' and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    logger.debug(f"Found sections: {list(sections.keys())}")
    return sections

def extract_list_items(text: str) -> List[str]:
    """Extract bullet points or numbered items from a section."""
    if not text or not text.strip():
        return []
    lines = text.split('\n')
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[\-\*\•]\s+(.+)', line):
            items.append(re.sub(r'^[\-\*\•]\s+', '', line))
        elif re.match(r'^\d+\.\s+(.+)', line):
            items.append(re.sub(r'^\d+\.\s+', '', line))
        elif re.match(r'^[a-z]\.\s+(.+)', line):
            items.append(re.sub(r'^[a-z]\.\s+', '', line))
        else:
            items.append(line)
    return items


# Section parsers
def parse_profile(profile_text: str) -> str:
    return profile_text.strip()

def parse_education(section_text: str) -> List[Dict[str, str]]:
    if not section_text.strip():
        return []
    
    degree_pattern = re.compile(r'\b(MSc|M\.Sc|Master|BSc|B\.Sc|Bachelor|PhD|Ph\.D|Diploma|HND|Associate|WASSCE|GCSE)\b', re.IGNORECASE)
    
    lines = section_text.split('\n')
    entries = []
    current = {}
    current_details = []
    
    # Date patterns: month year - month year, year-year, or single year
    date_pattern = re.compile(
        r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*[–-]\s*(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}|\bpresent\b))|(\d{4}\s*[–-]\s*\d{4})|(\d{4})\b',
        re.IGNORECASE
    )
    
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                if current_details:
                    current['details'] = '\n'.join(current_details)
                entries.append(current)
                current = {}
                current_details = []
            continue
        
        # Extract date
        date_match = date_pattern.search(line)
        if date_match and 'date' not in current:
            # Get the matched group that is not None
            matched = next((g for g in date_match.groups() if g), None)
            if matched:
                current['date'] = matched
                # Remove date from line
                line = date_pattern.sub('', line).strip()
        
        # Detect degree line
        if degree_pattern.search(line) or ('|' in line and ('University' in line or 'College' in line or 'SHS' in line)):
            if current and current_details:
                current['details'] = '\n'.join(current_details)
                entries.append(current)
                current = {}
                current_details = []
            if '|' in line:
                parts = line.split('|')
                current['degree'] = parts[0].strip()
                current['institution'] = parts[1].strip() if len(parts) > 1 else ''
            else:
                current['degree'] = line
                current['institution'] = ''
        else:
            current_details.append(line)
    
    if current:
        if current_details:
            current['details'] = '\n'.join(current_details)
        entries.append(current)
    
    return entries

def parse_experience(section_text: str) -> List[Dict[str, str]]:
    if not section_text.strip():
        return []
    
    lines = section_text.split('\n')
    jobs = []
    current_job = {}
    bullets = []
    
    # Simple date pattern: month year - month year, or year-year, or year-present
    date_pattern = re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*[-–]\s*(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}|\bpresent\b)|\d{4}\s*[-–]\s*\d{4}|\d{4}\s*[-–]\s*present', re.IGNORECASE)
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_job and bullets:
                current_job['bullets'] = bullets
                jobs.append(current_job)
                current_job = {}
                bullets = []
            continue
        
        # Extract date if present
        date_match = date_pattern.search(line)
        if date_match and 'dates' not in current_job:
            current_job['dates'] = date_match.group()
            # Remove date from line to avoid confusion
            line = date_pattern.sub('', line).strip()
        
        # Check for bullet points
        if line.startswith(('-', '*', '•')):
            bullet_text = re.sub(r'^[\-\*\•]\s+', '', line)
            bullets.append(bullet_text)
        else:
            # Non-bullet: likely job title and company
            if 'title' not in current_job:
                # Try to split by '|' first (common in some CVs)
                if '|' in line:
                    parts = line.split('|')
                    current_job['title'] = parts[0].strip()
                    if len(parts) > 1:
                        current_job['company'] = parts[1].strip()
                elif ' at ' in line:
                    parts = line.split(' at ')
                    current_job['title'] = parts[0].strip()
                    current_job['company'] = parts[1].strip()
                elif ' - ' in line:
                    parts = line.split(' - ')
                    current_job['title'] = parts[0].strip()
                    if len(parts) > 1:
                        current_job['company'] = parts[1].strip()
                elif ',' in line:
                    parts = line.split(',', 1)
                    current_job['title'] = parts[0].strip()
                    if len(parts) > 1:
                        current_job['company'] = parts[1].strip()
                else:
                    current_job['title'] = line
            else:
                # Additional info might be company if not set
                if 'company' not in current_job:
                    current_job['company'] = line
                else:
                    bullets.append(line)
    
    if current_job and bullets:
        current_job['bullets'] = bullets
        jobs.append(current_job)
    
    return jobs

def parse_skills(section_text: str) -> List[str]:
    if not section_text.strip():
        return []
    # First, try to extract bullet list
    items = extract_list_items(section_text)
    if items:
        # Split each item by common separators (•, comma, etc.) to flatten
        flat_skills = []
        for item in items:
            # Split by bullet symbols, commas, or spaces with dots
            parts = re.split(r'[•·,]\s*', item)
            for p in parts:
                p = p.strip()
                if p and len(p) < 50:  # reasonable skill length
                    flat_skills.append(p)
        if flat_skills:
            return flat_skills
        return items
    else:
        # Comma-separated list
        skills = [s.strip() for s in section_text.split(',') if s.strip()]
        return skills

def parse_certificates(section_text: str) -> List[Dict[str, str]]:
    if not section_text.strip():
        return []
    lines = section_text.split('\n')
    certs = []
    current = {}
    for line in lines:
        line = line.strip()
        if not line:
            if current.get('name'):
                certs.append(current)
                current = {}
            continue
        # Certification name typically doesn't start with a date
        if not re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4}|In Progress|Completed)', line, re.IGNORECASE):
            if current.get('name'):
                certs.append(current)
                current = {}
            current['name'] = line
        else:
            if 'date' not in current:
                current['date'] = line
            else:
                current['status'] = line
    if current.get('name'):
        certs.append(current)
    if not certs:
        # Fallback to simple list
        return extract_list_items(section_text)
    return certs

def parse_projects(section_text: str) -> List[Dict[str, str]]:
    if not section_text.strip():
        return []
    projects = []
    lines = section_text.split('\n')
    current = {}
    desc_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            if current and desc_lines:
                current['description'] = '\n'.join(desc_lines)
                projects.append(current)
                current = {}
                desc_lines = []
            continue
        if '| Link to project' in line or ('|' in line and ('github' in line.lower() or 'link' in line.lower())):
            if current and desc_lines:
                current['description'] = '\n'.join(desc_lines)
                projects.append(current)
                current = {}
                desc_lines = []
            name = line.split('|')[0].strip()
            current['name'] = name
        elif line.startswith(('-', '*', '•')):
            bullet = re.sub(r'^[\-\*\•]\s+', '', line)
            desc_lines.append(bullet)
        else:
            if not current and not desc_lines:
                if len(line) < 100 and not any(x in line.lower() for x in ['tensorflow', 'pytorch', 'deployed', 'pipeline']):
                    current['name'] = line
                else:
                    desc_lines.append(line)
            else:
                desc_lines.append(line)
    if current and desc_lines:
        current['description'] = '\n'.join(desc_lines)
        projects.append(current)
    return projects

def parse_leadership(section_text: str) -> List[Dict[str, str]]:
    if not section_text.strip():
        return []
    lines = section_text.split('\n')
    roles = []
    current = {}
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                roles.append(current)
                current = {}
            continue
        # Check if line contains a date pattern
        date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–]\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s*\d{4}|\d{4}\s*[-–]\s*\d{4}', line, re.IGNORECASE)
        if date_match and 'date' not in current:
            current['date'] = date_match.group()
            line = re.sub(date_match.group(), '', line).strip()
        if line.startswith(('-', '*', '•')):
            # Bullet line: might be role name or continuation
            role_text = re.sub(r'^[\-\*\•]\s+', '', line)
            if 'role' not in current:
                current['role'] = role_text
            else:
                if 'organization' not in current:
                    current['organization'] = role_text
                else:
                    current.setdefault('details', []).append(role_text)
        else:
            if 'role' not in current:
                current['role'] = line
            else:
                if 'organization' not in current:
                    current['organization'] = line
                else:
                    current.setdefault('details', []).append(line)
    if current:
        roles.append(current)
    # If no structure detected, fallback to list of strings
    if not roles:
        return extract_list_items(section_text)
    return roles

def parse_soft_skills(section_text: str) -> List[str]:
    if not section_text.strip():
        return []
    return parse_skills(section_text)  # reuse skills parser

def parse_references(section_text: str) -> List[str]:
    if not section_text.strip():
        return []
    return extract_list_items(section_text)


# Main parser class
class DocumentParser:
    def __init__(self, trace_id: Optional[str] = None):
        if trace_id:
            set_trace_id(trace_id)
        self.trace_id = _get_trace_id()
        logger.info(f"DocumentParser initialized (trace: {self.trace_id})")
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        logger.info(f"Starting parse of {file_path}")
        if not file_path.exists():
            raise ParsingError(f"File not found: {file_path}")
        
        raw_text = extract_text(file_path)
        logger.debug(f"Extracted {len(raw_text)} characters")
        
        sections = split_into_sections(raw_text)
        
        result = {
            'profile_summary': parse_profile(sections.get('profile', '')),
            'education': parse_education(sections.get('education', '')),
            'experience': parse_experience(sections.get('experience', '')),
            'projects': parse_projects(sections.get('projects', '')),
            'skills': parse_skills(sections.get('skills', '')),
            'certificates': parse_certificates(sections.get('certificates', '')),
            'leadership': parse_leadership(sections.get('leadership', '')),
            'soft_skills': parse_soft_skills(sections.get('soft_skills', '')),
            'references': parse_references(sections.get('references', '')),
        }
        
        config = get_config()
        if config.app_mode == 'development':
            result['raw_text'] = raw_text
        
        # Fallback profile if empty
        if not result['profile_summary'] and raw_text:
            first_para = re.split(r'\n\s*\n', raw_text)[0].strip()
            if len(first_para) < 500:
                result['profile_summary'] = first_para
        
        logger.info(f"Parse complete. Profile: {bool(result['profile_summary'])}, "
                    f"Education: {len(result['education'])}, "
                    f"Experience: {len(result['experience'])}, "
                    f"Projects: {len(result['projects'])}, "
                    f"Skills: {len(result['skills'])}, "
                    f"Certifications: {len(result['certificates'])}, "
                    f"Leadership: {len(result['leadership'])}, "
                    f"Soft skills: {len(result['soft_skills'])}, "
                    f"References: {len(result['references'])}")
        return result


# CLI test
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python parser.py <file_path>")
        sys.exit(1)
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    parser = DocumentParser(trace_id="cli-test")
    try:
        result = parser.parse(file_path)
        print(json.dumps(result, indent=2, default=str))
    except ParsingError as e:
        print(f"Parsing error: {e}")
        sys.exit(1)