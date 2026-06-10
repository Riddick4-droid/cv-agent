"""
agents/latex_writer.py - LaTeX Writer Agent.
Generates .tex files from optimized CV and cover letter.
Uses PersonalInfo for contact details.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from ..exceptions import FileOperationError
from ..logger import get_logger
from ..config import get_config
from ..agents.personal_info_extractor import PersonalInfo

logger = get_logger(__name__)


# LaTeX special character escaping

def escape_latex(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '\\': r'\textbackslash{}',
        '#': r'\#',
        '$': r'\$',
        '%': r'\%',
        '&': r'\&',
        '~': r'\textasciitilde{}',
        '_': r'\_',
        '^': r'\textasciicircum{}',
        '{': r'\{',
        '}': r'\}',
        '[': r'{[}',
        ']': r'{]}',
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text



# Cover Letter Template (single closing, no duplicate)
COVER_LETTER_TEMPLATE = r"""
\documentclass[12pt]{letter}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue,
}

\setlength{\longindentation}{0pt}

\geometry{
    top=1in,
    bottom=1in,
    left=1in,
    right=1in,
}

%ADDRESS_BLOCK%

\begin{document}

\begin{letter}{
    Hiring Team \\
    {COMPANY_NAME} \\
    {COMPANY_ADDRESS}
}

\date{\today}

\opening{Dear Hiring Team,}

\noindent\underline{\textbf{{RE: {JOB_TITLE}}}}
\\[2\baselineskip]

{CONTENT}

\vspace{0.5\baselineskip}
Yours sincerely,\\[0.2\baselineskip]
{CANDIDATE_NAME}

\end{letter}
\end{document}
"""


# CV Template (unchanged, but we keep it for reference)
CV_TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{titlesec}
\usepackage{enumitem}

\geometry{letterpaper, margin=1in}

\titleformat{\section}{\large\bfseries}{}{0em}{}[\titlerule]
\titlespacing{\section}{0pt}{12pt}{4pt}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\begin{document}

\begin{center}
    {\Huge \textbf{{{NAME}}}} \\
    {CONTACT_INFO}
\end{center}

\section*{Professional Summary}
{PROFILE_SUMMARY}

\section*{Experience}
{EXPERIENCE}

\section*{Education}
{EDUCATION}

\section*{Skills}
{SKILLS}

\section*{Certifications}
{CERTIFICATIONS}

\end{document}
"""



# CV formatting helpers (simplified for brevity)
def format_experience(experience_list: list) -> str:
    if not experience_list:
        return "No experience listed."
    output = []
    for job in experience_list:
        title = escape_latex(job.get('title', 'Role'))
        company = escape_latex(job.get('company', 'Company'))
        dates = escape_latex(job.get('dates', ''))
        bullets = job.get('bullets', [])
        output.append(f"\\textbf{{{title}}} at {company}")
        if dates:
            output.append(f"\\hfill {dates}")
        output.append("\\vspace{-4pt}")
        for bullet in bullets:
            bullet_escaped = escape_latex(bullet)
            output.append(f"\\begin{{itemize}}[leftmargin=*]\n    \\item {bullet_escaped}\n\\end{{itemize}}")
        output.append("")
    return "\n".join(output)


def format_education(education_list: list) -> str:
    if not education_list:
        return "No education listed."
    output = []
    for edu in education_list:
        degree = escape_latex(edu.get('degree', 'Degree'))
        institution = escape_latex(edu.get('institution', 'Institution'))
        year = escape_latex(edu.get('year', ''))
        details = escape_latex(edu.get('details', ''))
        line = f"\\textbf{{{degree}}}, {institution}"
        if year:
            line += f" ({year})"
        output.append(line)
        if details:
            output.append(f"\\hfill {details}")
        output.append("")
    return "\n".join(output)


def format_skills(skills_list: list) -> str:
    if not skills_list:
        return "No skills listed."
    escaped = [escape_latex(s) for s in skills_list]
    return ", ".join(escaped)


def format_certifications(certs_list: list) -> str:
    if not certs_list:
        return "No certifications listed."
    output = ["\\begin{itemize}[leftmargin=*]"]
    for cert in certs_list:
        if isinstance(cert, dict):
            name = escape_latex(cert.get('name', ''))
            date = escape_latex(cert.get('date', ''))
            line = f"{name} ({date})" if date else name
        else:
            line = escape_latex(cert)
        output.append(f"    \\item {line}")
    output.append("\\end{itemize}")
    return "\n".join(output)



# CV writer
def write_cv_tex(optimized_cv: Any, personal_info: PersonalInfo, output_path: Path) -> Path:
    logger.info(f"Generating CV LaTeX file: {output_path}")
    
    if hasattr(optimized_cv, 'sections'):
        sections = optimized_cv.sections
    elif isinstance(optimized_cv, dict):
        sections = optimized_cv
    else:
        sections = {}
    
    profile = sections.get('profile_summary', '')
    experience_list = sections.get('experience', [])
    education_list = sections.get('education', [])
    skills_list = sections.get('skills', [])
    certificates_list = sections.get('certificates', [])
    
    experience_tex = format_experience(experience_list)
    education_tex = format_education(education_list)
    skills_tex = format_skills(skills_list)
    certifications_tex = format_certifications(certificates_list)
    
    # Build contact info line from PersonalInfo
    contact_parts = []
    if personal_info.location:
        contact_parts.append(personal_info.location)
    if personal_info.phone:
        contact_parts.append(personal_info.phone)
    if personal_info.email:
        contact_parts.append(personal_info.email)
    if personal_info.linkedin:
        contact_parts.append(personal_info.linkedin)
    if personal_info.github:
        contact_parts.append(personal_info.github)
    if personal_info.x:
        contact_parts.append(personal_info.x)
    if personal_info.instagram:
        contact_parts.append(personal_info.instagram)
    contact_info = " | ".join(contact_parts) if contact_parts else ""
    
    name = personal_info.name or "Your Name"
    profile_escaped = escape_latex(profile)
    
    tex_content = CV_TEMPLATE
    tex_content = tex_content.replace('{{NAME}}', name)
    tex_content = tex_content.replace('{CONTACT_INFO}', contact_info)
    tex_content = tex_content.replace('{PROFILE_SUMMARY}', profile_escaped)
    tex_content = tex_content.replace('{EXPERIENCE}', experience_tex)
    tex_content = tex_content.replace('{EDUCATION}', education_tex)
    tex_content = tex_content.replace('{SKILLS}', skills_tex)
    tex_content = tex_content.replace('{CERTIFICATIONS}', certifications_tex)
    
    output_path.write_text(tex_content, encoding='utf-8')
    logger.info(f"CV .tex file saved: {output_path}")
    return output_path



# Cover letter writer
def write_cover_letter_tex(
    cover_letter: Any,
    job_info: dict,
    personal_info: PersonalInfo,
    output_path: Path
) -> Path:
    logger.info(f"Generating cover letter LaTeX file: {output_path}")
    
    # Extract letter content
    if hasattr(cover_letter, 'content'):
        letter_content = cover_letter.content
    else:
        letter_content = str(cover_letter)
    letter_content_escaped = escape_latex(letter_content)
    
    # Build address block (only non-empty lines)
    address_lines = []
    if personal_info.name:
        address_lines.append(personal_info.name)
    if personal_info.location:
        address_lines.append(personal_info.location)
    if personal_info.phone:
        address_lines.append(personal_info.phone)
    if personal_info.email:
        address_lines.append(personal_info.email)
    if personal_info.linkedin:
        address_lines.append(personal_info.linkedin)
    if personal_info.github:
        address_lines.append(personal_info.github)
    if personal_info.x:
        address_lines.append(personal_info.x)
    if personal_info.instagram:
        address_lines.append(personal_info.instagram)
    
    if address_lines:
        address_block = "\\address{\n    " + " \\\\\n    ".join(address_lines) + "\n}"
    else:
        address_block = ""
    
    company_name = escape_latex(job_info.get('company', '[Organization Name]'))
    company_address = escape_latex(job_info.get('address', '[Organization Address]'))
    job_title = escape_latex(job_info.get('title', 'the position'))
    candidate_name = personal_info.name or "Your Name"
    
    tex_content = COVER_LETTER_TEMPLATE
    tex_content = tex_content.replace('%ADDRESS_BLOCK%', address_block)
    tex_content = tex_content.replace('{COMPANY_NAME}', company_name)
    tex_content = tex_content.replace('{COMPANY_ADDRESS}', company_address)
    tex_content = tex_content.replace('{JOB_TITLE}', job_title)
    tex_content = tex_content.replace('{CONTENT}', letter_content_escaped)
    tex_content = tex_content.replace('{CANDIDATE_NAME}', candidate_name)
    
    output_path.write_text(tex_content, encoding='utf-8')
    logger.info(f"Cover letter .tex file saved: {output_path}")
    return output_path


# Main agent function
def latex_writer_agent(
    optimized_cv: Any,
    cover_letter: Any,
    job_info: dict,
    personal_info: PersonalInfo
) -> Dict[str, Path]:
    logger.info("Starting LaTeX writer agent")
    
    config = get_config()
    outputs_dir = config.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    cv_tex_path = outputs_dir / "optimized_cv.tex"
    write_cv_tex(optimized_cv, personal_info, cv_tex_path)
    
    cover_tex_path = outputs_dir / "cover_letter.tex"
    write_cover_letter_tex(cover_letter, job_info, personal_info, cover_tex_path)
    
    logger.info(f"LaTeX writer complete. Outputs: {cv_tex_path}, {cover_tex_path}")
    return {
        "cv_tex": cv_tex_path,
        "cover_letter_tex": cover_tex_path
    }