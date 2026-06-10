#!/usr/bin/env python3
"""
orchestrator.py - Main orchestrator with interactive personal info collection.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .exceptions import ParsingError, AgentError, ConfigurationError, LaTeXCompilationError
from .logger import get_logger, set_trace_id
from .config import get_config
from .parser import DocumentParser
from .compiler import compile_tex_to_pdf, get_available_compiler

from src.agents.gap_analyzer import gap_analysis_agent
from src.agents.cv_rewriter import cv_rewrite_agent
from src.agents.cover_letter_writer import cover_letter_agent
from src.agents.latex_writer import latex_writer_agent
from src.agents.personal_info_extractor import extract_personal_info, PersonalInfo

logger = get_logger(__name__)


@dataclass
class ParsedCV:
    profile_summary: str
    education: list
    experience: list
    projects: list
    skills: list
    certificates: list
    leadership: list
    soft_skills: list
    references: list
    personal_info: PersonalInfo = None
    raw_text: str = ""


@dataclass
class ParsedJob:
    profile_summary: str
    education: list
    experience: list
    skills: list
    certificates: list
    projects: list = None
    leadership: list = None
    soft_skills: list = None
    references: list = None
    raw_text: str = ""


def interactive_personal_info(auto_extracted: PersonalInfo) -> PersonalInfo:
    """
    Interactive confirmation/modification of personal information.
    For each field, user types 'yes' to keep, or enters a new value.
    Empty input keeps current value if present, else leaves empty.
    """
    print("\n" + "="*60)
    print("PERSONAL INFORMATION")
    print("="*60)
    print("For each field, type 'yes' to keep the displayed value,")
    print("or enter a new value. Press Enter to keep (if value exists) or skip.\n")
    
    def ask_field(name: str, current: str, required: bool = False) -> str:
        if current:
            prompt = f"{name}: {current}\n  Keep? (yes/new value) [yes]: "
            default = current
        else:
            prompt = f"{name}: (not found)\n  Enter value or press Enter to skip: "
            default = ""
        response = input(prompt).strip()
        if response.lower() == 'yes' or (response == "" and default):
            return default
        elif response:
            return response
        else:
            return ""

    name = ask_field("Name", auto_extracted.name)
    email = ask_field("Email", auto_extracted.email)
    phone = ask_field("Phone", auto_extracted.phone)
    location = ask_field("Location", auto_extracted.location)

    print("\n--- Social Links (leave empty to skip) ---")
    linkedin = ask_field("LinkedIn URL", auto_extracted.linkedin)
    github = ask_field("GitHub URL", auto_extracted.github)
    x = ask_field("X/Twitter URL", auto_extracted.x)
    instagram = ask_field("Instagram URL", auto_extracted.instagram)

    return PersonalInfo(
        name=name, email=email, phone=phone, location=location,
        linkedin=linkedin, github=github, x=x, instagram=instagram
    )


class JobApplicationOrchestrator:
    def __init__(self, trace_id: Optional[str] = None, interactive: bool = True):
        if trace_id:
            set_trace_id(trace_id)
        self.trace_id = trace_id or "orchestrator"
        self.config = get_config()
        self.cv_parser = DocumentParser(trace_id=self.trace_id)
        self.jd_parser = DocumentParser(trace_id=self.trace_id)
        self.interactive = interactive
        logger.info(f"Orchestrator initialized (trace: {self.trace_id}, interactive={interactive})")

    def _prepare_cv_json(self, cv: ParsedCV) -> dict:
        return {
            "profile_summary": cv.profile_summary,
            "education": cv.education,
            "experience": cv.experience,
            "projects": cv.projects,
            "skills": cv.skills,
            "certificates": cv.certificates,
            "leadership": cv.leadership,
            "soft_skills": cv.soft_skills,
            "references": cv.references,
        }

    def _prepare_job_json(self, job: ParsedJob) -> dict:
        return {
            "profile_summary": job.profile_summary,
            "education": job.education,
            "experience": job.experience,
            "skills": job.skills,
            "certificates": job.certificates,
            "projects": job.projects or [],
            "leadership": job.leadership or [],
            "soft_skills": job.soft_skills or [],
            "references": job.references or [],
        }

    def _extract_job_info(self, job_json: dict) -> dict:
        company = "the organization"
        title = "the position"
        profile = job_json.get("profile_summary", "")
        if profile:
            import re
            at_match = re.search(r'(?:for|at|with)\s+([A-Z][a-zA-Z0-9\s&]+?)(?:\.|\s|$)', profile)
            if at_match:
                company = at_match.group(1).strip()
            title_match = re.search(r'^(.*?)\s+(?:at|for|–|-)', profile)
            if title_match:
                title = title_match.group(1).strip()
        return {"title": title, "company": company, "address": ""}

    def run(self, cv_path: Path, jd_path: Path) -> Dict[str, Any]:
        logger.info(f"Starting orchestration: CV={cv_path}, JD={jd_path}")

        # Step 1: Parse documents
        logger.info("Parsing CV...")
        cv_raw = self.cv_parser.parse(cv_path)
        raw_text = cv_raw.get("raw_text", "")
        auto_info = extract_personal_info(raw_text, trace_id=self.trace_id) if raw_text else PersonalInfo()
        
        if self.interactive:
            final_info = interactive_personal_info(auto_info)
        else:
            final_info = auto_info
        
        cv = ParsedCV(**cv_raw, personal_info=final_info)

        logger.info("Parsing Job Description...")
        jd_raw = self.jd_parser.parse(jd_path)
        job = ParsedJob(**jd_raw)

        cv_json = self._prepare_cv_json(cv)
        job_json = self._prepare_job_json(job)

        # Step 2: Gap analysis
        logger.info("Running gap analysis...")
        gap = gap_analysis_agent(cv_json, job_json, trace_id=self.trace_id)
        logger.info(f"Gap analysis: ATS score={gap.likely_ats_score}, missing skills={len(gap.missing_skills)}")

        # Step 3: CV rewriting
        logger.info("Rewriting CV...")
        optimized_cv = cv_rewrite_agent(cv_json, gap, trace_id=self.trace_id)
        logger.info(f"CV rewrite: new ATS score={optimized_cv.ats_score}, metrics added={len(optimized_cv.metrics_added)}")

        # Step 4: Cover letter generation
        logger.info("Generating cover letter...")
        cover_letter = cover_letter_agent(cv_json, job_json, gap, trace_id=self.trace_id)
        logger.info(f"Cover letter: {cover_letter.word_count} words")

        # Step 5: LaTeX generation
        logger.info("Generating LaTeX files...")
        job_info = self._extract_job_info(job_json)
        tex_files = latex_writer_agent(optimized_cv, cover_letter, job_info, final_info)
        logger.info(f"LaTeX files: CV={tex_files['cv_tex']}, Letter={tex_files['cover_letter_tex']}")

        # Step 6: PDF compilation
        pdf_paths = {}
        compiler = get_available_compiler()
        if compiler and self.config.latex_compiler != "none":
            logger.info("Compiling LaTeX to PDF...")
            for name, tex_path in tex_files.items():
                result = compile_tex_to_pdf(tex_path, tex_path.parent)
                if result.success:
                    pdf_paths[name.replace("_tex", "_pdf")] = str(result.pdf_path)
                    logger.info(f"Compiled {name} -> {result.pdf_path}")
                else:
                    logger.warning(f"Compilation failed for {name}: {result.error_message}")
                    pdf_paths[name.replace("_tex", "_pdf")] = None
        else:
            logger.info("PDF compilation skipped (no compiler or disabled).")

        return {
            "cv_tex": str(tex_files["cv_tex"]),
            "cv_pdf": pdf_paths.get("cv_pdf"),
            "cover_letter_tex": str(tex_files["cover_letter_tex"]),
            "cover_letter_pdf": pdf_paths.get("cover_letter_pdf"),
            "ats_score": optimized_cv.ats_score,
            "metrics_added": optimized_cv.metrics_added,
            "cover_letter_word_count": cover_letter.word_count,
            "gap_analysis": {
                "missing_skills": gap.missing_skills,
                "weak_sections": gap.weak_sections,
                "keyword_suggestions": gap.keyword_suggestions,
                "recommendations": gap.recommendations,
            },
        }


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.orchestrator <cv_file> <job_description_file>")
        print("Example: python -m src.orchestrator ./data/Riddick_CV.pdf ./data/job_desc.txt")
        sys.exit(1)

    cv_path = Path(sys.argv[1])
    jd_path = Path(sys.argv[2])

    if not cv_path.exists():
        print(f"Error: CV file not found: {cv_path}")
        sys.exit(1)
    if not jd_path.exists():
        print(f"Error: Job description file not found: {jd_path}")
        sys.exit(1)

    orchestrator = JobApplicationOrchestrator(trace_id="cli-run", interactive=True)
    try:
        result = orchestrator.run(cv_path, jd_path)
        print("\n" + "="*60)
        print("ORCHESTRATION RESULT")
        print("="*60)
        print(json.dumps(result, indent=2, default=str))
    except (ParsingError, AgentError, ConfigurationError, LaTeXCompilationError) as e:
        print(f"\n❌ Orchestration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()