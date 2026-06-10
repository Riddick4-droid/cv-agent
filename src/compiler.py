#!/usr/bin/env python3
"""
compiler.py - LaTeX to PDF compiler with Windows path search.
Supports pdflatex (MiKTeX), latexmk, and Tectonic.
"""

import subprocess
import shutil
import os
from pathlib import Path
from typing import Optional, List
from enum import Enum
from dataclasses import dataclass

from .exceptions import LaTeXCompilationError
from .logger import get_logger
from .config import get_config

logger = get_logger(__name__)


class CompilerType(Enum):
    TECTONIC = "tectonic"
    LATEXMK = "latexmk"
    PDFLATEX = "pdflatex"


@dataclass
class CompileResult:
    success: bool
    pdf_path: Optional[Path]
    log_output: str
    error_message: str



# Windows-specific: find pdflatex in common MiKTeX paths
def find_pdflatex_windows() -> Optional[Path]:
    """Search for pdflatex.exe in common MiKTeX installation directories."""
    common_paths = [
        r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\Program Files (x86)\MiKTeX\miktex\bin\pdflatex.exe",
        r"C:\Users\{user}\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
    ]
    user = os.getenv("USERNAME")
    for path_template in common_paths:
        path = path_template.replace("{user}", user) if "{user}" in path_template else path_template
        p = Path(path)
        if p.exists():
            return p
    # Also try 'where' command
    try:
        result = subprocess.run(["where", "pdflatex"], capture_output=True, text=True, shell=True)
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip().split('\n')[0])
    except:
        pass
    return None


def check_compiler_available(compiler: CompilerType) -> bool:
    """Check if a compiler is available, with special handling for pdflatex on Windows."""
    if compiler == CompilerType.PDFLATEX:
        # First check if pdflatex is in PATH
        if shutil.which("pdflatex"):
            return True
        # Otherwise search common MiKTeX paths
        return find_pdflatex_windows() is not None
    else:
        cmd = compiler.value
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, timeout=5, shell=True)
            return result.returncode == 0
        except:
            return False


def get_available_compiler() -> Optional[CompilerType]:
    """Return first available compiler (pdflatex first on Windows)."""
    config = get_config()
    requested = config.latex_compiler

    if requested != "auto" and requested != "none":
        try:
            compiler = CompilerType(requested)
            if check_compiler_available(compiler):
                logger.info(f"Using requested compiler: {compiler.value}")
                return compiler
            else:
                logger.warning(f"Requested compiler '{requested}' not found. Falling back to auto-detect.")
        except ValueError:
            logger.warning(f"Unknown compiler type: {requested}. Auto-detecting.")

    # Priority: pdflatex (MiKTeX) first, then latexmk, then tectonic
    for compiler in [CompilerType.PDFLATEX, CompilerType.LATEXMK, CompilerType.TECTONIC]:
        if check_compiler_available(compiler):
            logger.info(f"Auto-detected compiler: {compiler.value}")
            return compiler
    return None


def compile_with_pdflatex(tex_path: Path, output_dir: Path) -> CompileResult:
    """Compile using pdflatex (via MiKTeX or system pdflatex)."""
    logger.info(f"Compiling {tex_path} with pdflatex")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find pdflatex executable
    pdflatex_path = shutil.which("pdflatex")
    if not pdflatex_path:
        found = find_pdflatex_windows()
        if found:
            pdflatex_path = str(found)
            logger.info(f"Using pdflatex from: {pdflatex_path}")
        else:
            return CompileResult(
                success=False,
                pdf_path=None,
                log_output="",
                error_message="pdflatex not found. Install MiKTeX or TeX Live."
            )

    original_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        # Copy tex file if needed
        tex_in_output = output_dir / tex_path.name
        if tex_path != tex_in_output:
            shutil.copy2(tex_path, tex_in_output)

        base_name = tex_in_output.stem
        # Use shell=True for Windows compatibility
        cmd = f'"{pdflatex_path}" -interaction=nonstopmode "{base_name}.tex"'
        logger.debug(f"Running: {cmd}")

        # First pass
        result1 = subprocess.run(cmd, capture_output=True, text=True, timeout=60, shell=True)
        # Second pass
        result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=60, shell=True)

        pdf_path = output_dir / f"{base_name}.pdf"
        combined_log = result1.stdout + result1.stderr + result2.stdout + result2.stderr

        if pdf_path.exists():
            logger.info(f"PDF created: {pdf_path}")
            return CompileResult(
                success=True,
                pdf_path=pdf_path,
                log_output=combined_log,
                error_message=""
            )
        else:
            errors = _extract_latex_errors(combined_log)
            error_msg = "\n".join(errors) if errors else combined_log[:500]
            return CompileResult(
                success=False,
                pdf_path=None,
                log_output=combined_log,
                error_message=f"pdflatex failed: {error_msg}"
            )
    except subprocess.TimeoutExpired:
        return CompileResult(
            success=False,
            pdf_path=None,
            log_output="",
            error_message="Compilation timed out (60 seconds per pass)"
        )
    except Exception as e:
        logger.error(f"pdflatex error: {e}")
        return CompileResult(
            success=False,
            pdf_path=None,
            log_output="",
            error_message=str(e)
        )
    finally:
        os.chdir(original_cwd)


def _extract_latex_errors(log_text: str) -> List[str]:
    errors = []
    lines = log_text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('!'):
            errors.append(line)
            if i + 1 < len(lines) and lines[i+1].strip():
                errors.append(lines[i+1].strip())
        elif 'Error' in line and ('fatal' in line.lower() or 'undefined' in line.lower()):
            errors.append(line)
    return errors[:10]


def compile_tex_to_pdf(tex_path: Path, output_dir: Optional[Path] = None) -> CompileResult:
    if not tex_path.exists():
        return CompileResult(False, None, "", f"TeX file not found: {tex_path}")

    if output_dir is None:
        output_dir = tex_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    compiler = get_available_compiler()
    if compiler is None:
        return CompileResult(
            False, None, "",
            "No LaTeX compiler found. Please install:\n"
            "  - MiKTeX: https://miktex.org/download\n"
            "  - Tectonic: https://tectonic-typesetting.github.io/"
        )

    if compiler == CompilerType.PDFLATEX:
        return compile_with_pdflatex(tex_path, output_dir)
    else:
        return CompileResult(False, None, "", f"Unsupported compiler: {compiler.value}")