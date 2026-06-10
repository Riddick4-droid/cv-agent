from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).parent

long_description = (root / "README.md").read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="cv-agent",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="Multi-agent workflow for CV revamp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cv-agent",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(include=["cv_agent", "cv_agent.*"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)

