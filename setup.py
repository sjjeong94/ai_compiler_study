from pathlib import Path

from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aicom",
    version="0.1.0",
    author="Sinjin Jeong",
    description="AI compiler study with triton",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sjjeong94/ai_compiler_study",
    packages=[
        "aicom",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "numpy",
        "torch",
        "triton>=2.2.0",
    ],
    extras_require={
        "linting": [
            "pre-commit>=3.5.0",
        ],
        "testing": [
            "pytest>=7.4.0",
            "pytest-xdist>=3.5.0",
        ],
    },
    python_requires=">=3.8",
)
