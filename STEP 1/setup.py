# Package setup configuration
#!/usr/bin/env python3
"""
Setup configuration for the Elephant Corridor Analysis Project
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return "Elephant Corridor Prediction using Energy Landscapes and Circuit Theory"

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            requirements = [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith("#")
            ]
    return requirements

setup(
    name="elephant-corridors",
    version="0.1.0",
    description="Elephant Corridor Prediction using Energy Landscapes and Circuit Theory",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Guillaume Atencia",
    author_email="your.email@example.com",  # Update with your email
    url="https://github.com/yourusername/elephant-corridors",  # Update with your repo
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
        "dash": [
            "dash>=2.11.0",
            "plotly>=5.17.0",
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "elephant-corridors=elephant_corridors.cli:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Include additional files
    include_package_data=True,
    package_data={
        "elephant_corridors": [
            "config/*.yml",
            "data/sample/*",
        ],
    },
    
    # Keywords for PyPI
    keywords=[
        "elephant", "corridors", "conservation", "gis", "ecology",
        "movement ecology", "circuitscape", "energy landscapes"
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/elephant-corridors/issues",
        "Source": "https://github.com/yourusername/elephant-corridors",
        "Documentation": "https://elephant-corridors.readthedocs.io/",
    },
)