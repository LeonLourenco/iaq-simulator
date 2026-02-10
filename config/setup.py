"""
Setup configuration for IAQ Simulator
"""

from setuptools import setup, find_packages
import os

# Ler README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Ler requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="iaq-simulator",
    version="1.0.0",
    author="IAQ Simulator Team",
    author_email="contact@iaq-simulator.org",
    description="Simulador avançado de qualidade do ar interno com CFD e agentes inteligentes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/iaq-simulator",
    project_urls={
        "Bug Tracker": "https://github.com/seu-usuario/iaq-simulator/issues",
        "Documentation": "https://github.com/seu-usuario/iaq-simulator/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pylint>=2.12.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iaq-simulate=run_simulation:main",
            "iaq-dashboard=final_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["data/scenarios/*.json", "data/materials/*.json"],
    },
    zip_safe=False,
)
