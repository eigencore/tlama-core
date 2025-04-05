from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="tlama-core",
    version="0.0.1",
    author="Eigen Core",  
    author_email="main@eigencore.org",  
    description="Core library for training Tlama models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eigencore/tlama-core",  # Your repository URL
    project_urls={
        "Bug Tracker": "https://github.com/eigencore/tlama-core/issues",
        "Documentation": "https://eigen-core.gitbook.io/tlama-core-docs",
        "Source Code": "https://github.com/eigencore/tlama-core",
    },
    packages=find_packages(include=["tlama_core", "tlama_core.*"]),  # Specify main packages
    classifiers=[
        "Development Status :: 3 - Alpha",  # Indicates it's an initial version
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Minimum compatible Python version
    install_requires=requirements,  # Install required packages
    # extras_require={ # TODO: Add optional dependencies
    #     "dev": [
    #         "pytest>=6.0",
    #         "black",
    #         "isort",
    #         "flake8",
    #     ],
    #     "docs": [
    #         "sphinx>=4.0.0",
    #         "sphinx-rtd-theme",
    #     ],
    # },
    keywords="llama, transformers, nlp, machine learning, deep learning",
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    zip_safe=False,  # Better for debugging
)