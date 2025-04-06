from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
__version__ = "0.0.3" 

setup(
    name="tlama-core",
    version=__version__,  
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
    packages=find_packages(include=["tlamacore", "tlamacore.*"]),  # Specify main packages
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
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
        "einops>=0.6.0",
        "flash-attn>=2.0.0",
        "xformers>=0.0.20",
        "huggingface-hub>=0.16.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.0.0",
        "safetensors>=0.3.0",
        "triton>=2.0.0",
        "rich==13.9.4"
    ],
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