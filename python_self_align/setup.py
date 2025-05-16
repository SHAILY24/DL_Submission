from setuptools import setup, find_packages

setup(
    name="python_self_align",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "datasets>=2.18.0",
        "huggingface-hub>=0.21.0",
        "transformers>=4.36.0",
        "vllm>=0.3.3",
        "openai>=1.10.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "smart_open>=6.4.0",
        "boto3>=1.28.0",
        "tree-sitter>=0.20.0",
        "black>=23.0.0",
        "mypy>=1.7.0",
        "pylint>=3.0.0",
        "jsonlines>=3.1.0",
        "tiktoken>=0.3.0",
        "tenacity>=8.2.0",
        "pyright>=1.1.0"
    ],
    author="NJIT DL Project Team",
    author_email="ss5235@njit.edu",
    description="Python Self-Align: Automatic generation of instruction-response pairs for fine-tuning",
    python_requires=">=3.8",
) 