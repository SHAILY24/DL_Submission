# DL PROJECT Spring 2025 NJIT - Python Self-Align Pipeline

Python Self-Align is a pipeline for automatically generating instruction-response pairs for fine-tuning code generation models. This project is an implementation of the [StarCoder2 Self-Align](https://github.com/bigcode-project/starcoder2-self-align) methodology, adapted and enhanced for Python code. It leverages advanced parsing techniques, large-scale datasets, and rigorous validation to produce high-quality training data.

This project was developed by Shaily (ss5235@njit.edu) for Professor Phan's DL PROJECT course (DS677004 Deep Learning), Spring 2025, NJIT, Data Science Department.

## Key Features

-   **Advanced Code Parsing**: Utilizes Tree-Sitter for accurate and robust parsing of Python code, enabling precise extraction and analysis of functions.
-   **Large-Scale Data Source**: Leverages "The Stack v2" dataset from Hugging Face, streaming and filtering it efficiently to gather diverse Python code samples.
-   **Multi-Stage Generation Pipeline**:
    1.  **Seed Gathering**: Extracts high-quality Python functions based on configurable criteria.
    2.  **Self-OSS-Instruct**: Transforms seeds into instruction-response pairs through:
        *   S→C (Seed to Concept): Generates a conceptual description of a seed function.
        *   C→I (Concept to Instruction): Converts the concept into a natural language instruction.
        *   I→R (Instruction to Response): Generates a Python code solution given the instruction.
-   **Comprehensive Validation**: Employs a robust validation pipeline that:
    *   Executes code in isolated and resource-limited environments for safety.
    *   Checks for PEP 8 compliance, type hint coverage, docstring quality, and code complexity.
    *   Verifies alignment between the generated code and the original instruction.
-   **GPU Optimization**: Incorporates techniques like 8-bit quantization and batch size optimization for efficient use of GPU resources during model inference.
-   **Checkpointing & Modularity**: Each pipeline stage supports checkpointing, and the modular design allows for individual stage execution.
-   **Detailed Logging & Reporting**: Provides comprehensive logging and generates a summary report of the pipeline run.

## Project Structure

build/                        # Tree-sitter language build artifacts
python_self_align.egg-info/   # Python packaging metadata
__pycache__/                  # Python bytecode cache
pipeline_output_test/         # Example output directory (or user-specified)
*.log                         # Log files

ENV_SETUP.md                  # Guide for setting up the environment
requirements.txt              # Python package dependencies
setup.py                      # Package setup script

run_full_pipeline.py          # Main script to run the entire pipeline
inference_example.py          # Example for running inference (not part of main pipeline)
test_pipeline.py              # Pytest tests for the pipeline

seed_gathering/               # Stage 1: Seed function extraction
  ├── run_seed_gathering.py     # Script to run seed gathering
  ├── stack_v2_connector.py   # Connects to and streams from The Stack v2
  ├── tree_sitter_parser.py   # Parses Python code using Tree-sitter
  └── function_filter.py      # Filters functions based on quality criteria

self_oss_instruct/            # Stage 2: Instruction-response pair generation
  ├── self_oss_instruct.py    # Core logic for S->C, C->I, I->R generation
  ├── prompt_templates.py     # Prompts used for LLM generation
  └── run_self_oss_instruct.py # Script to run this stage

validation/                   # Stage 3: Validation of generated pairs
  ├── run_validation.py       # Script to run validation
  ├── validation_pipeline.py  # Orchestrates validation checks
  └── execution_validator.py  # Executes code safely and performs quality checks

util/                         # Utility scripts
  ├── common.py               # Common helper functions
  └── env_loader.py           # Loads environment variables (e.g., HF_TOKEN)

# Technical Notes Summary
The `presentation.md` file within this directory contains detailed technical notes on Tree-Sitter integration, Stack v2 usage, GPU optimization techniques, and the validation pipeline.

## Prerequisites

1.  **Python Environment**: Python 3.9+ is recommended.
2.  **Tree-Sitter**: You need to have a C compiler installed to build the Tree-sitter Python grammar. Refer to `ENV_SETUP.md` for detailed OS-specific instructions.
3.  **Hugging Face Token**: To access models like StarCoder2 and datasets like The Stack v2, you'll need a Hugging Face Hub token with appropriate permissions.
    *   Set it as an environment variable: `export HF_TOKEN=your_token_here`
    *   Or, create a `.env` file in the `python_self_align` directory with the line: `HF_TOKEN=your_token_here`

## Installation

1.  Ensure the project files are extracted/copied to your local system. Navigate into the `python_self_align` directory:
    ```bash
    cd path/to/python_self_align 
    ```

2.  It's highly recommended to create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    This will also attempt to build the Tree-sitter grammar. If it fails, please consult `ENV_SETUP.md`.

## Usage

### Running the Full Pipeline

To run the complete pipeline from the root directory of the project (i.e., the directory containing `python_self_align` and `pipeline_output_test`):

```bash
python -m python_self_align.run_full_pipeline --output_dir pipeline_output_test --num_samples 5000 --clean
```

**Arguments:**
*   `--output_dir`: (Required) Directory to store all pipeline outputs (e.g., `pipeline_output_test`).
*   `--num_samples`: Number of seed samples to gather (e.g., `5000`). Default is `20`.
*   `--clean`: If specified, cleans the output directory before running.
*   `--batch_size`: Batch size for the Self-OSS-Instruct generation steps. Default is `4`.
*   `--delay`: Delay in seconds between batches in Self-OSS-Instruct. Default is `0.5`.

This command will execute the following stages:
1.  **Seed Generation**: Gathers and filters Python functions from The Stack v2. Output in `<output_dir>/seed_output/`.
2.  **Self-OSS-Instruct**: Generates concepts, instructions, and responses. Output in `<output_dir>/self_oss_output/`.
3.  **Validation**: Validates the generated instruction-response pairs. Output in `<output_dir>/validation_output/`.
4.  **Final Dataset Creation**: Creates the final `instruction_dataset.jsonl`. Output in `<output_dir>/final_dataset/`.
5.  **Summary Report**: Generates `pipeline_report.md` in `<output_dir>`.

### Running Individual Stages

The pipeline is modular, and individual stages can be run if needed. Scripts for each stage are located in their respective directories (`seed_gathering/`, `self_oss_instruct/`, `validation/`). Consult the `run_full_pipeline.py` script and the individual stage runners for specific arguments.

## Pipeline Output Structure

The specified `--output_dir` (e.g., `pipeline_output_test/`) will contain the following subdirectories after a successful run:

*   `seed_output/`: Contains `filtered_functions.jsonl` (the gathered seed functions) and `stats.json`.
*   `self_oss_output/`: Contains intermediate files from the generation process:
    *   `concepts.jsonl`
    *   `instructions.jsonl`
    *   `responses.jsonl`
*   `validation_output/`: Contains detailed validation results for each response and aggregated statistics like `validation_stats.json`.
*   `final_dataset/`: Contains the final `instruction_dataset.jsonl` ready for model fine-tuning, and `stats.json`.
*   `pipeline.log`: Detailed log file for the pipeline run.
*   `pipeline_report.md`: A summary report of the pipeline execution and statistics.

## Technical Details

This project incorporates several advanced techniques for robust and high-quality data generation:

### Tree-Sitter Integration
Tree-Sitter is used for parsing Python code, offering superior accuracy over regex:
-   **Language-aware parsing**: Understands Python's grammar.
-   **Robustness**: Handles complex syntax, multiline constructs.
-   **Precise targeting**: Uses a query language to find specific code elements (functions, docstrings, parameters, type annotations, return statements).
-   **Enhanced Queries**: Basic queries are enhanced to filter for higher-quality functions by checking for docstrings, type annotations, return statements, and applying complexity filtering.
    *Example query for functions with docstrings:*
    ```
    (function_definition
      body: (block
        (expression_statement
          (string) @docstring)))
    ```

### The Stack v2 Integration
The Stack v2 dataset is accessed via the Hugging Face `datasets` API:
-   **Efficient Streaming**: Uses `streaming=True` to avoid downloading the entire dataset.
-   **Filtering**: Applies criteria to select high-quality Python files (well-formed syntax, docstrings, type hints, appropriate complexity, clear imports, return statements, self-contained).
-   **Oversampling**: Fetches more samples than needed to account for filtering.

### GPU Optimization Techniques
For the language model inference stages (S->C, C->I, I->R):
-   **8-bit Quantization**: Reduces memory footprint significantly with minimal impact on code generation quality. Achieved using `quantization_config=True` with `AutoModelForCausalLM`.
-   **Batch Size Optimization**: Heuristics are used to determine appropriate batch sizes based on available VRAM.
-   **Memory-Efficient Generation**: Uses `@torch.inference_mode()` and optimizes `model.generate()` parameters.
-   **Trade-offs**: Balances speed, memory, and precision (e.g., quantization reduces VRAM but can slightly decrease quality or speed up individual inferences while allowing larger batches for overall throughput).

### Quality Assurance and Validation
A multi-faceted validation approach ensures the quality of generated instruction-response pairs:
-   **Safe Code Execution**:
    *   Uses separate, isolated subprocesses.
    *   Implements resource limits (time, memory, CPU).
    *   Restricts namespaces, imports, filesystem access, and network access.
-   **Comprehensive Metrics**:
    *   **PEP 8 Compliance**: Checked using `pycodestyle`.
    *   **Type Annotation Coverage**: Measures hints for parameters and returns.
    *   **Docstring Quality**: Evaluates against PEP 257.
    *   **Cyclomatic Complexity**: Identifies overly complex code.
    *   **Instruction Alignment**: Checks if the generated code correctly addresses the instruction.

## Contributing

This project was developed as part of the DS677004 Deep Learning course at NJIT.

## Acknowledgments

- This project is based on the [StarCoder2 Self-Align](https://github.com/bigcode-project/starcoder2-self-align) pipeline by the BigCode project.
- The Stack v2 dataset is used for seed gathering.

For questions or feedback regarding this specific implementation for the DL PROJECT, please contact us.