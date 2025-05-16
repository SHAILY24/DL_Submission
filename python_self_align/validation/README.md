# Python Self-Align Validation

This directory contains the validation pipeline for the Python Self-Align project. The validation pipeline ensures that generated Python code meets quality standards by performing both static and dynamic validation.

## Components

- `validation_pipeline.py`: Core validation pipeline that performs static analysis
- `execution_validator.py`: Executes and validates Python code in a sandboxed environment
- `run_validation.py`: Runner script to orchestrate the entire validation process

## Usage

To run the complete validation pipeline:

```bash
python run_validation.py --input_dir /path/to/input --output_dir /path/to/output --max_workers 4
```

### Arguments

- `--input_dir`: Directory containing the input files (must include a `responses.jsonl` file)
- `--output_dir`: Directory where validated outputs and statistics will be saved
- `--max_workers`: Maximum number of parallel workers to use (default: 4)
- `--max_items`: Optional limit on the number of items to process

## Validation Process

The validation pipeline consists of three main stages:

1. **Static Validation**: Checks code syntax, formatting, and basic quality metrics
2. **Execution Validation**: Executes code in a sandboxed environment to verify functionality
3. **Final Validation**: Combines results and produces the final validated dataset

## Output Files

The validation process generates the following files in the output directory:

- `static_validated.jsonl`: Items that passed static validation
- `execution_validated.jsonl`: Items that passed execution validation
- `validated.jsonl`: Final validated dataset
- `validation_stats.json`: Comprehensive validation statistics

## Validation Criteria

The validation pipeline applies the following criteria:

### Static Validation
- Valid Python syntax
- Proper code formatting (PEP 8 compliance)
- Manageable code complexity
- Presence of type annotations
- Function signatures extraction

### Execution Validation
- Successful execution in a sandbox environment
- Execution within time limits
- No runtime errors or exceptions
- Proper test coverage

## Installation

Before using the validators, install the required dependencies:

```bash
pip install black autopep8 radon
```

## Usage

### Execution Validator

The execution validator can be used directly to validate a single JSONL file:

```bash
python execution_validator.py --input_file path/to/input.jsonl --output_file path/to/output.jsonl
```

Additional options:
- `--max_workers`: Maximum number of worker processes (default: 4)
- `--max_items`: Maximum number of items to process (optional)

### Validation Pipeline

The validation pipeline processes all JSONL files in a directory:

```bash
python validation_pipeline.py --input_dir path/to/input_dir --output_dir path/to/output_dir
```

Additional options:
- `--report_path`: Path to write the validation report (default: validation_report.json)
- `--max_workers`: Maximum number of worker processes (default: 4)
- `--max_items`: Maximum number of items to process per file (optional)

## Input/Output Format

### Input Format

The input JSONL files should contain one JSON object per line, with each object containing at least:
- A `response` field with the Python code to validate

Example:
```json
{"concept": "A function to calculate the factorial of a number", "instruction": "Write a Python function to calculate factorial", "response": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)"}
```

### Output Format

The output JSONL files will contain validated items with additional metadata:
- `execution_result`: Success or failure status
- `execution_time`: Time taken to execute the code
- `error_message`: Any error messages (if execution failed)
- `cyclomatic_complexity`: McCabe complexity score
- `has_type_annotations`: Whether the code includes type annotations

### Validation Report

A validation report will be generated in JSON format with:
- Summary statistics (files processed, success rate, etc.)
- Detailed statistics for each file

## Safety Considerations

The execution validator includes safety measures:
- Removes potentially dangerous imports
- Uses timeouts for code execution
- Runs code in a controlled subprocess
- Sanitizes input code

## Examples

### Example 1: Process a single file

```bash
python execution_validator.py --input_file samples/generated_code.jsonl --output_file samples/validated_code.jsonl
```

### Example 2: Process all files in a directory

```bash
python validation_pipeline.py --input_dir samples/ --output_dir validated/ --max_workers 8
```

### Example 3: Process a limited number of items

```bash
python validation_pipeline.py --input_dir samples/ --output_dir validated/ --max_items 100
```

## Customization

To customize the validation process:
- Modify the `DANGEROUS_IMPORTS` list in `execution_validator.py`
- Adjust the complexity thresholds in the `calculate_complexity` function
- Change the execution timeout in the `execute_code` function 