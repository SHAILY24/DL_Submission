# Python Self-Align: Automatic Generation of Instruction-Response Pairs
DS677004 Deep Learning - NJIT Spring 2025

*A Pipeline for Creating Instruction-Tuning Datasets for Code LLMs*

---

## Project Overview

The **Python Self-Align** pipeline, developed for Professor Phan's DL PROJECT course, automates the creation of high-quality instruction-response pairs for fine-tuning code generation models without requiring human annotations.

Key objectives:
- Implement the StarCoder2 Self-Align methodology for Python.
- Leverage advanced parsing (Tree-Sitter) and large datasets (The Stack v2).
- Ensure data quality through a rigorous, multi-faceted validation process.
- Optimize for GPU efficiency during language model inference.

---

## Pipeline Architecture

Our pipeline consists of three main stages, executed sequentially:

1.  **Seed Gathering**: Extracts high-quality Python functions from a large corpus.
2.  **Self-OSS-Instruct**: Transforms these seed functions into instruction-response pairs using a Large Language Model (LLM).
3.  **Validation**: Critically assesses the generated pairs for correctness, safety, and quality.

The pipeline is designed to be modular, with checkpointing at each major step.

---

## Stage 1: Seed Gathering

This stage focuses on sourcing initial high-quality Python code.

-   **Data Source**: Utilizes **The Stack v2**, a massive dataset of source code, accessed efficiently via the Hugging Face `datasets` API in streaming mode.
-   **Parsing with Tree-Sitter**:
    -   Python code is parsed using **Tree-Sitter**, which builds an accurate Abstract Syntax Tree (AST). This is superior to regex for understanding code structure.
    -   Tree-Sitter's query language allows precise targeting of code elements like functions, docstrings, parameters, and type annotations.
-   **Quality Filtering Criteria**:
    -   Well-formed syntax (must parse correctly).
    -   Presence of comprehensive docstrings (following PEP 257).
    -   Existence of type annotations for parameters and return types.
    -   Non-trivial but not overly complex functions (e.g., 5-100 lines).
    -   Clear, well-structured imports.
    -   Presence of explicit return statements.
    -   Preference for self-contained functions with minimal external dependencies.
-   **Output**: `filtered_functions.jsonl` containing the selected seed functions.

---

## Stage 2: Self-OSS-Instruct

This stage uses a pre-trained LLM (e.g., `bigcode/starcoder2-15b`) to generate data.

1.  **S→C (Seed to Concept)**:
    -   Input: A seed Python function with its docstring.
    -   Process: The LLM generates a concise, high-level natural language concept describing the function's purpose and core logic.
    -   Output: `concepts.jsonl`

2.  **C→I (Concept to Instruction)**:
    -   Input: The generated natural language concept.
    -   Process: The LLM transforms the concept into a detailed, actionable coding instruction or prompt that a developer could follow.
    -   Output: `instructions.jsonl`

3.  **I→R (Instruction to Response)**:
    -   Input: The generated coding instruction.
    -   Process: The LLM generates a Python code implementation that aims to fulfill the instruction.
    -   Output: `responses.jsonl`

---

## GPU Optimization Techniques

To manage resources during LLM inference in Stage 2:

-   **8-bit Quantization**:
    -   Models (e.g., StarCoder2 15B) are loaded with `quantization_config=True`.
    -   Reduces VRAM usage significantly (e.g., ~30GB to ~12GB for 15B models) with minimal impact on code generation quality for this task.
-   **Batch Size Optimization**:
    -   Batch sizes are determined based on available VRAM (heuristics provided for common GPU types like A100, RTX 3090/4090).
    -   Larger batches improve throughput.
-   **Memory-Efficient Generation**:
    -   Utilizes `@torch.inference_mode()` which is more memory-efficient than `torch.no_grad()`.
    -   Generation parameters like `output_scores=False` are used when scores are not needed.
-   **KV-Caching**: Leveraged by default in Hugging Face Transformers for faster generation, though it consumes memory.

---

## Stage 3: Validation

Ensures the quality, correctness, and safety of generated instruction-response pairs.

-   **Safe Code Execution Environment**:
    -   Generated code is run in **isolated subprocesses**.
    -   **Resource limitations**: Time limits (e.g., 5 seconds), memory limits (e.g., 256MB), and CPU usage caps are enforced.
    -   **Namespace isolation** and **restricted imports** prevent access to sensitive modules or operations.
    -   **Filesystem sandboxing** and **network isolation**.
-   **Comprehensive Quality Metrics**:
    -   **Execution Success**: Did the code run without errors?
    -   **PEP 8 Compliance**: Checked using `pycodestyle`.
    -   **Type Annotation Coverage**: Percentage of parameters and return values with type hints.
    -   **Docstring Quality**: Evaluated against PEP 257 for completeness and format.
    -   **Code Complexity**: Measured using metrics like cyclomatic complexity to filter overly complex or trivial code.
    -   **Instruction Alignment**: Semantic check to ensure the generated code fulfills the original instruction (this can be a challenging automated task, often relying on heuristic checks or proxy metrics).
-   **Output**: Validated and filtered instruction-response pairs, forming the `instruction_dataset.jsonl`.

---

## Implementation Details Summary

-   **Programming Language**: Python
-   **Core Libraries**: Hugging Face `transformers`, `datasets`, `tree-sitter`, `torch`.
-   **Modularity**: Code is organized into stages (`seed_gathering`, `self_oss_instruct`, `validation`).
-   **Configuration**: Key parameters (model names, batch sizes, paths) are configurable.
-   **Logging & Reporting**: Detailed logs (`pipeline.log`) and a summary markdown report (`pipeline_report.md`) are generated.

---

## Results & Example Output

The pipeline generates an `instruction_dataset.jsonl` file. The example `pipeline_output_test` directory (generated from a run with 5000 initial samples) showcases the typical outputs of each stage, including intermediate files and the final dataset.

**Example Instruction-Response Pair:**

**Instruction:**
```
Write a Python function called `calculate_triangle_area` that takes the base and height of a triangle as inputs and returns its area. Ensure the function includes type hints for arguments and the return value, and a docstring explaining its purpose, arguments, and what it returns.
```

**Response:**
```python
def calculate_triangle_area(base: float, height: float) -> float:
    """
    Calculates the area of a triangle given its base and height.

    Args:
        base (float): The base of the triangle.
        height (float): The height of thetriangle.

    Returns:
        float: The area of the triangle.
    """
    return 0.5 * base * height
```

---

## Future Work

Potential future directions for this project could include:

-   Direct integration with model fine-tuning scripts.
-   Support for additional programming languages by extending Tree-Sitter grammars and filtering logic.
-   More sophisticated metrics for instruction alignment and semantic code quality.
-   Automated evaluation of the generated dataset's impact on downstream model performance.

---

## Conclusion

The Python Self-Align pipeline provides a robust and extensible framework for generating instruction-response datasets for code LLMs. By automating data creation with a focus on quality and safety, this project contributes to the development of more capable and specialized code generation models. This work successfully demonstrates the core principles of the Self-Align methodology applied to Python.

---

## Thank You!

This project was undertaken as part of the DS677004 Deep Learning course at NJIT.