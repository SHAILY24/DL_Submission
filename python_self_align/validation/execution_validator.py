#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Execution Validator for Python Self-Align

This module validates Python code by executing it in a safe environment and
checking for correctness, complexity, and quality.
"""

import os
import sys
import re
import ast
import json
import time
import tempfile
import subprocess
import concurrent.futures
import black
import logging
from typing import Dict, List, Any, Tuple, Optional
import argparse
import traceback
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Timeout for code execution in seconds
EXECUTION_TIMEOUT = 5

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read a jsonl file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Write a list of dictionaries to a jsonl file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def sanitize_code(code: str) -> str:
    """
    Sanitize Python code by removing potentially harmful operations.
    
    Args:
        code: The Python code to sanitize
        
    Returns:
        Sanitized Python code
    """
    # Remove imports that might be harmful
    dangerous_imports = [
        "os", "sys", "subprocess", "importlib", "socket", 
        "shutil", "pathlib", "pickle", "requests"
    ]
    
    # Replace dangerous imports with comments
    for imp in dangerous_imports:
        code = re.sub(
            fr"(^|\n)(\s*)import\s+{imp}.*?($|\n)",
            r"\1\2# Import removed for security: {}\3".format(imp),
            code
        )
        code = re.sub(
            fr"(^|\n)(\s*)from\s+{imp}\s+import.*?($|\n)",
            r"\1\2# Import removed for security: {}\3".format(imp),
            code
        )
    
    # Remove file operations
    code = re.sub(r"open\s*\(", "# open(", code)
    
    # Remove eval and exec functions
    code = re.sub(r"eval\s*\(", "# eval(", code)
    code = re.sub(r"exec\s*\(", "# exec(", code)
    
    return code

def format_code(code: str) -> str:
    """
    Format Python code using black.
    
    Args:
        code: The Python code to format
        
    Returns:
        Formatted Python code or original code if formatting fails
    """
    try:
        return black.format_str(code, mode=black.Mode())
    except Exception as e:
        logger.warning(f"Code formatting failed: {e}")
        return code

def extract_function_name(code: str) -> Optional[str]:
    """
    Extract the main function name from Python code.
    
    Args:
        code: The Python code to analyze
        
    Returns:
        The name of the main function or None if not found
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
        return None
    except Exception as e:
        logger.error(f"Failed to extract function name: {e}")
        return None

def create_test_script(code: str, function_name: Optional[str]) -> str:
    """
    Create a test script that executes the function with some sample inputs.
    
    Args:
        code: The Python code containing the function
        function_name: The name of the function to test
        
    Returns:
        A test script that imports and calls the function
    """
    if not function_name:
        return code
    
    test_script = f"""
{code}

# Test the function with some basic inputs
if __name__ == "__main__":
    try:
        import inspect
        sig = inspect.signature({function_name})
        param_count = len(sig.parameters)
        
        # Create some test values based on parameter count
        if param_count == 0:
            result = {function_name}()
            print(f"Result: {{result}}")
        elif param_count == 1:
            result = {function_name}(5)
            print(f"Result: {{result}}")
        elif param_count == 2:
            result = {function_name}(5, 10)
            print(f"Result: {{result}}")
        else:
            args = [1] * param_count
            result = {function_name}(*args)
            print(f"Result: {{result}}")
    except Exception as e:
        print(f"Test execution failed: {{e}}")
    """
    
    return test_script

def execute_code(code: str) -> Tuple[bool, str, float]:
    """
    Execute Python code in a safe environment and capture the output.
    
    Args:
        code: The Python code to execute
        
    Returns:
        A tuple containing (success_flag, output, execution_time)
    """
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
        tmp.write(code.encode('utf-8'))
        tmp_name = tmp.name
    
    start_time = time.time()
    try:
        # Execute the code in a separate process with timeout
        result = subprocess.run(
            [sys.executable, tmp_name],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT
        )
        success = result.returncode == 0
        output = result.stdout if success else result.stderr
        
    except subprocess.TimeoutExpired:
        success = False
        output = f"Execution timed out after {EXECUTION_TIMEOUT} seconds"
    except Exception as e:
        success = False
        output = f"Execution failed: {str(e)}"
    finally:
        execution_time = time.time() - start_time
        # Clean up the temporary file
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
    
    return success, output, execution_time

def calculate_complexity(code: str) -> int:
    """
    Calculate the cyclomatic complexity of the code.
    
    Args:
        code: The Python code to analyze
        
    Returns:
        The cyclomatic complexity score
    """
    try:
        tree = ast.parse(code)
        complexity = 1  # Base complexity
        
        # Count branches that increase complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                complexity += len(node.values) - 1
        
        return complexity
    except Exception as e:
        logger.error(f"Failed to calculate complexity: {e}")
        return 100  # High value to indicate a problem

def has_type_annotations(code: str) -> bool:
    """
    Check if the code contains type annotations.
    
    Args:
        code: The Python code to analyze
        
    Returns:
        True if the code has type annotations, False otherwise
    """
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            # Check function definitions for annotations
            if isinstance(node, ast.FunctionDef):
                # Check return type annotation
                if node.returns is not None:
                    return True
                    
                # Check parameter annotations
                for arg in node.args.args:
                    if arg.annotation is not None:
                        return True
                        
            # Check variable annotations
            elif isinstance(node, ast.AnnAssign):
                return True
                
        return False
    except Exception as e:
        logger.error(f"Failed to check type annotations: {e}")
        return False

def validate_code(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single code item by executing it and collecting metrics.
    
    Args:
        item: A dictionary containing the code to validate
        
    Returns:
        The input dictionary with additional validation fields
    """
    result = item.copy()
    code = item.get("response", "")
    
    if not code or not isinstance(code, str):
        result["execution_valid"] = False
        result["execution_output"] = "No code provided"
        return result
    
    # Sanitize and format the code
    sanitized_code = sanitize_code(code)
    formatted_code = format_code(sanitized_code)
    
    # Extract function name and create test script
    function_name = extract_function_name(sanitized_code)
    test_script = create_test_script(sanitized_code, function_name)
    
    # Execute the code
    success, output, execution_time = execute_code(test_script)
    
    # Calculate code complexity
    complexity = calculate_complexity(sanitized_code)
    has_types = has_type_annotations(sanitized_code)
    
    # Update the result with validation information
    result["execution_valid"] = success
    result["execution_output"] = output
    result["execution_time"] = execution_time
    result["complexity"] = complexity
    result["has_type_annotations"] = has_types
    result["function_name"] = function_name
    
    return result

def validate_items(items: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Validate a list of code items in parallel.
    
    Args:
        items: A list of dictionaries containing code to validate
        max_workers: Maximum number of parallel workers
        
    Returns:
        A list of dictionaries with additional validation fields
    """
    results = []
    total_items = len(items)
    
    logger.info(f"Starting execution validation for {total_items} items with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(validate_code, items)):
            results.append(result)
            if (i + 1) % 100 == 0 or (i + 1) == total_items:
                logger.info(f"Processed {i + 1}/{total_items} items ({(i + 1) / total_items * 100:.1f}%)")
    
    return results

def process_file(input_file: str, output_file: str, max_workers: int = 4, max_items: Optional[int] = None) -> None:
    """
    Process a JSONL file containing code items, validate them, and write results.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSONL file
        max_workers: Maximum number of parallel workers
        max_items: Maximum number of items to process (None for all)
    """
    logger.info(f"Reading items from {input_file}...")
    items = read_jsonl(input_file)
    
    if max_items is not None:
        items = items[:max_items]
        logger.info(f"Limited to processing {max_items} items")
    
    validated_items = validate_items(items, max_workers)
    
    # Count valid items
    valid_count = sum(1 for item in validated_items if item.get("execution_valid", False))
    logger.info(f"Validation complete. {valid_count} out of {len(validated_items)} items passed execution validation.")
    
    # Write validated items to output file
    logger.info(f"Writing validated items to {output_file}...")
    write_jsonl(validated_items, output_file)
    
    logger.info(f"Processing complete. Results written to {output_file}")

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Validate Python code through execution")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Input JSONL file containing code items"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Output JSONL file for validated items"
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=4, 
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--max_items", 
        type=int, 
        default=None, 
        help="Maximum number of items to process"
    )
    
    args = parser.parse_args()
    
    try:
        process_file(
            args.input_file,
            args.output_file,
            args.max_workers,
            args.max_items
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 