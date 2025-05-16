"""
Function filtering for high-quality Python functions
"""

import argparse
import logging
import json
import re
import tempfile
import os
import subprocess
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import concurrent.futures
from tqdm.auto import tqdm
from datasets import Dataset, load_from_disk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Words that indicate a function might be of low quality
BANNED_WORDS = {
    'TODO', 'FIXME', 'XXX', 'BUG', 'HACK', 'NOTE',
    'todo', 'fixme', 'xxx', 'bug', 'hack', 'note',
    'DEPRECATED', 'deprecated', 'debug', 'DEBUG',
    'unimplemented', 'UNIMPLEMENTED', 'WIP', 'wip'
}

# Problematic imports that might cause security or execution issues
PROBLEMATIC_IMPORTS = {
    'os', 'sys', 'subprocess', 'pathlib',
    'shutil', 'tempfile', 'pty', 'tty',
    'docker', 'kubernetes', 'socket',
    'multiprocessing', 'threading'
}

# Regex patterns for finding import statements
IMPORT_PATTERN = re.compile(r'^\s*(?:from|import)\s+([a-zA-Z0-9_.]+)', re.MULTILINE)

def function_has_banned_words(function_text: str) -> bool:
    """Check if the function contains any banned words"""
    for word in BANNED_WORDS:
        if word in function_text:
            return True
    return False

def extract_imports(function_text: str) -> Set[str]:
    """Extract all imported modules from a function"""
    imports = set()
    matches = IMPORT_PATTERN.findall(function_text)
    for match in matches:
        # Extract the top-level module
        top_module = match.split('.')[0]
        imports.add(top_module)
    return imports

def has_problematic_imports(function_text: str) -> bool:
    """Check if the function imports any problematic modules"""
    imports = extract_imports(function_text)
    return bool(imports.intersection(PROBLEMATIC_IMPORTS))

def type_check_function(function_text: str) -> bool:
    """
    Type check a Python function using mypy.
    
    Returns:
        True if the function passes type checking, False otherwise
    """
    try:
        # Create a temporary file with the function
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as tmp:
            # Add type ignore for import errors
            header = "# mypy: ignore-missing-imports\n"
            # Add necessary imports
            import_lines = ""
            for import_match in IMPORT_PATTERN.findall(function_text):
                import_lines += f"try:\n    import {import_match.split('.')[0]}\nexcept ImportError:\n    pass\n"
            
            tmp.write(header + import_lines + "\n" + function_text)
            tmp_path = tmp.name
        
        # Run mypy
        result = subprocess.run(
            ["mypy", "--python-version", "3.8", "--no-error-summary", tmp_path],
            capture_output=True,
            text=True
        )
        
        # Delete the temporary file
        os.unlink(tmp_path)
        
        # Check if mypy found type errors
        return "error:" not in result.stdout and result.returncode == 0
    except Exception as e:
        logger.error(f"Error in type checking: {e}")
        return False

def has_return_statement(function_text: str) -> bool:
    """Check if a function contains a return statement"""
    # This is a simple regex check, could be improved with ast parsing
    return bool(re.search(r'\breturn\b', function_text))

def has_args(function_text: str) -> bool:
    """Check if a function has arguments"""
    # Match def function_name(...) with something in the parentheses
    match = re.search(r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\((.*?)\)', function_text)
    if not match:
        return False
    # Check if the args are non-empty, self doesn't count
    args = match.group(1).strip()
    if not args or args == 'self':
        return False
    return True

def check_function_quality(function_text: str) -> bool:
    """Check a function against all quality criteria"""
    # First, quick checks
    if function_has_banned_words(function_text):
        return False
    
    if has_problematic_imports(function_text):
        return False
        
    if not has_return_statement(function_text):
        return False
        
    if not has_args(function_text):
        return False
    
    # More expensive checks
    if not type_check_function(function_text):
        return False
        
    return True

def filter_functions(functions_dataset: Dataset, max_workers: int = 4) -> Dataset:
    """
    Filter functions based on quality criteria.
    
    Args:
        functions_dataset: Dataset containing Python functions
        max_workers: Number of parallel workers
        
    Returns:
        Filtered dataset
    """
    functions = functions_dataset["content"]
    logger.info(f"Filtering {len(functions)} functions...")
    
    quality_functions = []
    quality_ids = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all functions for checking
        future_to_idx = {
            executor.submit(check_function_quality, func): (i, func) 
            for i, func in enumerate(functions)
        }
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx)):
            idx, func = future_to_idx[future]
            try:
                if future.result():  # If the function passes quality checks
                    quality_functions.append(func)
                    quality_ids.append(idx)
            except Exception as e:
                logger.error(f"Error processing function {idx}: {e}")
    
    logger.info(f"Filtered to {len(quality_functions)} high-quality functions")
    
    # Create a new dataset with the filtered functions
    filtered_dataset = Dataset.from_dict({
        "content": quality_functions,
        "id": quality_ids
    })
    
    return filtered_dataset

def main():
    """Main function to run the filtering process"""
    parser = argparse.ArgumentParser(description="Filter Python functions based on quality criteria")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input dataset with Python functions")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the filtered functions")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Load the dataset
    try:
        functions_dataset = load_from_disk(args.input_file)
    except:
        # Try to load as JSONL
        with open(args.input_file, 'r') as f:
            functions = [json.loads(line) for line in f]
        functions_dataset = Dataset.from_list(functions)
    
    # Filter the functions
    filtered_dataset = filter_functions(
        functions_dataset=functions_dataset,
        max_workers=args.max_workers
    )
    
    # Save the filtered dataset
    filtered_dataset.save_to_disk(args.output_file)
    logger.info(f"Saved filtered functions to {args.output_file}")
    
    # Additionally save as JSONL for easier inspection
    output_jsonl = args.output_file + '.jsonl'
    with open(output_jsonl, 'w') as f:
        for i, func in enumerate(filtered_dataset["content"]):
            json.dump({"id": i, "content": func}, f)
            f.write('\n')
    logger.info(f"Saved filtered functions to {output_jsonl}")

if __name__ == "__main__":
    main() 