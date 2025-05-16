#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Python Self-Align pipeline
"""

import os
import subprocess
import argparse
import logging
import sys
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, cwd=None):
    """Run a command and return its output"""
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd
    )
    return process.stdout

def check_output_dir(output_dir):
    """Check that the output directory contains the expected files"""
    files_to_check = [
        "seed_output/filtered_functions.jsonl",
        "self_oss_output/concepts.jsonl",
        "self_oss_output/instructions.jsonl",
        "self_oss_output/responses.jsonl",
        "validation_output/validated_responses.jsonl",
        "final_dataset/instruction_dataset.jsonl",
        "pipeline_report.md"
    ]
    
    missing_files = []
    for file in files_to_check:
        full_path = os.path.join(output_dir, file)
        if not os.path.exists(full_path):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing expected files: {', '.join(missing_files)}")
        return False
    
    logger.info("All expected output files are present")
    return True

def count_items(file_path):
    """Count items in a JSONL file"""
    try:
        with open(file_path, 'r') as f:
            count = sum(1 for line in f if line.strip())
        return count
    except Exception as e:
        logger.error(f"Error counting items in {file_path}: {e}")
        return 0

def test_pipeline(output_dir, num_samples=5):
    """Run a test of the complete pipeline"""
    # Add current directory to PYTHONPATH
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f".{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = "."
    
    # Set working directory to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Run the pipeline
        logger.info(f"Running pipeline with {num_samples} samples")
        cmd = [
            sys.executable,
            "run_full_pipeline.py",
            "--output_dir", output_dir,
            "--num_samples", str(num_samples)
        ]
        
        subprocess.run(cmd, env=env, check=True, cwd=project_root)
        
        # Check output directory
        if not check_output_dir(output_dir):
            return False
        
        # Check counts
        instruction_dataset = os.path.join(output_dir, "final_dataset/instruction_dataset.jsonl")
        count = count_items(instruction_dataset)
        
        if count != num_samples:
            logger.error(f"Expected {num_samples} items in final dataset, but found {count}")
            return False
        
        logger.info(f"Pipeline test completed successfully with {count} items in the final dataset")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(f"Stdout: {e.stdout if e.stdout else 'None'}")
        logger.error(f"Stderr: {e.stderr if e.stderr else 'None'}")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the Python Self-Align pipeline")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_output",
        help="Directory to store test outputs"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate"
    )
    
    args = parser.parse_args()
    
    success = test_pipeline(args.output_dir, args.num_samples)
    
    if success:
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 