#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python Self-Align Validation Runner

This script orchestrates the complete validation pipeline for Python code samples:
1. Static validation
2. Execution validation
3. Final validation and statistics generation
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from .validation_pipeline import process_directory, aggregate_results
from .execution_validator import process_file, read_jsonl, write_jsonl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def run_validation_pipeline(
    input_file: str,
    output_dir: str,
    max_workers: int = 4,
    max_items: Optional[int] = None,
    skip_static: bool = False,
    skip_execution: bool = False
) -> Dict[str, Any]:
    """
    Run the complete validation pipeline on a single file.
    
    Args:
        input_file: Path to the input file
        output_dir: Directory to save validation outputs
        max_workers: Maximum number of worker processes
        max_items: Maximum number of items to process
        skip_static: Skip static validation
        skip_execution: Skip execution validation
        
    Returns:
        Dictionary with validation statistics
    """
    start_time = time.time()
    
    # Determine file paths
    file_name = os.path.basename(input_file)
    base_name = os.path.splitext(file_name)[0]
    validated_file = os.path.join(output_dir, f"validated_{base_name}.jsonl")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run execution validation
    logger.info(f"Running execution validation on {input_file}")
    
    # Process the file
    process_file(
        input_file=input_file,
        output_file=validated_file,
        max_workers=max_workers,
        max_items=max_items
    )
    
    # Count the results
    validated_items = read_jsonl(validated_file)
    valid_items = sum(1 for item in validated_items if item.get("execution_valid", False))
    total_items = len(validated_items)
    validation_rate = valid_items / total_items if total_items > 0 else 0
    processing_time = time.time() - start_time
    
    # Create stats dictionary
    stats = {
        "input_file": input_file,
        "output_file": validated_file,
        "total_items": total_items,
        "valid_items": valid_items,
        "validation_rate": validation_rate,
        "processing_time": processing_time
    }
    
    # Save statistics
    stats_file = os.path.join(output_dir, f"{base_name}_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Validation complete. Results saved to {validated_file}")
    logger.info(f"Total items: {total_items}, Valid items: {valid_items}, Rate: {validation_rate:.2%}")
    
    return stats

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Python Self-Align validation pipeline")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input files to validate"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where validation results will be stored"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Specific input file to validate (if not using a directory)"
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
        help="Maximum number of items to validate from each file"
    )
    
    parser.add_argument(
        "--skip_static",
        action="store_true",
        help="Skip static validation"
    )
    
    parser.add_argument(
        "--skip_execution",
        action="store_true",
        help="Skip execution validation"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the validation runner."""
    args = parse_args()
    
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input files
    input_files = []
    if args.input_file:
        input_files = [args.input_file]
    else:
        input_dir = Path(args.input_dir)
        input_files = [str(f) for f in input_dir.glob("*.jsonl")]
        
    if not input_files:
        logger.error(f"No input files found in {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} files to validate")
    
    # Process each input file
    for input_file in input_files:
        file_name = os.path.basename(input_file)
        logger.info(f"Processing file: {file_name}")
        
        # Run validation pipeline
        try:
            stats = run_validation_pipeline(
                input_file=input_file,
                output_dir=args.output_dir,
                max_workers=args.max_workers,
                max_items=args.max_items,
                skip_static=args.skip_static,
                skip_execution=args.skip_execution
            )
            
            # Print statistics
            logger.info(f"Validation complete for {file_name}")
            logger.info(f"Total items: {stats['total_items']}")
            logger.info(f"Valid items: {stats['valid_items']}")
            logger.info(f"Validation rate: {stats['validation_rate']:.2%}")
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # --- Add Aggregation Step ---
    logger.info(f"Aggregating results from {args.output_dir}")
    aggregated_file_path = output_dir / "aggregated.jsonl"
    try:
        aggregate_results(str(args.output_dir), str(aggregated_file_path))
        logger.info(f"Aggregation complete. Output saved to {aggregated_file_path}")
    except Exception as e:
        logger.error(f"Error during result aggregation: {e}")
        import traceback
        logger.error(traceback.format_exc())
    # --- End Aggregation Step ---

    total_time = time.time() - start_time
    logger.info(f"Validation pipeline completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 