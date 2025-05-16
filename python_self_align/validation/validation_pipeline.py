#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation pipeline for Python code generated in self-alignment pipeline.
This module processes directories of generated code, validates execution,
and aggregates results.
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .execution_validator import process_file, read_jsonl, write_jsonl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def find_jsonl_files(directory: str) -> List[str]:
    """
    Find all JSONL files in a directory.
    
    Args:
        directory: Path to the directory
        
    Returns:
        List of paths to JSONL files
    """
    files = []
    try:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.append(os.path.join(root, filename))
        
        logger.info(f"Found {len(files)} JSONL files in {directory}")
        return files
    except Exception as e:
        logger.error(f"Error finding JSONL files in {directory}: {e}")
        return []

def process_directory(
    input_dir: str,
    output_dir: str,
    max_workers: int = 4,
    max_items: Optional[int] = None,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process all JSONL files in a directory and write the results to a new directory.
    
    Args:
        input_dir: Path to the input directory
        output_dir: Path to the output directory
        max_workers: Maximum number of worker processes
        max_items: Maximum number of items to process per file
        max_files: Maximum number of files to process
        
    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()
    
    # Find all JSONL files
    input_files = find_jsonl_files(input_dir)
    
    if not input_files:
        logger.warning(f"No JSONL files found in {input_dir}")
        return {
            "total_files": 0,
            "total_items": 0,
            "valid_items": 0,
            "success_rate": 0,
            "processing_time": 0,
        }
    
    # Limit the number of files if requested
    if max_files is not None:
        input_files = input_files[:max_files]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    file_stats = []
    for input_file in input_files:
        # Determine the output file path
        rel_path = os.path.relpath(input_file, input_dir)
        output_file = os.path.join(output_dir, rel_path)
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Process the file
        logger.info(f"Processing {input_file}")
        stats = process_file(input_file, output_file, max_workers, max_items)
        stats["input_file"] = input_file
        stats["output_file"] = output_file
        file_stats.append(stats)
    
    # Aggregate statistics
    total_items = sum(stats["total_items"] for stats in file_stats)
    valid_items = sum(stats["valid_items"] for stats in file_stats)
    success_rate = valid_items / total_items if total_items > 0 else 0
    processing_time = time.time() - start_time
    
    aggregated_stats = {
        "total_files": len(input_files),
        "total_items": total_items,
        "valid_items": valid_items,
        "success_rate": success_rate,
        "processing_time": processing_time,
        "file_stats": file_stats,
    }
    
    # Write statistics to a JSON file
    stats_file = os.path.join(output_dir, "validation_stats.json")
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(aggregated_stats, f, indent=2)
    except Exception as e:
        logger.error(f"Error writing statistics to {stats_file}: {e}")
    
    logger.info(f"Processed {len(input_files)} files with {total_items} total items")
    logger.info(f"Found {valid_items} valid items ({success_rate:.2%} success rate)")
    
    return aggregated_stats

def aggregate_results(output_dir: str, output_file: str) -> None:
    """
    Aggregate all valid items from processed files into a single file.
    
    Args:
        output_dir: Path to the directory containing processed files
        output_file: Path to the output file for aggregated results
    """
    try:
        # Find all JSONL files in the output directory
        output_files = find_jsonl_files(output_dir)
        
        # Exclude the aggregated file if it already exists
        if output_file in output_files:
            output_files.remove(output_file)
        
        # Read all items from all files
        all_items = []
        for file_path in output_files:
            items = read_jsonl(file_path)
            all_items.extend(items)
        
        # Write all items to the output file
        write_jsonl(all_items, output_file)
        
        logger.info(f"Aggregated {len(all_items)} items from {len(output_files)} files to {output_file}")
    except Exception as e:
        logger.error(f"Error aggregating results: {e}")

def generate_summary(stats: Dict[str, Any], output_file: str) -> None:
    """
    Generate a summary report of the validation results.
    
    Args:
        stats: Dictionary with validation statistics
        output_file: Path to the output file for the summary report
    """
    try:
        # Calculate additional statistics
        valid_items = stats["valid_items"]
        total_items = stats["total_items"]
        success_rate = stats["success_rate"]
        processing_time = stats["processing_time"]
        
        # Format the summary report
        summary = [
            "# Validation Summary Report",
            "",
            f"## Overview",
            f"- Total files processed: {stats['total_files']}",
            f"- Total items processed: {total_items}",
            f"- Valid items: {valid_items}",
            f"- Success rate: {success_rate:.2%}",
            f"- Processing time: {processing_time:.2f} seconds",
            "",
            "## File Details",
            "",
        ]
        
        # Add file-specific statistics
        for file_stat in stats.get("file_stats", []):
            input_file = file_stat["input_file"]
            file_success_rate = file_stat["success_rate"]
            summary.append(f"### {os.path.basename(input_file)}")
            summary.append(f"- Items: {file_stat['total_items']}")
            summary.append(f"- Valid: {file_stat['valid_items']}")
            summary.append(f"- Success Rate: {file_success_rate:.2%}")
            summary.append("")
        
        # Write the summary to a file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(summary))
        
        logger.info(f"Generated summary report at {output_file}")
    except Exception as e:
        logger.error(f"Error generating summary: {e}")

def main():
    """Main function to run the validation pipeline."""
    parser = argparse.ArgumentParser(description="Validation pipeline for Python code")
    
    # Input/output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_file", type=str, help="Path to the input JSONL file")
    input_group.add_argument("--input_dir", type=str, help="Path to the input directory")
    
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output_file", type=str, help="Path to the output JSONL file")
    output_group.add_argument("--output_dir", type=str, help="Path to the output directory")
    
    # Processing options
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker processes")
    parser.add_argument("--max_items", type=int, help="Maximum number of items to process per file")
    parser.add_argument("--max_files", type=int, help="Maximum number of files to process")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results into a single file")
    parser.add_argument("--summary_file", type=str, help="Path to the summary report file")
    
    args = parser.parse_args()
    
    try:
        # Process a single file
        if args.input_file and args.output_file:
            stats = process_file(
                args.input_file,
                args.output_file,
                args.max_workers,
                args.max_items,
            )
            
            if args.summary_file:
                generate_summary(
                    {
                        "total_files": 1,
                        "total_items": stats["total_items"],
                        "valid_items": stats["valid_items"],
                        "success_rate": stats["success_rate"],
                        "processing_time": stats["processing_time"],
                        "file_stats": [stats],
                    },
                    args.summary_file,
                )
        
        # Process a directory
        elif args.input_dir and args.output_dir:
            stats = process_directory(
                args.input_dir,
                args.output_dir,
                args.max_workers,
                args.max_items,
                args.max_files,
            )
            
            # Aggregate results if requested
            if args.aggregate:
                aggregate_file = os.path.join(args.output_dir, "aggregated.jsonl")
                aggregate_results(args.output_dir, aggregate_file)
            
            # Generate summary report if requested
            if args.summary_file:
                generate_summary(stats, args.summary_file)
        
        # Print final message
        logger.info("Validation pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main() 