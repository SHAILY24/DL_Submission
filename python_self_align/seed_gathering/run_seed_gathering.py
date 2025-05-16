"""
Main script for the seed gathering stage of the Python self-align pipeline
"""

import os
import argparse
import logging
import json
from typing import Optional
from pathlib import Path

from .stack_v2_connector import StackV2Connector
from .function_filter import filter_functions
from datasets import load_from_disk

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_seed_gathering_pipeline(
    output_dir: str,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    num_workers: int = os.cpu_count() or 4,
    chunk_size: int = 1000
):
    """
    Run the complete seed gathering pipeline.
    
    Args:
        output_dir: Directory to save output files
        cache_dir: Directory to cache the dataset
        max_samples: Maximum number of samples to process
        num_workers: Number of workers for parallel processing
        chunk_size: Size of chunks for parallel processing
    """
    logger.info(f"Starting seed gathering pipeline. Output Dir: {output_dir}, Max Samples: {max_samples}, Workers: {num_workers}")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract functions with docstrings from The Stack v2
    logger.info("Step 1: Extracting functions with docstrings")
    raw_functions_path = os.path.join(output_dir, "raw_functions")
    
    try:
        logger.info("Initializing StackV2Connector...")
        connector = StackV2Connector(num_workers=num_workers)
        logger.info("Loading dataset...")
        dataset = connector.load_dataset(cache_dir=cache_dir)
        logger.info(f"Dataset loaded: {dataset}")
        logger.info(f"Starting parallel processing to {raw_functions_path}...")
        connector.process_parallel(
            dataset=dataset,
            output_file=raw_functions_path,
            chunk_size=chunk_size,
            max_samples=max_samples
        )
        logger.info("Parallel processing finished.")
    except Exception as e:
        logger.error(f"Error during Step 1 (Extraction): {e}", exc_info=True)
        raise
    
    # Step 2: Filter the functions for quality
    logger.info("Step 2: Filtering functions for quality")
    filtered_functions_path = os.path.join(output_dir, "filtered_functions")
    
    try:
        logger.info(f"Loading raw functions from disk: {raw_functions_path}")
        raw_functions = load_from_disk(raw_functions_path)
        logger.info(f"Loaded {len(raw_functions)} raw functions.")
        logger.info(f"Starting filtering with {num_workers} workers...")
        filtered_functions = filter_functions(
            functions_dataset=raw_functions,
            max_workers=num_workers
        )
        logger.info(f"Filtering finished. Got {len(filtered_functions)} filtered functions.")
        logger.info(f"Saving filtered functions to disk: {filtered_functions_path}")
        filtered_functions.save_to_disk(filtered_functions_path)
        logger.info("Saving filtered functions to JSONL...")
        # Additionally save as JSONL for easier inspection and for the next pipeline stage
        filtered_jsonl_path = os.path.join(output_dir, "filtered_functions.jsonl")
        with open(filtered_jsonl_path, 'w') as f:
            # Assuming filtered_functions is a datasets.Dataset object
            for i, item in enumerate(filtered_functions):
                # Adapt this based on the actual structure of filtered_functions dataset
                # We need the actual code content under a key like 'content' or 'seed'
                seed_content = item.get("content") or item.get("seed") 
                if seed_content:
                    json.dump({"id": i, "seed": seed_content}, f)
                    f.write('\n')
                else:
                    logger.warning(f"Item {i} in filtered_functions missing 'content' or 'seed' key.")
        logger.info("JSONL saving complete.")
    except Exception as e:
        logger.error(f"Error during Step 2 (Filtering/Saving): {e}", exc_info=True)
        raise
    
    # Print statistics
    logger.info(f"Raw functions: {len(raw_functions)}")
    logger.info(f"Filtered functions: {len(filtered_functions)}")
    # Calculate filter rate only if raw functions were found
    if len(raw_functions) > 0:
        logger.info(f"Filter rate: {len(filtered_functions) / len(raw_functions) * 100:.2f}%")
        filter_rate = len(filtered_functions) / len(raw_functions) * 100
    else:
        logger.info("Filter rate: N/A (no raw functions found)")
        filter_rate = 0.0
    
    # Save statistics
    stats = {
        "raw_functions": len(raw_functions),
        "filtered_functions": len(filtered_functions),
        "filter_rate": filter_rate
    }
    
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Seed gathering complete. Results saved to {output_dir}")
    logger.info(f"Use {filtered_jsonl_path} for the next pipeline stage")

def main():
    """Main function to run the seed gathering pipeline"""
    parser = argparse.ArgumentParser(description="Run the seed gathering pipeline")
    parser.add_argument("--output_dir", type=str, default="seed_output",
                        help="Directory to save output files")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache the dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Number of workers for parallel processing")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of chunks for parallel processing")
    
    args = parser.parse_args()
    
    run_seed_gathering_pipeline(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size
    )

if __name__ == "__main__":
    main() 