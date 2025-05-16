"""
Main script for the self-OSS-instruct stage of the Python self-align pipeline
"""

import os
import argparse
import logging
import json
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any
from datasets import load_dataset

from .self_oss_instruct import generate_instruction_pairs
from .vllm_wrapper import create_model_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the real generation function
from mock_generation import run_generation, run_mock_generation

def run_full_pipeline(
    seed_file: str,
    concept_file: str,
    instruction_file: str,
    response_file: str,
    model: str = "bigcode/starcoder2-15b",
    use_vllm_server: bool = False,
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
    s2c_samples: int = 1,
    c2i_samples: int = 1,
    i2r_samples: int = 1,
    num_examples: int = 8,
    async_micro_batch_size: int = 8,
    seed: int = 42,
    start_indices: Optional[Tuple[int, int, int]] = None,
    max_items: Optional[Tuple[int, int, int]] = None,
    continue_from: Optional[Tuple[str, str, str]] = None,
    delay: Optional[float] = None,
    use_chat_format: bool = False,
    skip_stages: Optional[list] = None,
    use_mock: bool = False,
    batch_size: int = 10
):
    """
    Run the complete Self-OSS-Instruct pipeline.
    
    Args:
        seed_file: Path to the seed data
        concept_file: Path to save the concept data
        instruction_file: Path to save the instruction data
        response_file: Path to save the response data
        model: Model name or path
        use_vllm_server: Whether to use vLLM server
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        s2c_samples: Number of samples per item for S->C
        c2i_samples: Number of samples per item for C->I
        i2r_samples: Number of samples per item for I->R
        num_examples: Number of examples to use
        async_micro_batch_size: Size of async micro-batches
        seed: Random seed
        start_indices: Tuple of start indices for each stage (S->C, C->I, I->R)
        max_items: Tuple of max items for each stage
        continue_from: Tuple of paths to continue from for each stage
        delay: Delay between batches
        use_chat_format: Whether to use chat format
        skip_stages: List of stages to skip
        use_mock: Whether to use mock generation instead of real generation
        batch_size: Batch size for real generation
    """
    # Set default values for optional arguments
    skip_stages = skip_stages or []
    start_indices = start_indices or (0, 0, 0)
    max_items_tuple = max_items or (None, None, None)
    continue_from_tuple = continue_from or (None, None, None)
    
    # Create output directories
    for file_path in [concept_file, instruction_file, response_file]:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Choose the generation function
    generation_func = run_mock_generation if use_mock else run_generation
    
    # Stage 1: S->C (Seed to Concept)
    if "S->C" not in skip_stages:
        logger.info("Running S->C (Seed to Concept) stage")
        if use_mock:
            generation_func(
                input_file=seed_file,
                output_file=concept_file,
                mode="S->C",
                max_items=max_items_tuple[0]
            )
        else:
            generation_func(
                input_file=seed_file,
                output_file=concept_file,
                mode="S->C",
                max_items=max_items_tuple[0],
                batch_size=batch_size,
                delay=delay or 0.5
            )
    else:
        logger.info("Skipping S->C stage")
    
    # Stage 2: C->I (Concept to Instruction)
    if "C->I" not in skip_stages:
        logger.info("Running C->I (Concept to Instruction) stage")
        if use_mock:
            generation_func(
                input_file=concept_file,
                output_file=instruction_file,
                mode="C->I",
                max_items=max_items_tuple[1]
            )
        else:
            generation_func(
                input_file=concept_file,
                output_file=instruction_file,
                mode="C->I",
                max_items=max_items_tuple[1],
                batch_size=batch_size,
                delay=delay or 0.5
            )
    else:
        logger.info("Skipping C->I stage")
    
    # Stage 3: I->R (Instruction to Response)
    if "I->R" not in skip_stages:
        logger.info("Running I->R (Instruction to Response) stage")
        if use_mock:
            generation_func(
                input_file=instruction_file,
                output_file=response_file,
                mode="I->R",
                max_items=max_items_tuple[2]
            )
        else:
            generation_func(
                input_file=instruction_file,
                output_file=response_file,
                mode="I->R",
                max_items=max_items_tuple[2],
                batch_size=batch_size,
                delay=delay or 0.5
            )
    else:
        logger.info("Skipping I->R stage")
    
    logger.info("Self-OSS-Instruct pipeline completed")

def main():
    """Main function to run the Self-OSS-Instruct pipeline"""
    parser = argparse.ArgumentParser(description="Run the Self-OSS-Instruct pipeline")
    parser.add_argument("--seed_file", type=str, required=True,
                        help="Path to the seed data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument("--model", type=str, default="bigcode/starcoder2-15b",
                        help="Model name or path")
    parser.add_argument("--use_vllm_server", action="store_true",
                        help="Use vLLM server through OpenAI API")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_output_tokens", type=int, default=1024,
                        help="Maximum output tokens")
    parser.add_argument("--s2c_samples", type=int, default=1,
                        help="Number of samples per item for S->C")
    parser.add_argument("--c2i_samples", type=int, default=1,
                        help="Number of samples per item for C->I")
    parser.add_argument("--i2r_samples", type=int, default=1,
                        help="Number of samples per item for I->R")
    parser.add_argument("--num_examples", type=int, default=8,
                        help="Number of examples to use")
    parser.add_argument("--async_micro_batch_size", type=int, default=8,
                        help="Size of async micro-batches")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--start_indices", type=str, default=None,
                        help="Comma-separated start indices for each stage")
    parser.add_argument("--max_items", type=str, default=None,
                        help="Comma-separated max items for each stage")
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Comma-separated paths to continue from for each stage")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between batches")
    parser.add_argument("--use_chat_format", action="store_true",
                        help="Use chat format")
    parser.add_argument("--skip_stages", type=str, default=None,
                        help="Comma-separated list of stages to skip")
    parser.add_argument("--use_mock", action="store_true",
                        help="Use mock generation instead of real generation")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for real generation")
    
    args = parser.parse_args()
    
    # Process comma-separated arguments
    start_indices = tuple(map(int, args.start_indices.split(","))) if args.start_indices else None
    max_items = tuple(map(lambda x: int(x) if x != "None" else None, args.max_items.split(","))) if args.max_items else None
    continue_from = tuple(args.continue_from.split(",")) if args.continue_from else None
    skip_stages = args.skip_stages.split(",") if args.skip_stages else None
    
    # Create output files
    concept_file = os.path.join(args.output_dir, "concepts.jsonl")
    instruction_file = os.path.join(args.output_dir, "instructions.jsonl")
    response_file = os.path.join(args.output_dir, "responses.jsonl")
    
    # Run the pipeline
    run_full_pipeline(
        seed_file=args.seed_file,
        concept_file=concept_file,
        instruction_file=instruction_file,
        response_file=response_file,
        model=args.model,
        use_vllm_server=args.use_vllm_server,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        s2c_samples=args.s2c_samples,
        c2i_samples=args.c2i_samples,
        i2r_samples=args.i2r_samples,
        num_examples=args.num_examples,
        async_micro_batch_size=args.async_micro_batch_size,
        seed=args.seed,
        start_indices=start_indices,
        max_items=max_items,
        continue_from=continue_from,
        delay=args.delay,
        use_chat_format=args.use_chat_format,
        skip_stages=skip_stages,
        use_mock=args.use_mock,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main() 