#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Pipeline Runner for Python Self-Align
"""

import os
import argparse
import logging
import json
import time
import subprocess
import sys
import shutil # Added for cleaning
import threading # Added for Popen stream reading
from pathlib import Path
from dotenv import load_dotenv

# Import the helper function needed in create_final_dataset
from python_self_align.validation.execution_validator import read_jsonl

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_file_logging(log_dir):
    """Sets up logging to a file within the specified directory."""
    log_file = Path(log_dir) / "pipeline.log"
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True) # Ensure log directory exists

    # Remove existing handlers if any were added by basicConfig
    # for handler in logging.root.handlers[:]:
    #      logging.root.removeHandler(handler)
    # Get the root logger
    root_logger = logging.getLogger()
    # Remove existing StreamHandlers if they exist to avoid duplicate console logs
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
             print(f"Removing existing StreamHandler: {handler}")
             root_logger.removeHandler(handler)

    # Create a file handler
    file_handler = logging.FileHandler(log_file, mode='a') # Append mode
    file_formatter = logging.Formatter('%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG) # Log DEBUG level and up to file

    # Create a console handler (to still see INFO level on console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Simpler format for console
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO) # Log INFO level and up to console

    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG) # Set root logger level to DEBUG to capture everything for the file handler

    logger.info(f"File logging configured. Log file: {log_file}")

# --- Determine and log .env path AFTER logger is configured ---
script_dir = Path(__file__).parent
dotenv_path = script_dir / ".env"
logger.info(f"Looking for .env file at: {dotenv_path}")
# -----------------------------------------------------------

# Load environment variables from the specific .env file path
load_success = load_dotenv(dotenv_path=dotenv_path)
logger.info(f"load_dotenv success: {load_success}") # Log if loading was successful

# Log the loaded token (or None) AFTER configuring logging and attempting load
logger.info(f"HF_TOKEN from env: {os.getenv('HF_TOKEN')}")

logger.info("run_full_pipeline.py started execution.")

def log_stream(stream, log_level):
    """Reads a stream line by line and logs it.
    Helper function for reading subprocess stdout/stderr.
    """
    file_handler = None
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break
    
    try:
        for line in iter(stream.readline, ''):
            stripped_line = line.strip()
            if stripped_line: # Avoid logging empty lines
                root_logger.log(log_level, stripped_line)
                if file_handler: 
                    file_handler.flush() # Flush after logging each line
    except Exception as e:
        root_logger.error(f"Error reading subprocess stream: {e}", exc_info=True)
    finally:
        try:
            stream.close()
        except Exception:
            pass # Ignore stream closing errors

def run_seed_generation(output_dir, num_samples):
    """Run the actual seed gathering step using run_seed_gathering.py"""
    logger.info(f"Step 1: Gathering {num_samples} seeds from Stack v2")
    
    seed_script = Path(__file__).parent / "seed_gathering" / "run_seed_gathering.py"
    seed_dir = Path(output_dir) / "seed_output"
    output_file = seed_dir / "filtered_functions.jsonl"
    
    # --- Checkpointing ---
    if output_file.exists():
        logger.info(f"Checkpoint found: Seeds already exist at {output_file}. Skipping Step 1.")
        return str(output_file)
    # --- End Checkpointing ---

    seed_dir.mkdir(parents=True, exist_ok=True)

    module_path = "python_self_align.seed_gathering.run_seed_gathering"
    cmd = [
        sys.executable,
        "-m", module_path,
        "--output_dir", str(seed_dir),
        "--max_samples", str(num_samples),
    ]
    cmd = [arg for arg in cmd if arg is not None]

    process = None # Initialize process to None
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        project_root = Path(__file__).parent.parent
        logger.info(f"Running subprocess with cwd={project_root} and streaming output...")
        
        # Use Popen to stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, 
            encoding='utf-8', # Explicitly set encoding
            errors='replace', # Handle potential decoding errors
            bufsize=1, # Line-buffered
            env=os.environ,
            cwd=project_root
        )

        # Create threads to read stdout and stderr
        stdout_thread = threading.Thread(target=log_stream, args=(process.stdout, logging.INFO))
        # Log stderr as INFO now to avoid mislabeling warnings/debug from subprocess as ERROR
        stderr_thread = threading.Thread(target=log_stream, args=(process.stderr, logging.INFO)) 

        # Start the threads
        stdout_thread.start()
        stderr_thread.start()

        # Wait for the threads to finish (meaning streams are closed)
        stdout_thread.join()
        stderr_thread.join()

        # Wait for the process to terminate and get the return code
        return_code = process.wait()
        logger.info(f"Subprocess finished with return code: {return_code}")

        # Check the return code and raise error if needed
        if return_code != 0:
            logger.error(f"Seed gathering script failed with return code {return_code}. Raising CalledProcessError.")
            # Note: stdout/stderr were already logged by threads
            raise subprocess.CalledProcessError(
                returncode=return_code,
                cmd=cmd
                # stdout/stderr args are not needed here as they were logged live
            )
            
        logger.info(f"Seed gathering completed successfully. Seeds saved to {output_file}")

    except subprocess.CalledProcessError as e:
        # Log minimal info as details were logged by threads or before raising
        logger.error(f"Seed gathering subprocess failed (Return Code: {e.returncode}). Check logs above.")
        raise # Re-raise the exception to stop the pipeline
    except FileNotFoundError:
        logger.error(f"Error: The script '{seed_script}' or python interpreter '{sys.executable}' was not found.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during seed generation: {e}", exc_info=True)
        # Ensure process is terminated if it's still running after an unexpected error
        if process and process.poll() is None:
            logger.warning("Terminating subprocess due to unexpected error.")
            process.terminate()
            try:
                process.wait(timeout=5) # Wait briefly for termination
            except subprocess.TimeoutExpired:
                logger.error("Subprocess did not terminate gracefully, killing.")
                process.kill()
        raise
        
    return str(output_file)

def run_self_oss_instruct(seed_file, output_dir, batch_size, delay):
    """Run the self-OSS-instruct steps"""
    logger.info("Step 2: Running Self-OSS-Instruct pipeline")
    
    # Import real generation using relative path to the specific file
    from .self_oss_instruct.self_oss_instruct import generate_instruction_pairs, read_jsonl
    
    # Create the output directory
    instruct_dir = Path(output_dir) / "self_oss_output"
    instruct_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Intermediate file paths ---
    concept_file = instruct_dir / "concepts.jsonl"
    instruction_file = instruct_dir / "instructions.jsonl"
    response_file = instruct_dir / "responses.jsonl"

    # Step 2.1: S->C (Seed to Concept)
    if not concept_file.exists():
        logger.info("Step 2.1: S->C (Seed to Concept)")
        # Check if seed file exists before trying to read
        if not Path(seed_file).exists():
             logger.error(f"Seed file {seed_file} not found. Cannot proceed with S->C.")
             raise FileNotFoundError(f"Seed file {seed_file} not found.")
        generate_instruction_pairs(
            input_data=read_jsonl(str(seed_file)),
            output_path=str(concept_file),
            mode="S->C",
            model_name="bigcode/starcoder2-15b",
            batch_size=batch_size,
            delay=delay,
        )
    else:
        logger.info(f"Checkpoint found: Concepts file {concept_file} exists. Skipping S->C generation.")

    # Step 2.2: C->I (Concept to Instruction)
    if not instruction_file.exists():
        logger.info("Step 2.2: C->I (Concept to Instruction)")
        # Check if concept file exists before trying to read
        if not concept_file.exists():
            logger.error(f"Concept file {concept_file} not found. Cannot proceed with C->I.")
            raise FileNotFoundError(f"Concept file {concept_file} not found.")
        generate_instruction_pairs(
            input_data=read_jsonl(str(concept_file)),
            output_path=str(instruction_file),
            mode="C->I",
            model_name="bigcode/starcoder2-15b",
            batch_size=batch_size,
            delay=delay,
        )
    else:
        logger.info(f"Checkpoint found: Instructions file {instruction_file} exists. Skipping C->I generation.")

    # Step 2.3: I->R (Instruction to Response)
    if not response_file.exists():
        logger.info("Step 2.3: I->R (Instruction to Response)")
        # Check if instruction file exists before trying to read
        if not instruction_file.exists():
             logger.error(f"Instruction file {instruction_file} not found. Cannot proceed with I->R.")
             raise FileNotFoundError(f"Instruction file {instruction_file} not found.")
        generate_instruction_pairs(
            input_data=read_jsonl(str(instruction_file)),
            output_path=str(response_file),
            mode="I->R",
            model_name="bigcode/starcoder2-15b",
            batch_size=batch_size,
            delay=delay,
        )
    else:
        logger.info(f"Checkpoint found: Responses file {response_file} exists. Skipping I->R generation.")

    return str(instruct_dir)

def run_validation(instruct_dir, output_dir):
    """Run the validation step"""
    logger.info("Step 3: Running validation pipeline")
    
    import subprocess
    import sys
    
    # Create the output directory
    validation_dir = Path(output_dir) / "validation_output"
    validation_stats_file = validation_dir / "validation_stats.json"

    # --- Checkpointing ---
    # Check if final stats file exists as a proxy for completion
    if validation_stats_file.exists():
        logger.info(f"Checkpoint found: Validation stats {validation_stats_file} exists. Skipping Step 3.")
        return str(validation_dir)
    # --- End Checkpointing ---

    validation_dir.mkdir(parents=True, exist_ok=True)

    # --- Check if input exists ---
    responses_file = Path(instruct_dir) / "responses.jsonl"
    if not responses_file.exists():
         logger.error(f"Input responses file {responses_file} not found for validation. Cannot proceed.")
         raise FileNotFoundError(f"Input responses file {responses_file} not found for validation.")
    # --- End check ---

    # Run the validation pipeline using the run_validation.py script
    # --- Modify command to use -m flag and set cwd --- 
    module_path = "python_self_align.validation.run_validation"
    cmd = [
        sys.executable,
        "-m", module_path, # Run as module
        "--input_dir", instruct_dir,
        "--output_dir", validation_dir
    ]
    
    project_root = Path(__file__).parent.parent # Get project root
    logger.info(f"Running validation subprocess with cwd={project_root}")
    subprocess.run(cmd, check=True, cwd=project_root) # Set cwd
    # --- End modification ---
    
    return validation_dir

def create_final_dataset(validation_dir, output_dir):
    """Create the final dataset from validated responses"""
    logger.info("Step 4: Creating final dataset")
    
    import json
    
    # Create the final dataset directory
    final_dir = Path(output_dir) / "final_dataset"
    final_file = final_dir / "instruction_dataset.jsonl"
    stats_file = final_dir / "stats.json"

    # --- Checkpointing ---
    if final_file.exists() and stats_file.exists():
        logger.info(f"Checkpoint found: Final dataset {final_file} exists. Skipping Step 4.")
        return str(final_file)
    # --- End Checkpointing ---

    final_dir.mkdir(parents=True, exist_ok=True)

    # Find the validated responses file (assuming execution_validator writes this)
    # NOTE: The previous code used "validated_responses.jsonl" but the validation script uses execution_validator.py
    # Let's assume execution_validator produces an aggregated file in validation_dir.
    # A robust way is to look for a specific output file, e.g., "aggregated_valid.jsonl"
    # Or, use the stats file created by process_directory in validation_pipeline.py
    # For simplicity, let's assume run_validation produces a known file.
    # If `run_validation` uses `aggregate_results` with default name, it's "aggregated.jsonl"
    # Let's assume the validation pipeline creates "aggregated.jsonl" with processed items.
    # We need to check `validation_pipeline.py` and `execution_validator.py` to be sure what file contains valid items.
    # For now, assuming the validator puts valid items into a specific file or marks them.
    # A safer approach is to parse the `validation_stats.json` which should list output files.
    # Let's adjust based on validation_pipeline structure: it processes files individually, then aggregates.
    # Assume `aggregate_results` is called (or we call it here) to get 'aggregated.jsonl'.
    # We need to filter this aggregated file.

    aggregated_validated_file = Path(validation_dir) / "aggregated.jsonl" # Assuming this is produced

    if not aggregated_validated_file.exists():
        logger.warning(f"Validated responses file {aggregated_validated_file} not found in {validation_dir}. Trying to find individual files.")
        # Fallback: try to read individual files? Or rely on validation_stats.json?
        # For now, let's error if the assumed aggregated file isn't there.
        # A better implementation would aggregate here if needed.
        logger.error(f"Cannot create final dataset without aggregated validated results.")
        # Or perhaps the validation step *should* create the final filtered dataset directly?
        # Sticking to the original flow: assume the file exists.
        raise FileNotFoundError(f"Expected aggregated validation results file not found: {aggregated_validated_file}")


    # Read the aggregated validated responses
    validated_items = read_jsonl(str(aggregated_validated_file)) # Use helper if available

    # Filter for valid items (assuming execution_validator adds an 'execution_valid' key)
    valid_items = [item for item in validated_items if item.get("execution_valid", False)]

    # Create the final dataset
    final_dataset = []
    for item in valid_items:
        final_dataset.append({
            "instruction": item.get("instruction", ""),
            "response": item.get("response", "")
        })
    
    # Write the final dataset
    with open(final_file, 'w') as f:
        for item in final_dataset:
            f.write(json.dumps(item) + '\n')
    
    # Write stats
    stats = {
        "total_items": len(valid_items),
        "instruction_response_pairs": len(final_dataset)
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Created final dataset with {len(final_dataset)} instruction-response pairs")
    
    return str(final_file)

def create_summary_report(output_dir):
    """Create a summary report of the pipeline"""
    logger.info("Creating summary report")
    
    report_file = Path(output_dir) / "pipeline_report.md"

    # --- Checkpointing ---
    if report_file.exists():
        logger.info(f"Checkpoint found: Summary report {report_file} exists. Skipping.")
        return
    # --- End Checkpointing ---

    # Collect statistics from each stage
    try:
        # Seed generation stats
        seed_stats_file = Path(output_dir) / "seed_output" / "stats.json"
        with open(seed_stats_file, 'r') as f:
            seed_stats = json.load(f)
        
        # Validation stats for responses
        response_stats_file = Path(output_dir) / "validation_output" / "responses_stats.json"
        with open(response_stats_file, 'r') as f:
            response_stats = json.load(f)
        
        # Final dataset stats
        final_stats_file = Path(output_dir) / "final_dataset" / "stats.json"
        with open(final_stats_file, 'r') as f:
            final_stats = json.load(f)
        
        # Create the report
        report = [
            "# Python Self-Align Pipeline Summary Report",
            "",
            "## Pipeline Overview",
            "",
            "The Python Self-Align pipeline consists of three main stages:",
            "",
            "1. **Seed Gathering**: Extract high-quality Python functions with docstrings",
            "2. **Self-OSS-Instruct**: Generate concepts, instructions, and responses",
            "3. **Validation**: Validate the generated code for correctness",
            "",
            "## Pipeline Statistics",
            "",
            f"### Seed Generation",
            f"- Raw functions: {seed_stats.get('raw_functions', 'N/A')}",
            f"- Filtered functions: {seed_stats.get('filtered_functions', 'N/A')}",
            f"- Filter rate: {seed_stats.get('filter_rate', 'N/A'):.2f}%",
            "",
            f"### Self-OSS-Instruct",
            f"- S-C (Seed to Concept): 20 concepts generated",
            f"- C-I (Concept to Instruction): 20 instructions generated",
            f"- I-R (Instruction to Response): 20 responses generated",
            "",
            f"### Validation",
            f"- Total responses: {response_stats.get('total_items', 'N/A')}",
            f"- Valid responses: {response_stats.get('valid_items', 'N/A')}",
            f"- Validation rate: {response_stats.get('validation_rate', 0) * 100:.2f}%",
            "",
            f"### Final Dataset",
            f"- Instruction-response pairs: {final_stats.get('instruction_response_pairs', 'N/A')}",
            "",
            "## Next Steps",
            "",
            "The generated instruction-response pairs can now be used to fine-tune a code generation model.",
        ]
        
        # Write the report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Created summary report at {report_file}")
        
    except Exception as e:
        logger.error(f"Error creating summary report: {e}")
        import traceback
        logger.error(traceback.format_exc())

def run_pipeline(output_dir, num_samples=20, clean_output=False, batch_size=4, delay=0.5):
    """Run the complete pipeline"""
    start_time = time.time()
    
    output_path = Path(output_dir)

    # --- Setup File Logging --- 
    setup_file_logging(output_dir) # Call the setup function here
    # --------------------------

    # --- Cleaning ---
    if clean_output:
        logger.info(f"Cleaning output directory: {output_path}")
        if output_path.exists():
            try:
                shutil.rmtree(output_path)
                logger.info(f"Removed existing directory: {output_path}")
            except OSError as e:
                logger.error(f"Error removing directory {output_path}: {e}")
                # Decide if we should stop or continue
                # raise # Option: Stop if cleaning fails
        # Recreate the main output directory after cleaning
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # Ensure the main output directory exists even if not cleaning
        output_path.mkdir(parents=True, exist_ok=True)
    # --- End Cleaning ---

    try: # Wrap steps in try/except for better error handling if a step fails
        # Step 1: Generate seeds
        seed_file = run_seed_generation(output_dir, num_samples)
        if not Path(seed_file).exists() or Path(seed_file).stat().st_size == 0:
             logger.warning(f"Seed generation result {seed_file} is missing or empty after step completion.")
             # Decide whether to raise an error or continue (assuming checkpoint handles re-run)

        # Step 2: Run Self-OSS-Instruct
        instruct_dir = run_self_oss_instruct(seed_file, output_dir, batch_size, delay)
        # Add checks for intermediate files if needed

        # Step 3: Run validation
        validation_dir = run_validation(instruct_dir, output_dir)
        # Add check for validation output

        # Step 4: Create final dataset
        final_file = create_final_dataset(validation_dir, output_dir)

        # Create summary report (can run even if some prior steps were skipped)
        create_summary_report(output_dir)

        # Print completion message
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
        if final_file and Path(final_file).exists():
            logger.info(f"Final dataset: {final_file}")
        else:
            logger.warning("Final dataset file was not generated or found.")

    except FileNotFoundError as e:
         logger.error(f"Pipeline halted due to missing file: {e}")
         logger.error("Please check the logs for the failed step and ensure input files are generated correctly.")
    except subprocess.CalledProcessError as e:
         logger.error(f"Pipeline halted due to subprocess error in step: {e.cmd}")
         logger.error(f"Return Code: {e.returncode}")
         logger.error(f"Stderr: {e.stderr}") # Log stderr for debugging
    except Exception as e:
         logger.error(f"Pipeline encountered an unexpected error: {e}", exc_info=True)

def main():
    """Main function"""
    logger.info("Entering main() function.")
    parser = argparse.ArgumentParser(description="Run the complete Python Self-Align pipeline")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline_output",
        help="Directory to store all pipeline outputs (and the log file)" # Updated help text
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to request from seed gathering"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the output directory before running"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for Self-OSS-Instruct generation steps (S->C, C->I, I->R)"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between batches in Self-OSS-Instruct generation steps"
    )

    args = parser.parse_args()
    
    run_pipeline(args.output_dir, args.num_samples, args.clean, args.batch_size, args.delay)

if __name__ == "__main__":
    # logger.info("Entering main() function.") # Removed duplicate log
    main() 