"""
Stack v2 Connector - Enhanced seed gathering from The Stack v2 dataset
"""

import os
import signal
import logging
import sys # Added for flushing
from typing import List, Dict, Any, Set, Optional
from multiprocessing import Pool
import boto3
import botocore # Import botocore
import smart_open
import requests # Import requests
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
import argparse
from .tree_sitter_parser import LANGUAGE, make_parser, node_to_string
import gzip # Add gzip import

# === Module-Level Constants & Functions for Multiprocessing ===

# Configure logging (ensure basicConfig is called somewhere, e.g., main script or here)
logging.basicConfig(
    level=logging.DEBUG, # Keep DEBUG for now
    format='%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Tree-sitter Queries (Module Level) ---
FUNC_DEF_QUERY = LANGUAGE.query("(function_definition) @function.def")
STRING_QUERY = LANGUAGE.query("(string) @string")

# --- Global Boto3 Session Cache (for workers) ---
# Avoid creating sessions repeatedly in each call within a worker
_worker_boto_session = None

def _get_worker_boto_session():
    """Initializes or returns a boto3 session for the current worker process."""
    global _worker_boto_session
    if _worker_boto_session is None:
        try:
            logger.debug(f"Worker {os.getpid()} creating Boto3 session.")
            _worker_boto_session = boto3.Session(
                aws_access_key_id='aws_access_key',
                aws_secret_access_key='aws_secret_access_key',
                region_name='us-east-1'
            )
            # Perform a quick credential check if possible (optional)
            sts = _worker_boto_session.client('sts')
            sts.get_caller_identity()
            logger.debug(f"Worker {os.getpid()} Boto3 session created and credentials validated.")
        except Exception as e:
            logger.error(f"Worker {os.getpid()} failed to create Boto3 session or validate credentials: {e}", exc_info=True)
            # Optionally, re-raise or handle differently if session is critical
    return _worker_boto_session

def _download_content_s3(blob_id, src_encoding):
    """(Top Level) Download content from SWH S3 bucket."""
    session = _get_worker_boto_session()
    if not session:
        logger.error(f"Worker {os.getpid()} cannot download {blob_id}: Boto3 session not available.")
        return None

    s3_url = f"s3://softwareheritage/content/{blob_id}"
    logger.debug(f"Worker {os.getpid()} attempting download from S3: {s3_url}")
    try:
        # Use the session's client for smart_open, configure for anonymous access
        from botocore.client import Config
        from botocore import UNSIGNED # Explicitly import UNSIGNED
        
        # Create a config object specifying unsigned requests
        s3_config = Config(signature_version=UNSIGNED)
        s3_client = session.client("s3", config=s3_config)
        
        # Try without assuming .gz compression first
        with smart_open.open(s3_url, "rb", transport_params={"client": s3_client}) as fin:
             content_bytes = fin.read()
        
        # --- Attempt Gzip Decompression --- 
        # Default to original bytes
        processed_bytes = content_bytes 
        try:
            # Check for gzip magic number
            if content_bytes.startswith(b'\x1f\x8b'): # Corrected magic number check
                logger.debug(f"  Detected possible gzip content for {blob_id}. Attempting decompression.")
                processed_bytes = gzip.decompress(content_bytes)
                logger.debug(f"  Decompressed {blob_id} successfully.")
            # else: processed_bytes remains content_bytes
        except gzip.BadGzipFile:
             logger.warning(f"  File {blob_id} started with gzip magic number but failed decompression. Treating as raw bytes.")
             processed_bytes = content_bytes # Revert to original bytes on error
        except Exception as gzip_err:
             logger.error(f"  Error during potential gzip decompression for {blob_id}: {gzip_err}. Treating as raw bytes.", exc_info=True)
             processed_bytes = content_bytes # Revert to original bytes on error
        # --- End Decompression Attempt ---

        # Decode using provided encoding, falling back to utf-8 with error handling
        if not src_encoding:
            src_encoding = 'utf-8' # Default if not provided
        
        try:
            # Decode the (potentially decompressed) bytes
            content_string = processed_bytes.decode(src_encoding)
            logger.debug(f"  Successfully decoded {blob_id} using {src_encoding}.")
            return content_string
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode {blob_id} (after potential decompression) with encoding '{src_encoding}'. Trying utf-8 with errors='replace'.")
            # Fallback decode using replace
            return processed_bytes.decode('utf-8', errors='replace')
        except LookupError: # Handle invalid src_encoding values
             logger.warning(f"Invalid src_encoding '{src_encoding}' provided for {blob_id}. Trying utf-8 with errors='replace'.")
             return processed_bytes.decode('utf-8', errors='replace')

    except botocore.exceptions.ClientError as e:
        # Handle S3 specific errors (e.g., NoCredentialsError, NoSuchKey, AccessDenied)
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == 'NoCredentialsError':
             logger.error(f"S3 Download failed for {blob_id}: AWS Credentials not found. Ensure they are configured (env vars, config file, or instance profile).", exc_info=False)
        elif error_code == 'NoSuchKey':
             logger.warning(f"S3 Download failed for {blob_id}: Object not found (NoSuchKey).")
        elif error_code == 'AccessDenied':
             logger.error(f"S3 Download failed for {blob_id}: Access Denied. Check credentials and permissions for s3://softwareheritage bucket.", exc_info=False)
        else:
             logger.error(f"S3 ClientError downloading {blob_id}: {e}", exc_info=True) # Log full trace for other client errors
        return None
    except Exception as e:
        # Catch other potential errors during download or smart_open
        logger.error(f"Generic error downloading {blob_id} from S3: {e}", exc_info=True)
        return None

def _get_fns_with_docstrings(src, tree):
    """(Top Level) Extract functions containing a docstring.
       Moved outside the class for pickling.
    """
    if tree is None or tree.root_node is None:
        logger.warning("_get_fns_with_docstrings called with None tree.")
        return []
        
    func_captures = FUNC_DEF_QUERY.captures(tree.root_node)
    string_nodes = {node.id: node for node, name in STRING_QUERY.captures(tree.root_node) if name == 'string'}
    
    logger.debug(f"    FUNC_DEF_QUERY found {len(func_captures)} potential function captures.")
    logger.debug(f"    STRING_QUERY found {len(string_nodes)} string nodes.")

    valid_fns = []
    processed_func_ids = set()
    funcs_processed_count = 0

    for node, name in func_captures:
        logger.debug(f"    FUNC_DEF_QUERY Capture: name='{name}', node_id={node.id}, type={node.type}")
        if name == 'function.def' and node.id not in processed_func_ids:
            funcs_processed_count += 1
            func_name_node = node.child_by_field_name('name')
            func_name_str = node_to_string(src, func_name_node) if func_name_node else "<unknown>"
            logger.debug(f"  Processing function candidate: '{func_name_str}' (Node ID: {node.id})")
            processed_func_ids.add(node.id)

            body_node = node.child_by_field_name('body')

            if not body_node:
                logger.debug(f"    -> Skipping '{func_name_str}': No body node found.")
                continue

            found_docstring_in_body = False
            if body_node.child_count > 0:
                for statement_node in body_node.children:
                    if statement_node.type == 'expression_statement' and statement_node.child_count > 0:
                        string_node_candidate = statement_node.children[0]
                        if string_node_candidate.type == 'string' and string_node_candidate.id in string_nodes:
                            string_content = node_to_string(src, string_node_candidate)
                            logger.debug(f"      Found potential docstring node {string_node_candidate.id}. Content starts with: {string_content[:3]}...")
                            if string_content.startswith('"""') or string_content.startswith("'''"):
                                logger.debug(f"        -> MATCH FOUND for '{func_name_str}'! Adding.")
                                valid_fns.append(node_to_string(src, node))
                                found_docstring_in_body = True
                                break
                            else:
                                logger.debug(f"        -> String node {string_node_candidate.id} content does not start with triple quotes.")
            
            if not found_docstring_in_body:
                 logger.debug(f"    -> No valid docstring found directly in body for '{func_name_str}'.")
    
    logger.info(f"    Finished _get_fns_with_docstrings for this file. Processed {funcs_processed_count} func nodes. Found {len(valid_fns)} matching functions.")
    return valid_fns

def _parse_example(parser, example_data):
    """(Top Level) Parse single example. Needs parser, example dict.
       Gets content via S3 download now.
    """
    file_path = example_data.get("path", "<unknown_path>")
    blob_id = example_data.get("blob_id") # Use blob_id for download
    src_encoding = example_data.get("src_encoding", "utf-8")

    logger.info(f"-> Starting _parse_example for: {file_path} (blob: {blob_id})")
    sys.stdout.flush(); sys.stderr.flush()

    if not blob_id:
        logger.warning(f"Skipping row with missing 'blob_id'. Path: {file_path}")
        return set()

    # --- Download content using S3 --- 
    content = _download_content_s3(blob_id, src_encoding)
    if content is None:
        logger.warning(f"Download failed for blob {blob_id}. Skipping path {file_path}.")
        return set()
    # --- End Download ---

    tree = None
    try:
        # Attempt to encode to bytes assuming it's already a decoded string
        try:
             buf = bytes(content, "utf8") # Content is now guaranteed string from download func
        except Exception as encode_err:
             logger.error(f"Failed to encode downloaded content to bytes for {file_path}: {encode_err}. Skipping.")
             return set()

        logger.debug(f"  Parsing buffer (type: {type(buf)}, len: {len(buf)}). First 100 bytes: {buf[:100]}")
        tree = parser.parse(buf)
        logger.debug(f"  Parser returned tree of type: {type(tree)}")
        
        if tree is None or tree.root_node is None:
            logger.warning(f"Parsing failed for {file_path}. Skipping.")
            return set()

        logger.debug(f"  Tree root node type: {tree.root_node.type}")

        # --- Added logging for query captures --- 
        func_captures = FUNC_DEF_QUERY.captures(tree.root_node)
        logger.debug(f"  FUNC_DEF_QUERY captured {len(func_captures)} nodes before calling extractor.")
        # --- End added logging ---

        logger.info(f"  Successfully parsed {file_path}. Root: {tree.root_node.type}. Calling extractor...")
        sys.stdout.flush(); sys.stderr.flush()

        # Call the global extractor function
        extracted_fns = _get_fns_with_docstrings(buf, tree)
        logger.info(f"  Extraction complete for {file_path}. Found {len(extracted_fns)}.")
        sys.stdout.flush(); sys.stderr.flush()
        return set(extracted_fns) # Return a set

    except Exception as e:
        logger.error(f"Error in _parse_example for {file_path} after download: {e}", exc_info=True)
        logger.error(f"  State: content obtained, tree={tree is not None}")
        sys.stdout.flush(); sys.stderr.flush()
        return set()

def _process_chunk_wrapper(args):
    """(Top Level) Worker function for multiprocessing pool.
       Now initializes boto3 session and parser.
    """
    # Args now only contain index and chunk data
    worker_idx, chunk = args 
    # Get logger for this module within the worker
    worker_logger = logging.getLogger(__name__) # Get logger by name
    worker_logger.info(f"WORKER {os.getpid()} starting processing chunk index {worker_idx}.")
    sys.stdout.flush(); sys.stderr.flush()
    
    # Initialize Boto3 Session and Parser *inside* the worker process
    parser = None
    # Ensure boto session is initialized via the helper
    if _get_worker_boto_session() is None:
        logger.error(f"Worker {os.getpid()} failed to initialize Boto3 session. Cannot proceed.")
        return set()
        
    try:
        parser = make_parser()
        logger.debug(f"Worker {os.getpid()} created its parser.")
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to create parser: {e}", exc_info=True)
        return set() # Cannot proceed without parser

    chunk_fns = set()
    for ex_data in chunk:
        # Pass only parser and data (session is handled globally within worker)
        chunk_fns.update(_parse_example(parser, ex_data))
        
    logger.info(f"WORKER {os.getpid()} finished chunk index {worker_idx}. Found {len(chunk_fns)} funcs in chunk.")
    sys.stdout.flush(); sys.stderr.flush()
    return chunk_fns

# === End Module-Level Functions ===

class StackV2Connector:
    """Class to handle interactions with The Stack v2 dataset"""
    
    def __init__(self, num_workers: int = os.cpu_count() or 4):
        """Initialize the connector."""
        self.num_workers = num_workers

    def load_dataset(self, cache_dir: Optional[str] = None, 
                    streaming: bool = True, # Default back to True
                    max_samples: Optional[int] = None) -> Dataset:
        """
        Load The Stack v2 dataset, preferably streaming.
        Using the-stack-v2-dedup as it's smaller metadata-wise.
        Content will be downloaded via S3.
        """
        logger.info("Loading The Stack v2 dataset (dedup) for Python (Streaming Mode)")
        
        hf_token = os.getenv("HF_TOKEN") # Still needed for dataset access potentially
        if not hf_token:
            logger.warning("HF_TOKEN environment variable not set. May fail for gated datasets.")

        try:
            ds = load_dataset(
                "bigcode/the-stack-v2-dedup", # Using dedup version again
                "Python",
                cache_dir=cache_dir,
                streaming=True, # Set back to True
                split="train",
                token=hf_token # Pass token just in case needed for access control
            )
        except Exception as e:
             logger.error(f"Failed to load dataset stream. Ensure network access and dataset permissions.")
             logger.error(f"Error details: {e}", exc_info=True)
             raise
            
        logger.info(f"Dataset stream ready.")
        # Debug print features once
        try:
             peek = next(iter(ds))
             logger.info(f"[DEBUG] Dataset features: {list(peek.keys())}")
        except Exception:
             logger.warning("[DEBUG] Could not peek at dataset features.")
        return ds
    
    def process_parallel(self, dataset: Dataset, output_file: str, 
                         chunk_size: int = 1000, 
                         max_samples: Optional[int] = None) -> None:
        """Process the dataset stream in parallel using S3 download."""
        
        logger.info("Creating multiprocessing Pool...")
        actual_workers = min(self.num_workers, os.cpu_count() or 1) 
        if actual_workers != self.num_workers:
             logger.warning(f"Adjusted number of workers from {self.num_workers} to {actual_workers} based on cpu count.")
        
        # Use try-with-resources for the Pool
        try:
            with Pool(actual_workers) as p:
                logger.info(f"Pool created with {actual_workers} workers.")

                logger.info(f"Processing dataset stream up to {max_samples if max_samples else 'all'} samples...")
                
                # --- Streaming Logic --- 
                imap_args = []
                current_chunk = []
                worker_index = 0
                processed_count = 0
                dataset_iterator = iter(dataset)
                
                # Use tqdm for progress tracking
                with tqdm(total=max_samples, desc="Processing Stream Samples") as pbar:
                    while True:
                        if max_samples is not None and processed_count >= max_samples:
                            logger.info(f"Reached max_samples limit ({max_samples}). Stopping stream iteration.")
                            break
                        
                        try:
                            example = next(dataset_iterator)
                            processed_count += 1
                            pbar.update(1)
                            current_chunk.append(example)

                            if len(current_chunk) == chunk_size:
                                imap_args.append((worker_index, current_chunk))
                                worker_index += 1
                                current_chunk = []
                        
                        except StopIteration:
                            logger.info("Reached end of dataset stream.")
                            break
                        except Exception as e:
                            logger.error(f"Error reading from dataset stream at sample {processed_count}: {e}", exc_info=True)
                            break
                    
                if current_chunk:
                    imap_args.append((worker_index, current_chunk))
                # --- End Streaming Logic ---

                if not imap_args:
                    logger.warning("No examples collected from dataset stream. Cannot proceed.")
                    return

                total_to_process = processed_count
                logger.info(f"Collected {total_to_process} examples. Prepared {len(imap_args)} chunks for processing.")

                # --- Processing Logic --- 
                funs = set()
                results_iterator = p.imap_unordered(_process_chunk_wrapper, imap_args)
                
                processed_chunks = 0
                with tqdm(total=len(imap_args), desc="Processing Chunks") as pbar_chunks:
                    for chunk_result in results_iterator:
                        processed_chunks += 1
                        pbar_chunks.update(1)
                        len_before = len(funs)
                        funs.update(chunk_result)
                        # Reduce logging frequency
                        # logger.info(f"Collected results from chunk {processed_chunks}/{len(imap_args)}. Total funcs now: {len(funs)}. Gained: {len(funs) - len_before}")
                        # if processed_chunks % 10 == 0: 
                        #      sys.stdout.flush(); sys.stderr.flush()
                
                sys.stdout.flush(); sys.stderr.flush()
                logger.info(f"Finished processing {total_to_process} samples. Extracted {len(funs)} total functions.")
                
                # --- Saving Logic --- 
                if funs:
                    logger.info(f"Saving {len(funs)} extracted functions...")
                    # Save as HF Dataset
                    temp_output_dir = output_file + "_temp_save"
                    try:
                        new_ds_dict = {"content": list(funs)} # Just save content
                        new_ds = Dataset.from_dict(new_ds_dict)
                        new_ds.save_to_disk(temp_output_dir)
                        # If successful, rename
                        if os.path.exists(output_file):
                            import shutil
                            logger.warning(f"Removing existing output directory: {output_file}")
                            shutil.rmtree(output_file)
                        os.rename(temp_output_dir, output_file)
                        logger.info(f"Saved extracted functions dataset to {output_file}")
                    except Exception as save_err:
                        logger.error(f"Failed to save dataset to disk: {save_err}", exc_info=True)
                        # Clean up temp dir if it exists
                        if os.path.exists(temp_output_dir):
                            import shutil
                            try:
                                shutil.rmtree(temp_output_dir)
                            except Exception as clean_err:
                                logger.error(f"Failed to clean up temp dir {temp_output_dir}: {clean_err}")
                    finally:
                        # Ensure temp dir doesn't linger if rename failed but dir exists
                        if os.path.exists(temp_output_dir):
                            try:
                                import shutil
                                logger.error(f"Cleaning up leftover temp dir: {temp_output_dir}")
                                shutil.rmtree(temp_output_dir)
                            except Exception: pass # Ignore cleanup errors
                
                else:
                    logger.warning("No functions were extracted. Saving empty dataset structure.")
                    # Ensure directory exists if needed
                    os.makedirs(output_file, exist_ok=True) 
                    # Save empty dataset structure
                    Dataset.from_dict({"content": []}).save_to_disk(output_file) 
                    logger.info(f"Saved empty dataset structure to {output_file}")
        
        except Exception as pool_exc:
            logger.error(f"Error occurred during parallel processing: {pool_exc}", exc_info=True)
            # Pool context manager should handle cleanup, but re-raising might be needed
            raise

def main():
    """Main function to run the seed gathering process"""
    parser = argparse.ArgumentParser(description="Extract Python functions with docstrings from The Stack v2")
    parser.add_argument("--output_file", type=str, default="python_functions_with_docstrings.jsonl",
                        help="Path to save the extracted functions")
    parser.add_argument("--cache_dir", type=str, default=None, 
                        help="Directory to cache the dataset")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Number of workers for parallel processing")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of chunks for parallel processing")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    connector = StackV2Connector(num_workers=args.num_workers)
    dataset = connector.load_dataset(cache_dir=args.cache_dir)
    connector.process_parallel(
        dataset=dataset,
        output_file=args.output_file,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main() 
