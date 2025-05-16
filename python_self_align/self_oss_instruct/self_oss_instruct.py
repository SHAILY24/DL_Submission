"""
Self-OSS-Instruct: Convert Python function seeds to instruction-response pairs
"""

import os
import logging
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from tqdm.auto import tqdm
import jsonlines

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .prompt_templates import (
    format_s2c_prompt, format_c2i_prompt, format_i2r_prompt,
    PYTHON_S2C_EXAMPLES, PYTHON_C2I_EXAMPLES, PYTHON_I2R_EXAMPLES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Types for instruction modes
InstructMode = Literal["S->C", "C->I", "I->R"]

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read data from a jsonl file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Error decoding JSON on line: {line}")
    return data

def write_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Write data to a jsonl file"""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def extract_completion(full_output: str, prompt: str) -> str:
    """Extract the completion from a full response by removing the prompt"""
    # Basic check if prompt is at the start
    if prompt and full_output.startswith(prompt):
        return full_output[len(prompt):].strip()
    # Fallback or if prompt wasn't included in output (depends on generation args)
    return full_output.strip()

def extract_generated_content(
    response: str, 
    prompt: str, 
    mode: InstructMode
) -> Optional[str]:
    """
    Extract the generated content from a response.
    
    Args:
        response: The model's response
        prompt: The prompt that was used
        mode: The instruction mode
        
    Returns:
        The extracted content or None if extraction failed
    """
    # Extract the completion from the response
    completion = extract_completion(response, prompt)
    
    if not completion:
        return None
    
    # Clean up based on mode
    if mode == "S->C":
        # For S->C, we want a clean paragraph
        return completion.strip()
    elif mode == "C->I":
        # For C->I, we want a clean instruction.
        # The prompt ends with \"Instruction:\".
        # Sometimes the model adds extra text after the instruction.
        # Try to find the instruction and strip trailing unwanted text.
        instruction_part = completion.strip()
        
        # Define potential terminators that indicate the model went off-script
        terminators = ["\\nDescription:", "\\nInstruction:", "\\n```", "\\nBelow is a description:"]
        min_pos = len(instruction_part) # Initialize with the full length
        
        for term in terminators:
            pos = instruction_part.find(term)
            if pos != -1:
                min_pos = min(min_pos, pos)
                
        # Return the part up to the earliest terminator found
        return instruction_part[:min_pos].strip()
    elif mode == "I->R":
        # For I->R, we want to extract Python code
        
        # Try to extract code between triple backticks
        import re
        code_blocks = re.findall(r'```(?:python)?(.*?)```', completion, re.DOTALL)
        
        if code_blocks:
            # Return the first code block
            return code_blocks[0].strip()
        else:
            # If no code blocks, return the whole completion
            return completion.strip()
    
    return None

def build_prompt(
    item: Dict[str, Any],
    mode: InstructMode,
    use_examples: bool = True,
    use_chat_format: bool = False
) -> Any:
    """
    Build a prompt for the given item and mode.
    
    Args:
        item: The item to generate for
        mode: The instruction mode
        use_examples: Whether to use few-shot examples
        use_chat_format: Whether to use chat format
        
    Returns:
        The formatted prompt
    """
    if mode == "S->C":
        examples = PYTHON_S2C_EXAMPLES if use_examples else None
        return format_s2c_prompt(
            seed=item["seed"],
            examples=examples,
            use_chat_format=use_chat_format
        )
    elif mode == "C->I":
        examples = PYTHON_C2I_EXAMPLES if use_examples else None
        return format_c2i_prompt(
            concept=item["concept"],
            examples=examples,
            use_chat_format=use_chat_format
        )
    elif mode == "I->R":
        examples = PYTHON_I2R_EXAMPLES if use_examples else None
        return format_i2r_prompt(
            instruction=item["instruction"],
            examples=examples,
            use_chat_format=use_chat_format
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

def generate_instruction_pairs(
    input_data: List[Dict[str, Any]],
    output_path: str,
    mode: InstructMode,
    model_name: str,
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
    num_samples_per_item: int = 1,
    num_examples: int = 8,
    batch_size: int = 8,
    seed: int = 42,
    start_index: int = 0,
    max_items: Optional[int] = None,
    continue_from: Optional[str] = None,
    delay: Optional[float] = None,
    use_chat_format: bool = False,
    quantization_bits: Optional[Literal[4, 8]] = 8,
):
    """
    Generate instruction pairs using the specified model with transformers.
    
    Args:
        input_data: The input data to generate for
        output_path: Path to save the generated data
        mode: The instruction mode
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_output_tokens: Maximum *new* tokens to generate
        num_samples_per_item: Number of samples to generate per item
        num_examples: Number of examples to use (0 for none)
        batch_size: Batch size for generation
        seed: Random seed
        start_index: Index to start generation from
        max_items: Maximum number of items to generate for
        continue_from: Path to continue from (loads existing data)
        delay: Delay between batches
        use_chat_format: Whether to use chat format
        quantization_bits: Load model in 4-bit or 8-bit precision (requires bitsandbytes)
    """
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Load Model and Tokenizer using Transformers ---
    logger.info(f"Loading tokenizer for {model_name}")
    # Trust remote code for models like StarCoder
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        padding_side='left' 
    )
    
    # Set pad token if not present (common for GPT-like models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer pad_token set to eos_token")

    quantization_config = None
    if quantization_bits:
        logger.info(f"Setting up {quantization_bits}-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=(quantization_bits == 8),
            load_in_4bit=(quantization_bits == 4),
            llm_int8_enable_fp32_cpu_offload=(quantization_bits == 8),
            # Optional: Add 4-bit specific configs if needed
            # bnb_4bit_quant_type="nf4", 
            # bnb_4bit_compute_dtype=torch.bfloat16, # or torch.float16
            # bnb_4bit_use_double_quant=True, 
        )

    logger.info(f"Loading model {model_name} with quantization_config={quantization_config is not None} and device_map='auto'")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto", # Requires accelerate
        trust_remote_code=True,
        torch_dtype=torch.float16 # Explicitly request float16
    )
    logger.info(f"Model loaded successfully on device: {model.device}")
    # --- End Model Loading ---
    
    # Determine use of examples
    use_examples = num_examples > 0
    
    # Load existing data if continuing
    existing_data = []
    if continue_from and os.path.exists(continue_from):
        logger.info(f"Loading existing data from {continue_from}")
        existing_data = read_jsonl(continue_from)
        logger.info(f"Loaded {len(existing_data)} existing items")
    
    # Prepare data for generation
    input_items = input_data[start_index:start_index + max_items] if max_items else input_data[start_index:]
    logger.info(f"Generating for {len(input_items)} items in mode {mode}")
    
    # Track generated items
    generated_items = []
    
    # --- Refactored Generation Loop ---
    # Group prompts and original items for batch processing
    prompts_with_items = []
    for item in input_items:
        for _ in range(num_samples_per_item):
            prompt = build_prompt(item, mode, use_examples, use_chat_format)
            prompts_with_items.append({"prompt": prompt, "original_item": item})

    logger.info(f"Prepared {len(prompts_with_items)} prompts for generation")

    # Process in batches
    all_completions_data = []
    for i in tqdm(range(0, len(prompts_with_items), batch_size), desc=f"Generating {mode}"):
        batch_data = prompts_with_items[i:i + batch_size]
        batch_prompts = [d["prompt"] for d in batch_data]
        
        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_output_tokens, # Note: max_tokens -> max_new_tokens
                temperature=temperature,
                top_p=0.95 if temperature > 0 else 1.0, # Adjust top_p based on temp
                do_sample=temperature > 0, # Only sample if temp > 0
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1 # Corresponds to n=1 in previous code
            )

        # Decode and store completions with original item and prompt
        batch_full_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for idx, full_output in enumerate(batch_full_outputs):
            original_item = batch_data[idx]["original_item"]
            prompt_used = batch_prompts[idx]
            # Extract only the newly generated part
            completion_text = extract_completion(full_output, prompt_used)
            all_completions_data.append({
                "completion": completion_text, 
                "original_item": original_item,
                "prompt": prompt_used # Keep prompt for context if needed
            })

        # Add delay if specified
        if delay and i + batch_size < len(prompts_with_items):
            time.sleep(delay)
            
    logger.info(f"Finished generation. Processing {len(all_completions_data)} completions.")
    # --- End Refactored Generation Loop ---

    # --- Process completions (adjusted logic) ---
    # Group completions by original item index
    item_completion_groups = {}
    completion_idx = 0
    for item_idx, item in enumerate(input_items):
        item_key = start_index + item_idx # Use original index as key
        item_completion_groups[item_key] = []
        for _ in range(num_samples_per_item):
            if completion_idx < len(all_completions_data):
                # Ensure the completion corresponds to the correct original item
                # Note: This relies on the order being preserved. If batching/async changes order, 
                # more robust matching would be needed.
                comp_data = all_completions_data[completion_idx]
                # Double check item match (optional, good for robustness)
                # if comp_data["original_item"] == item: 
                item_completion_groups[item_key].append(comp_data)
                # else: logger.warning("Item mismatch detected!")
                completion_idx += 1
            else:
                 logger.warning(f"Missing completion for item index {item_key}, sample {_}")

    # Process grouped completions
    generated_items = []
    for item_idx, item in enumerate(input_items):
         original_item_key = start_index + item_idx
         completions_for_item = item_completion_groups.get(original_item_key, [])

         for j, comp_data in enumerate(completions_for_item):
            completion_text = comp_data["completion"]
            prompt_used = comp_data["prompt"] # Use the prompt stored during generation

            # Extract generated content using the actual completion text
            generated_content = extract_generated_content(completion_text, 
                                                          "", # Pass empty prompt here as we already extracted
                                                          mode)
            
            if not generated_content:
                logger.warning(f"Failed to extract structured content for item {original_item_key}, sample {j}. Raw completion: '{completion_text[:100]}...'")
                continue
            
            # Create new item with generated content
            new_item = item.copy()
            
            if mode == "S->C":
                new_item["concept"] = generated_content
            elif mode == "C->I":
                new_item["instruction"] = generated_content
            elif mode == "I->R":
                new_item["response"] = generated_content
            
            generated_items.append(new_item)
    # --- End Process completions ---

    # Combine with existing data
    all_items = existing_data + generated_items
    
    # Save generated data
    write_jsonl(all_items, output_path)
    logger.info(f"Saved {len(generated_items)} newly generated items to {output_path}")
    logger.info(f"Total items: {len(all_items)}")

def main():
    """Main function to run the self-OSS-instruct pipeline"""
    parser = argparse.ArgumentParser(description="Self-OSS-Instruct: Generate instruction-response pairs")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input data")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the generated data")
    parser.add_argument("--mode", type=str, required=True, choices=["S->C", "C->I", "I->R"],
                        help="Instruction mode (S->C, C->I, I->R)")
    parser.add_argument("--model", type=str, default="bigcode/starcoder2-15b",
                        help="Model name or path")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_output_tokens", type=int, default=1024,
                        help="Maximum output tokens")
    parser.add_argument("--num_samples_per_item", type=int, default=1,
                        help="Number of samples to generate per item")
    parser.add_argument("--num_examples", type=int, default=8,
                        help="Number of examples to use (0 for none)")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Index to start generation from")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Maximum number of items to generate for")
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Path to continue from (loads existing data)")
    parser.add_argument("--delay", type=float, default=None,
                        help="Delay between batches")
    parser.add_argument("--use_chat_format", action="store_true",
                        help="Use chat format")
    parser.add_argument("--quantization_bits", type=int, choices=[4, 8], default=8,
                        help="Load model in 4-bit or 8-bit precision (requires bitsandbytes). Default: 8")
    
    args = parser.parse_args()
    
    # Run the generation (synchronously)
    generate_instruction_pairs(
        input_data=read_jsonl(args.input_file),
        output_path=args.output_file,
        mode=args.mode,
        model_name=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        num_samples_per_item=args.num_samples_per_item,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        seed=args.seed,
        start_index=args.start_index,
        max_items=args.max_items,
        continue_from=args.continue_from,
        delay=args.delay,
        use_chat_format=args.use_chat_format,
        quantization_bits=args.quantization_bits
    )

if __name__ == "__main__":
    main() 