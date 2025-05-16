#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script showing how to use a fine-tuned model for inference
"""

import os
import argparse
import logging
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_inference(
    model_path: str,
    instruction: str,
    max_length: int = 1024,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 50
):
    """
    Run inference with a fine-tuned model
    
    Args:
        model_path: Path to the fine-tuned model
        instruction: Instruction to generate code for
        max_length: Maximum length of the generated text
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        
    Returns:
        The generated code
    """
    logger.info(f"Running inference with model from {model_path}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("Please install transformers to use this function")
        return None
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Prepare the input
    prompt = f"<instruction>\n{instruction}\n</instruction>\n<response>"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    if torch.cuda.is_available():
        inputs = {key: value.cuda() for key, value in inputs.items()}
    
    # Generate the response
    logger.info("Generating response...")
    generate_kwargs = {
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    
    # Decode the response
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract the response part
    response_start = output_text.find("<response>") + len("<response>")
    response_end = output_text.find("</response>", response_start)
    
    if response_end == -1:  # If </response> tag is not found
        response = output_text[response_start:].strip()
    else:
        response = output_text[response_start:response_end].strip()
    
    return response

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate code using a fine-tuned model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="fine_tuned_model",
        help="Path to the fine-tuned model"
    )
    
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Instruction to generate code for"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum length of the generated text"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    
    args = parser.parse_args()
    
    # Run inference
    response = run_inference(
        model_path=args.model_path,
        instruction=args.instruction,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Print the response
    if response:
        print("\n" + "=" * 40 + " GENERATED CODE " + "=" * 40)
        print(response)
        print("=" * 90 + "\n")

if __name__ == "__main__":
    main() 