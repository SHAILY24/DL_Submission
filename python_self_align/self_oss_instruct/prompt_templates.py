"""
Enhanced Python-specific prompt templates for each stage of the self-OSS-instruct pipeline
"""

# S→C: Seed to Concept prompt with improved guidance
PYTHON_S2C_PROMPT = """Below is a Python function. Describe its functionality in a single comprehensive paragraph.
Include information about:
1. What the function does
2. Its inputs and outputs, including types
3. Any key algorithms or techniques it uses
4. Edge cases it handles
5. Any notable Python-specific features it demonstrates

Python function:
{seed}

Description:"""

# C→I: Concept to Instruction prompt with clearer transformation guidance
PYTHON_C2I_PROMPT = """Below is a description of what a Python function does:

{concept}

Convert this description into a clear coding instruction for implementing this function.
Your instruction should:
1. Clearly specify input parameters and their types
2. Clearly specify the expected return value and its type
3. Provide any necessary context about the algorithm or approach
4. Mention any edge cases that should be handled
5. Encourage the use of modern Python features (type hints, f-strings, etc.)

Instruction:"""

# I→R: Instruction to Response prompt with quality guidelines
PYTHON_I2R_PROMPT = """Write Python code according to the following instruction. 
Your code should:
1. Include comprehensive docstrings with type annotations
2. Follow PEP 8 style guidelines
3. Include comments for complex logic
4. Handle edge cases appropriately
5. Be efficient and well-structured
6. Use type hints for all parameters and return values

Instruction: {instruction}

Python code:"""

# Few-shot examples for Seed to Concept (S→C)
PYTHON_S2C_EXAMPLES = [
    {
        "seed": """def filter_list(items, predicate):
    \"\"\"
    Filter a list based on a predicate function.
    
    Args:
        items (list): The list to filter
        predicate (callable): A function that takes an item and returns a boolean
            
    Returns:
        list: A new list containing only items where predicate(item) is True
    \"\"\"
    result = []
    for item in items:
        if predicate(item):
            result.append(item)
    return result""",
        "concept": """This function filters a list by applying a predicate function to each element and retaining only the elements for which the predicate returns True. It takes two parameters: a list of items to filter and a callable predicate function that evaluates each item. The function iterates through each element in the input list, applies the predicate, and if the result is True, adds that element to a new result list. It then returns this filtered list, preserving the original list unchanged. This implements the common filtering operation found in functional programming, similar to Python's built-in filter function but with a straightforward imperative implementation."""
    },
    {
        "seed": """def merge_dicts(dict1, dict2, conflict_resolver=lambda x, y: y):
    \"\"\"
    Merge two dictionaries with a custom conflict resolution strategy.
    
    Args:
        dict1 (dict): First dictionary
        dict2 (dict): Second dictionary
        conflict_resolver (callable): A function that takes the values from dict1 and dict2
                                     for a conflicting key and returns the value to use
                                     
    Returns:
        dict: A new dictionary containing keys from both input dictionaries
    \"\"\"
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            result[key] = conflict_resolver(result[key], value)
        else:
            result[key] = value
    return result""",
        "concept": """This function merges two dictionaries into a new one with a customizable conflict resolution strategy. It takes three parameters: two dictionaries to merge and an optional callable conflict resolver function that determines how to handle key collisions. By default, the resolver keeps the value from the second dictionary. The function creates a copy of the first dictionary, then iterates through the second dictionary, adding its keys and values. For any key that exists in both dictionaries, it applies the conflict resolver function to determine which value to keep. This provides a flexible way to combine dictionaries beyond Python's built-in dictionary update methods, allowing for sophisticated handling of duplicate keys."""
    }
]

# Few-shot examples for Concept to Instruction (C→I)
PYTHON_C2I_EXAMPLES = [
    {
        "concept": """This function filters a list by applying a predicate function to each element and retaining only the elements for which the predicate returns True. It takes two parameters: a list of items to filter and a callable predicate function that evaluates each item. The function iterates through each element in the input list, applies the predicate, and if the result is True, adds that element to a new result list. It then returns this filtered list, preserving the original list unchanged. This implements the common filtering operation found in functional programming, similar to Python's built-in filter function but with a straightforward imperative implementation.""",
        "instruction": """Write a Python function called `filter_list` that filters elements from a list based on a predicate function. The function should:

1. Accept two parameters:
   - `items` (List[Any]): A list containing elements of any type to be filtered
   - `predicate` (Callable[[Any], bool]): A function that takes a single item from the list and returns a boolean

2. Create a new empty list to store the filtered results
3. Iterate through each element in the input list
4. Apply the predicate function to each element
5. If the predicate returns True, append the element to the result list
6. Return the new filtered list, leaving the original list unchanged

Include proper type hints and docstrings. The function should handle empty lists and predicates that might return non-boolean values by coercing them to booleans."""
    },
    {
        "concept": """This function merges two dictionaries into a new one with a customizable conflict resolution strategy. It takes three parameters: two dictionaries to merge and an optional callable conflict resolver function that determines how to handle key collisions. By default, the resolver keeps the value from the second dictionary. The function creates a copy of the first dictionary, then iterates through the second dictionary, adding its keys and values. For any key that exists in both dictionaries, it applies the conflict resolver function to determine which value to keep. This provides a flexible way to combine dictionaries beyond Python's built-in dictionary update methods, allowing for sophisticated handling of duplicate keys.""",
        "instruction": """Implement a Python function named `merge_dicts` that merges two dictionaries with a custom conflict resolution strategy. The function should:

1. Accept three parameters:
   - `dict1` (Dict[K, V]): The first dictionary
   - `dict2` (Dict[K, V]): The second dictionary
   - `conflict_resolver` (Callable[[V, V], V], optional): A function that takes two values (from dict1 and dict2) for a conflicting key and returns the value to use. Default should keep the value from dict2.

2. Create a copy of the first dictionary to avoid modifying the original
3. Iterate through all key-value pairs in the second dictionary
4. For each key in the second dictionary:
   - If the key already exists in the result dictionary, apply the conflict_resolver function to determine which value to keep
   - If the key doesn't exist in the result, add the key-value pair to the result
5. Return the new merged dictionary

Include proper type hints using Python's typing module. The function should handle empty dictionaries and ensure the original dictionaries remain unchanged."""
    }
]

# Few-shot examples for Instruction to Response (I→R)
PYTHON_I2R_EXAMPLES = [
    {
        "instruction": """Write a Python function called `filter_list` that filters elements from a list based on a predicate function. The function should:

1. Accept two parameters:
   - `items` (List[Any]): A list containing elements of any type to be filtered
   - `predicate` (Callable[[Any], bool]): A function that takes a single item from the list and returns a boolean

2. Create a new empty list to store the filtered results
3. Iterate through each element in the input list
4. Apply the predicate function to each element
5. If the predicate returns True, append the element to the result list
6. Return the new filtered list, leaving the original list unchanged

Include proper type hints and docstrings. The function should handle empty lists and predicates that might return non-boolean values by coercing them to booleans.""",
        "response": """from typing import List, Callable, TypeVar, Any

T = TypeVar('T')

def filter_list(items: List[T], predicate: Callable[[T], bool]) -> List[T]:
    \"\"\"
    Filter a list based on a predicate function.
    
    Args:
        items: The list of items to filter
        predicate: A function that takes an item and returns a boolean
            
    Returns:
        A new list containing only items where predicate(item) evaluates to True
        
    Examples:
        >>> filter_list([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        [2, 4]
        >>> filter_list(['apple', 'banana', 'cherry'], lambda x: 'a' in x)
        ['apple', 'banana']
    \"\"\"
    result = []
    for item in items:
        if bool(predicate(item)):  # Coerce to boolean for safety
            result.append(item)
    return result"""
    },
    {
        "instruction": """Implement a Python function named `merge_dicts` that merges two dictionaries with a custom conflict resolution strategy. The function should:

1. Accept three parameters:
   - `dict1` (Dict[K, V]): The first dictionary
   - `dict2` (Dict[K, V]): The second dictionary
   - `conflict_resolver` (Callable[[V, V], V], optional): A function that takes two values (from dict1 and dict2) for a conflicting key and returns the value to use. Default should keep the value from dict2.

2. Create a copy of the first dictionary to avoid modifying the original
3. Iterate through all key-value pairs in the second dictionary
4. For each key in the second dictionary:
   - If the key already exists in the result dictionary, apply the conflict_resolver function to determine which value to keep
   - If the key doesn't exist in the result, add the key-value pair to the result
5. Return the new merged dictionary

Include proper type hints using Python's typing module. The function should handle empty dictionaries and ensure the original dictionaries remain unchanged.""",
        "response": """from typing import Dict, Callable, TypeVar, Any

K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type

def merge_dicts(
    dict1: Dict[K, V],
    dict2: Dict[K, V],
    conflict_resolver: Callable[[V, V], V] = lambda x, y: y
) -> Dict[K, V]:
    \"\"\"
    Merge two dictionaries with a custom conflict resolution strategy.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        conflict_resolver: A function that takes the values from dict1 and dict2
                          for a conflicting key and returns the value to use.
                          Default keeps the value from dict2.
                          
    Returns:
        A new dictionary containing keys from both input dictionaries with
        conflicts resolved using the provided conflict_resolver function
        
    Examples:
        >>> merge_dicts({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
        {'a': 1, 'b': 3, 'c': 4}
        
        >>> merge_dicts({'a': 1, 'b': 2}, {'b': 3, 'c': 4}, lambda x, y: x + y)
        {'a': 1, 'b': 5, 'c': 4}
    \"\"\"
    # Create a copy to avoid modifying the original
    result = dict1.copy()
    
    # Process the second dictionary
    for key, value in dict2.items():
        if key in result:
            # Key exists in both dictionaries, use conflict resolver
            result[key] = conflict_resolver(result[key], value)
        else:
            # Key only exists in dict2, add it directly
            result[key] = value
            
    return result"""
    }
]

# System prompts for each stage
SYSTEM_S2C = """You are a helpful AI assistant that specializes in analyzing Python code and providing clear, comprehensive descriptions of what the code does. For each Python function provided, create a detailed explanation that covers:
1. The function's purpose and behavior
2. Input parameters and their types
3. Return values and their types
4. Any algorithms or techniques used
5. Edge cases handled
6. Notable Python features demonstrated"""

SYSTEM_C2I = """You are a helpful AI assistant that specializes in creating clear, detailed programming instructions based on functional descriptions. For each description of a Python function, create an instruction that would guide a programmer to implement it correctly. Your instructions should:
1. Be clear and specific
2. Include input parameter specifications with types
3. Specify return value requirements with types
4. Mention edge cases to handle
5. Encourage good Python practices (type hints, docstrings, etc.)"""

SYSTEM_I2R = """You are a helpful AI assistant that specializes in writing high-quality Python code based on instructions. For each programming instruction, generate code that:
1. Fully satisfies all requirements
2. Includes comprehensive docstrings with type information
3. Uses proper type hints (PEP 484)
4. Follows PEP 8 style guidelines
5. Handles edge cases appropriately
6. Is efficient and well-structured
7. Uses modern Python features when appropriate"""

# Additional helper functions for prompt formatting

def format_s2c_prompt(seed: str, examples: list = None, use_chat_format: bool = False) -> str:
    """Format the Seed to Concept prompt"""
    if not use_chat_format:
        base_prompt = PYTHON_S2C_PROMPT.format(seed=seed)
        
        # Add examples if provided
        if examples:
            examples_text = ""
            for example in examples:
                examples_text += f"\n\nPython function:\n{example['seed']}\n\nDescription:\n{example['concept']}"
            
            base_prompt = examples_text + "\n\n" + base_prompt
            
        return base_prompt
    else:
        # Chat format with system prompt and examples
        messages = [
            {"role": "system", "content": SYSTEM_S2C}
        ]
        
        # Add examples if provided
        if examples:
            for example in examples:
                messages.append({
                    "role": "user", 
                    "content": PYTHON_S2C_PROMPT.format(seed=example['seed'])
                })
                messages.append({
                    "role": "assistant", 
                    "content": example['concept']
                })
        
        # Add the current query
        messages.append({
            "role": "user",
            "content": PYTHON_S2C_PROMPT.format(seed=seed)
        })
        
        return messages

def format_c2i_prompt(concept: str, examples: list = None, use_chat_format: bool = False) -> str:
    """Format the Concept to Instruction prompt"""
    if not use_chat_format:
        base_prompt = PYTHON_C2I_PROMPT.format(concept=concept)
        
        # Add examples if provided
        if examples:
            examples_text = ""
            for example in examples:
                examples_text += f"\n\nDescription:\n{example['concept']}\n\nInstruction:\n{example['instruction']}"
            
            base_prompt = examples_text + "\n\n" + base_prompt
            
        return base_prompt
    else:
        # Chat format with system prompt and examples
        messages = [
            {"role": "system", "content": SYSTEM_C2I}
        ]
        
        # Add examples if provided
        if examples:
            for example in examples:
                messages.append({
                    "role": "user", 
                    "content": PYTHON_C2I_PROMPT.format(concept=example['concept'])
                })
                messages.append({
                    "role": "assistant", 
                    "content": example['instruction']
                })
        
        # Add the current query
        messages.append({
            "role": "user",
            "content": PYTHON_C2I_PROMPT.format(concept=concept)
        })
        
        return messages

def format_i2r_prompt(instruction: str, examples: list = None, use_chat_format: bool = False) -> str:
    """Format the Instruction to Response prompt"""
    if not use_chat_format:
        base_prompt = PYTHON_I2R_PROMPT.format(instruction=instruction)
        
        # Add examples if provided
        if examples:
            examples_text = ""
            for example in examples:
                examples_text += f"\n\nInstruction:\n{example['instruction']}\n\nPython code:\n{example['response']}"
            
            base_prompt = examples_text + "\n\n" + base_prompt
            
        return base_prompt
    else:
        # Chat format with system prompt and examples
        messages = [
            {"role": "system", "content": SYSTEM_I2R}
        ]
        
        # Add examples if provided
        if examples:
            for example in examples:
                messages.append({
                    "role": "user", 
                    "content": PYTHON_I2R_PROMPT.format(instruction=example['instruction'])
                })
                messages.append({
                    "role": "assistant", 
                    "content": example['response']
                })
        
        # Add the current query
        messages.append({
            "role": "user",
            "content": PYTHON_I2R_PROMPT.format(instruction=instruction)
        })
        
        return messages 