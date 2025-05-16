"""
Tree-sitter parser utilities for extracting Python functions
"""

import os
import platform
from typing import Any

# Import tree-sitter
from tree_sitter import Language, Parser

# Get the directory of this file
DIR = os.path.dirname(os.path.abspath(__file__))

# Find the correct tree-sitter-python directory path to load the library
def get_lib_path():
    """Get the path to the tree-sitter Python grammar library"""
    
    # Define the expected base path relative to this script
    # Assumes starcoder2-self-align is cloned alongside python_self_align
    expected_base = os.path.abspath(os.path.join(DIR, "..", "..")) # Should be /workspace/DL
    expected_path = os.path.join(expected_base, "starcoder2-self-align", "seed_gathering", "tree-sitter-python")

    print(f"[Debug TreeSitter] Calculated expected path: {expected_path}") # Added for debugging

    if os.path.exists(expected_path):
        print(f"[Debug TreeSitter] Found tree-sitter-python at: {expected_path}") # Added for debugging
        return expected_path

    # Fallback: Check other potential locations (less likely but for robustness)
    possible_paths = [
        os.path.join(DIR, "tree-sitter-python"), # Directly inside seed_gathering?
        os.path.join(DIR, "..", "..", "tree-sitter-python"), # Directly under workspace root?
    ]
    for path in possible_paths:
        print(f"[Debug TreeSitter] Checking alternative path: {path}") # Added for debugging
        if os.path.exists(path):
             print(f"[Debug TreeSitter] Found tree-sitter-python at alternative: {path}") # Added for debugging
             return path

    # If not found locally, fallback to cloning (keep this as last resort)
    import subprocess
    import tempfile
    
    tmp_dir = tempfile.mkdtemp()
    clone_path = os.path.join(tmp_dir, "tree-sitter-python")
    
    try:
        subprocess.check_call(
            ["git", "clone", "https://github.com/tree-sitter/tree-sitter-python", clone_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return clone_path
    except Exception as e:
        raise RuntimeError(f"Failed to clone tree-sitter-python: {e}")

# Get the library path
LIB_PATH = get_lib_path()

# Build and load the Python language
def build_language():
    """Build the tree-sitter Python language"""
    
    # Determine the correct build path
    system = platform.system().lower()
    if system == "windows":
        build_path = os.path.join(DIR, "python.dll")
    elif system == "darwin":
        build_path = os.path.join(DIR, "python.dylib")
    else:  # Linux
        build_path = os.path.join(DIR, "python.so")
    
    # Build the language
    Language.build_library(
        build_path,
        [LIB_PATH]
    )
    
    # Load and return the language
    return Language(build_path, "python")

# Try to load the language, or build it if it doesn't exist
try:
    system = platform.system().lower()
    if system == "windows":
        lang_lib = os.path.join(DIR, "python.dll")
    elif system == "darwin":
        lang_lib = os.path.join(DIR, "python.dylib")
    else:  # Linux
        lang_lib = os.path.join(DIR, "python.so")
        
    if os.path.exists(lang_lib):
        LANGUAGE = Language(lang_lib, "python")
    else:
        LANGUAGE = build_language()
except:
    LANGUAGE = build_language()

def make_parser() -> Parser:
    """Create a new tree-sitter parser with the Python language"""
    parser = Parser()
    parser.set_language(LANGUAGE)
    return parser

def node_to_string(source_bytes: bytes, node: Any) -> str:
    """Convert a tree-sitter node to a string"""
    return source_bytes[node.start_byte:node.end_byte].decode('utf8')

def extract_docstring(node: Any, source_bytes: bytes) -> str:
    """Extract the docstring from a function node"""
    try:
        # Get the block node (function body)
        for child in node.children:
            if child.type == "block":
                block_node = child
                break
        else:
            return ""
        
        # Check the first statement for a docstring
        if len(block_node.children) < 3:  # Needs at least { expression_statement }
            return ""
        
        first_stmt = block_node.children[1]  # Skip the opening {
        if first_stmt.type != "expression_statement":
            return ""
        
        # Check if it's a string (docstring)
        if len(first_stmt.children) == 0:
            return ""
            
        string_node = first_stmt.children[0]
        if string_node.type != "string":
            return ""
        
        # Extract the docstring content
        docstring = node_to_string(source_bytes, string_node)
        
        # Remove the triple quotes
        if docstring.startswith('"""') and docstring.endswith('"""'):
            docstring = docstring[3:-3]
        elif docstring.startswith("'''") and docstring.endswith("'''"):
            docstring = docstring[3:-3]
            
        return docstring.strip()
    except Exception as e:
        print(f"Error extracting docstring: {e}")
        return ""

def extract_return_type(node: Any, source_bytes: bytes) -> str:
    """Extract the return type annotation from a function node"""
    try:
        for child in node.children:
            if child.type == "typed_parameter" and "return" in node_to_string(source_bytes, child):
                return node_to_string(source_bytes, child).split(":")[-1].strip()
        return ""
    except Exception as e:
        print(f"Error extracting return type: {e}")
        return ""
        
def extract_parameters(node: Any, source_bytes: bytes) -> list:
    """Extract the parameters from a function node"""
    params = []
    try:
        for child in node.children:
            if child.type == "parameters":
                param_node = child
                break
        else:
            return []
            
        for child in param_node.children:
            if child.type == "identifier" or child.type == "typed_parameter":
                params.append(node_to_string(source_bytes, child))
                
        return params
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        return []

def check_for_returns(node: Any, source_bytes: bytes) -> bool:
    """Check if a function has return statements"""
    try:
        # Find all return statements
        def visitor(node):
            if node.type == "return_statement":
                return True
            
            # Check children
            for child in node.children:
                if visitor(child):
                    return True
            
            return False
            
        return visitor(node)
    except Exception as e:
        print(f"Error checking for returns: {e}")
        return False

def has_type_annotations(function_node: Any, source_bytes: bytes) -> bool:
    """Check if a function has type annotations"""
    try:
        function_str = node_to_string(source_bytes, function_node)
        # Check for -> in return type
        if "->" in function_str:
            return True
            
        # Check for : in parameter list
        for child in function_node.children:
            if child.type == "parameters":
                params_str = node_to_string(source_bytes, child)
                if ":" in params_str:
                    return True
                    
        return False
    except Exception as e:
        print(f"Error checking for type annotations: {e}")
        return False 