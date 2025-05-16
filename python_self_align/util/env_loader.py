#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for loading environment variables.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_env_vars(env_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file
    
    Args:
        env_file: Path to .env file (default: look in current and parent directories)
        
    Returns:
        Dictionary of environment variables
    """
    # If env_file not provided, look in standard locations
    if env_file is None:
        # Look in current directory
        if os.path.exists('.env'):
            env_file = '.env'
        # Look in parent directory
        elif os.path.exists('../.env'):
            env_file = '../.env'
        # Look in project root
        else:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.absolute()
            env_path = project_root / '.env'
            if env_path.exists():
                env_file = str(env_path)
    
    # Load environment variables from .env file if found
    if env_file and os.path.exists(env_file):
        logger.info(f"Loading environment variables from {env_file}")
        dotenv.load_dotenv(env_file)
        # Return a dictionary of the loaded variables
        return {key: value for key, value in os.environ.items()}
    else:
        logger.warning("No .env file found. Using existing environment variables.")
        return {}

def get_api_key(key_type: str = 'huggingface') -> str:
    """
    Get API key for specified service
    
    Args:
        key_type: Type of API key to retrieve ('huggingface' or 'openai')
        
    Returns:
        API key string
    """
    # Load environment variables if not already loaded
    load_env_vars()
    
    # Get the appropriate API key
    if key_type.lower() == 'huggingface':
        api_key = os.environ.get('HUGGINGFACE_API_KEY')
    elif key_type.lower() == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
    else:
        raise ValueError(f"Unknown API key type: {key_type}")
    
    # Check if API key is defined
    if not api_key:
        logger.warning(f"{key_type.upper()}_API_KEY not found in environment variables")
    elif api_key.startswith('your_') or api_key == 'your-api-key-here':
        logger.warning(f"{key_type.upper()}_API_KEY appears to be a placeholder. Please set a real API key.")
        api_key = None
    
    return api_key if api_key else ""

def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a configuration value from environment variables
    
    Args:
        key: Name of the config variable
        default: Default value if not found
        
    Returns:
        Configuration value
    """
    # Load environment variables if not already loaded
    load_env_vars()
    
    # Get the value
    value = os.environ.get(key, default)
    
    # Try to convert to int or float if applicable
    if isinstance(value, str):
        try:
            if value.isdigit():
                return int(value)
            elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                return float(value)
        except (ValueError, TypeError):
            pass
    
    return value 