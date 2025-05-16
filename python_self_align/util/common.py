#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common utility functions for the Python Self-Align project.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def ensure_directory(directory: str) -> str:
    """Ensure a directory exists and return its path"""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_file_size(file_path: str) -> str:
    """Get human-readable file size"""
    size_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def count_items(file_path: str) -> int:
    """Count items in a JSONL file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.error(f"Error counting items in {file_path}: {e}")
        return 0 