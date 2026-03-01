"""Logging utilities for experiment results."""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


def save_jsonl(data: List[Dict[str, Any]], filepath: Path) -> None:
    """Save a list of dictionaries as JSONL.
    
    Args:
        data: List of dictionaries to save.
        filepath: Path to save the JSONL file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for record in data:
            # Convert numpy types to native Python types for JSON serialization
            record_serializable = _make_serializable(record)
            f.write(json.dumps(record_serializable) + "\n")


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    else:
        return obj


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file.
    
    Args:
        filepath: Path to the JSONL file.
        
    Returns:
        List of dictionaries.
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data
