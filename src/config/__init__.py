"""
Configuration module for SingN'Seek
Loads and exposes configuration from config.yaml
"""

import yaml
from pathlib import Path

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

__all__ = ["CONFIG"]
