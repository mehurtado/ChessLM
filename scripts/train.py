#!/usr/bin/env python3
"""Script to train the chess transformer."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import main

if __name__ == "__main__":
    main()
