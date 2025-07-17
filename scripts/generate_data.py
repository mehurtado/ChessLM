
#!/usr/bin/env python3
"""Script to generate chess training data using Stockfish."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import SimpleStockfishGenerator


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate chess training data")
    parser.add_argument("--num-games", type=int, default=1000, help="Number of games to generate")
    parser.add_argument("--output", type=str, default="data/games.jsonl", help="Output file path")
    parser.add_argument("--depth", type=int, default=10, help="Stockfish search depth")
    parser.add_argument("--skill", type=int, default=15, help="Stockfish skill level (0-20)")
    parser.add_argument("--max-moves", type=int, default=100, help="Maximum moves per game")
    parser.add_argument("--min-moves", type=int, default=10, help="Minimum moves per game")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_games} games...")
    print(f"Stockfish depth: {args.depth}, skill: {args.skill}")
    print(f"Output: {args.output}")
    
    # Create generator
    generator = SimpleStockfishGenerator(
        depth=args.depth,
        skill_level=args.skill
    )
    
    # Generate games
    generator.generate_games(args.num_games, args.output)
    
    print("Data generation completed!")


if __name__ == "__main__":
    main()
