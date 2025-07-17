#!/usr/bin/env python3
"""Demo script showing how to use the trained chess transformer."""

import sys
from pathlib import Path
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import SimpleChessTokenizer
from model import SimpleChessTransformer


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str):
    """Load trained model and tokenizer."""
    # Load tokenizer
    tokenizer = SimpleChessTokenizer.load(tokenizer_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = SimpleChessTransformer(
        vocab_size=len(tokenizer.token_to_id),
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout'],
        pad_token_id=tokenizer.special_tokens["<pad>"]
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer


def generate_moves(model, tokenizer, starting_moves=None, max_new_moves=10):
    """Generate chess moves using the trained model."""
    if starting_moves is None:
        starting_moves = ["e4", "e5", "Nf3"]
    
    print(f"Starting position: {' '.join(starting_moves)}")
    
    # Encode starting moves
    encoded = tokenizer.encode(starting_moves, max_length=128)
    input_ids = encoded['input_ids'].unsqueeze(0)  # Add batch dimension
    
    # Generate new moves
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=max_new_moves,
            temperature=0.8,
            top_k=20
        )
    
    # Decode generated sequence
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Generated continuation: {generated_text}")
    
    return generated_text


def main():
    """Main demo function."""
    print("🏁 Chess Transformer Demo")
    print("=" * 50)
    
    # Paths
    checkpoint_path = "checkpoints/best_model.pt"
    tokenizer_path = "data/tokenizer.json"
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(checkpoint_path, tokenizer_path)
        
        print(f"✓ Model loaded with {model.count_parameters():,} parameters")
        print(f"✓ Tokenizer loaded with {len(tokenizer.token_to_id)} tokens")
        print()
        
        # Generate some moves
        print("Generating chess moves...")
        print("-" * 30)
        
        # Example 1: King's pawn opening
        generate_moves(model, tokenizer, ["e4"], max_new_moves=5)
        print()
        
        # Example 2: Queen's pawn opening
        generate_moves(model, tokenizer, ["d4", "d5"], max_new_moves=5)
        print()
        
        # Example 3: Sicilian defense
        generate_moves(model, tokenizer, ["e4", "c5", "Nf3"], max_new_moves=5)
        print()
        
        print("✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
