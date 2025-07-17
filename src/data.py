
"""Simple chess tokenizer and dataset."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleChessTokenizer:
    """Simple chess move tokenizer."""
    
    def __init__(self, vocab_size: int = 250):
        """Initialize tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        
        # Will be built during training
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        self.vocab_built = False
    
    def build_vocab(self, games: List[List[str]]) -> None:
        """Build vocabulary from games.
        
        Args:
            games: List of games, each game is a list of moves
        """
        logger.info("Building vocabulary...")
        
        # Count all tokens
        token_counts = {}
        for game in games:
            for move in game:
                # Split move into characters and common patterns
                tokens = self._tokenize_move(move)
                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1
        
        # Sort by frequency and take top tokens
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add tokens to vocabulary
        current_id = len(self.special_tokens)
        for token, count in sorted_tokens:
            if current_id >= self.vocab_size:
                break
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        self.vocab_built = True
        logger.info(f"Built vocabulary with {len(self.token_to_id)} tokens")
    
    def _tokenize_move(self, move: str) -> List[str]:
        """Tokenize a single move into subword tokens.
        
        Args:
            move: Chess move in SAN notation
            
        Returns:
            List of tokens
        """
        # Simple tokenization: split into characters and common patterns
        tokens = []
        
        # Common chess patterns
        patterns = ["O-O-O", "O-O", "=Q", "=R", "=B", "=N"]
        
        remaining = move
        for pattern in patterns:
            if pattern in remaining:
                parts = remaining.split(pattern)
                if len(parts) > 1:
                    tokens.extend([c for c in parts[0] if c])
                    tokens.append(pattern)
                    remaining = pattern.join(parts[1:])
        
        # Add remaining characters
        tokens.extend([c for c in remaining if c and c not in "?!"])
        
        return [t for t in tokens if t.strip()]
    
    def encode(self, moves: List[str], max_length: int = 256) -> Dict[str, torch.Tensor]:
        """Encode a sequence of moves.
        
        Args:
            moves: List of chess moves
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        # Tokenize all moves
        tokens = [self.special_tokens["<bos>"]]
        for move in moves:
            move_tokens = self._tokenize_move(move)
            for token in move_tokens:
                token_id = self.token_to_id.get(token, self.special_tokens["<unk>"])
                tokens.append(token_id)
        tokens.append(self.special_tokens["<eos>"])
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        attention_mask = [1] * len(tokens)
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.special_tokens["<pad>"])
            attention_mask.append(0)
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ["<pad>", "<bos>", "<eos>"]:
                    tokens.append(token)
        
        return " ".join(tokens)
    
    def save(self, path: str) -> None:
        """Save tokenizer to file.
        
        Args:
            path: File path to save to
        """
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "special_tokens": self.special_tokens,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved tokenizer to {path}")
    
    @classmethod
    def load(cls, path: str) -> "SimpleChessTokenizer":
        """Load tokenizer from file.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded tokenizer
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.vocab_built = True
        
        logger.info(f"Loaded tokenizer from {path}")
        return tokenizer


class ChessGameDataset(Dataset):
    """Simple chess game dataset."""
    
    def __init__(self, games_file: str, tokenizer: SimpleChessTokenizer, max_length: int = 256):
        """Initialize dataset.
        
        Args:
            games_file: Path to JSONL file with games
            tokenizer: Chess tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load games
        self.games = []
        with open(games_file, 'r') as f:
            for line in f:
                if line.strip():
                    game = json.loads(line)
                    if "moves" in game and len(game["moves"]) > 0:
                        self.games.append(game["moves"])
        
        logger.info(f"Loaded {len(self.games)} games from {games_file}")
    
    def __len__(self) -> int:
        return len(self.games)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single game.
        
        Args:
            idx: Game index
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        moves = self.games[idx]
        encoded = self.tokenizer.encode(moves, self.max_length)
        
        # For language modeling, labels are the same as input_ids shifted by 1
        labels = encoded["input_ids"].clone()
        labels[:-1] = encoded["input_ids"][1:]
        labels[-1] = self.tokenizer.special_tokens["<pad>"]
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


def load_games_for_tokenizer(games_file: str, max_games: int = 1000) -> List[List[str]]:
    """Load games for tokenizer training.
    
    Args:
        games_file: Path to JSONL file
        max_games: Maximum number of games to load
        
    Returns:
        List of games (each game is a list of moves)
    """
    games = []
    with open(games_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_games:
                break
            if line.strip():
                game = json.loads(line)
                if "moves" in game and len(game["moves"]) > 0:
                    games.append(game["moves"])
    
    return games
