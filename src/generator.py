
"""Simple single-threaded Stockfish game generator."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import chess
from stockfish import Stockfish
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleStockfishGenerator:
    """Simple single-threaded Stockfish game generator."""
    
    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 10, skill_level: int = 15):
        """Initialize the generator.
        
        Args:
            stockfish_path: Path to Stockfish binary (auto-detect if None)
            depth: Search depth
            skill_level: Skill level (0-20)
        """
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.depth = depth
        self.skill_level = skill_level
        
        # Test Stockfish
        self._test_stockfish()
    
    def _find_stockfish(self) -> str:
        """Find Stockfish binary."""
        import shutil
        
        candidates = ["stockfish", "stockfish.exe"]
        for candidate in candidates:
            path = shutil.which(candidate)
            if path:
                logger.info(f"Found Stockfish at: {path}")
                return path
        
        # Common paths
        common_paths = [
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/games/stockfish",
            "/opt/homebrew/bin/stockfish",
        ]
        
        for path in common_paths:
            if Path(path).exists():
                logger.info(f"Found Stockfish at: {path}")
                return path
        
        raise FileNotFoundError("Stockfish not found. Please install Stockfish.")
    
    def _test_stockfish(self):
        """Test that Stockfish is working."""
        try:
            engine = Stockfish(path=self.stockfish_path)
            engine.set_depth(1)
            move = engine.get_best_move()
            if move is None:
                raise RuntimeError("Stockfish test failed")
            logger.info("Stockfish test successful")
        except Exception as e:
            raise RuntimeError(f"Stockfish test failed: {e}")
    
    def generate_game(self, max_moves: int = 100, min_moves: int = 10) -> Optional[Dict[str, Any]]:
        """Generate a single chess game.
        
        Args:
            max_moves: Maximum number of moves
            min_moves: Minimum number of moves
            
        Returns:
            Game data dictionary or None if failed
        """
        try:
            # Create engine
            engine = Stockfish(
                path=self.stockfish_path,
                depth=self.depth,
                parameters={
                    "Skill Level": self.skill_level,
                    "UCI_LimitStrength": "true" if self.skill_level < 20 else "false",
                }
            )
            
            board = chess.Board()
            moves = []
            
            # Generate moves
            while not board.is_game_over() and len(moves) < max_moves:
                # Set position
                engine.set_fen_position(board.fen())
                
                # Get best move
                best_move = engine.get_best_move()
                if not best_move:
                    break
                
                # Make move
                try:
                    move = chess.Move.from_uci(best_move)
                    if move in board.legal_moves:
                        # Convert to SAN before making the move
                        san_move = board.san(move)
                        board.push(move)
                        moves.append(san_move)
                    else:
                        break
                except ValueError:
                    break
            
            # Check minimum moves
            if len(moves) < min_moves:
                return None
            
            # Determine result
            result = "*"
            if board.is_checkmate():
                result = "1-0" if board.turn == chess.BLACK else "0-1"
            elif board.is_stalemate() or board.is_insufficient_material():
                result = "1/2-1/2"
            
            return {
                "moves": moves,
                "result": result,
                "num_moves": len(moves),
            }
            
        except Exception as e:
            logger.error(f"Error generating game: {e}")
            return None
    
    def generate_games(self, num_games: int, output_path: str) -> None:
        """Generate multiple games and save to file.
        
        Args:
            num_games: Number of games to generate
            output_path: Output file path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        games = []
        logger.info(f"Generating {num_games} games...")
        
        for i in tqdm(range(num_games), desc="Generating games"):
            game = self.generate_game()
            if game:
                game["game_id"] = i
                games.append(game)
        
        # Save games
        with open(output_path, 'w') as f:
            for game in games:
                f.write(json.dumps(game) + '\n')
        
        logger.info(f"Generated {len(games)} games and saved to {output_path}")
