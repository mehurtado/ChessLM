import chess
import chess.engine
import torch
import torch.nn.functional as F
import os

# Adjust this path to your Stockfish binary location
STOCKFISH_PATH = "/usr/games/stockfish"

class RLChessAgent:
    def __init__(self, model, tokenizer, device='cpu', stockfish_path=STOCKFISH_PATH):
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish binary not found at {stockfish_path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # Set model to eval mode

    def generate_move(self, board):
        """
        Generate a move using the transformer model.
        Args:
            board (chess.Board): Current board state.
        Returns:
            chess.Move: The selected move.
        """
        # Get move history in SAN format
        move_history = [board.san(move) for move in board.move_stack]
        
        # Encode move history tokens
        input_tokens = self.tokenizer.encode(move_history)
        input_ids = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(0)  # batch size 1
        
        # Generate next token logits from model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs["logits"]  # shape: (1, seq_len, vocab_size)
        
        # Get logits for the last token position
        next_token_logits = logits[0, -1, :]
        
        # Greedy decoding (argmax)
        next_token_id = torch.argmax(next_token_logits).item()
        
        # Decode next token to move string
        next_move_str = self.tokenizer.decode([next_token_id])
        
        # Validate and convert to chess.Move
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            # Compare SAN or UCI with decoded move string
            if board.san(move) == next_move_str or move.uci() == next_move_str:
                return move
        
        # If decoded move is illegal or not found, fallback to first legal move
        print(f"Model generated illegal or unknown move '{next_move_str}', falling back to first legal move.")
        return legal_moves[0] if legal_moves else None

    def evaluate_move(self, board, move):
        if move not in board.legal_moves:
            return -10  # Penalty for illegal move

        board_copy = board.copy()
        board_copy.push(move)

        info = self.engine.analyse(board_copy, chess.engine.Limit(depth=10))
        score = info["score"].white().score(mate_score=10000)
        if score is None:
            # Mate detected, assign large positive or negative reward
            if info["score"].white().mate() > 0:
                score = 10000
            else:
                score = -10000

        return score / 100.0  # Scale centipawns to roughly [-100, 100]

    def rl_step(self, board):
        move = self.generate_move(board)
        reward = self.evaluate_move(board, move)
        print(f"Move: {board.san(move)}, Reward: {reward}")
        return move, reward

    def close(self):
        self.engine.quit()


import sys
import os

# Add the parent directory of the script to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import SimpleChessTokenizer, ChessGameDataset, load_games_for_tokenizer
from model import SimpleChessTransformer

def main():
    # Load tokenizer
    tokenizer_path = "../data/tokenizer.json"
    tokenizer = SimpleChessTokenizer.load(tokenizer_path)

    # Initialize model with tokenizer vocab size and config matching your training
    model = SimpleChessTransformer(
        vocab_size=len(tokenizer.token_to_id),
        d_model=512,          # Use your actual config values here
        num_layers=6,
        num_heads=8,
        max_seq_len=512,
        dropout=0.1,
        pad_token_id=tokenizer.special_tokens["<pad>"]
    )

    # Load pretrained weights
    checkpoint_path = "checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    agent = RLChessAgent(model=model, tokenizer=tokenizer, device=device)

    board = chess.Board()
    print("Starting position:")
    print(board)
    print()

    for step in range(10):
        move, reward = agent.rl_step(board)
        if move in board.legal_moves:
            board.push(move)
        else:
            print("Illegal move generated, skipping push.")
        print(f"After move {step + 1}:")
        print(board)
        print()

    agent.close()


if __name__ == "__main__":
    main()