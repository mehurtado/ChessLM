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

    def generate_move(self, board, max_move_length=5):
        # Use UCI strings for move history
        move_history = [move.uci() for move in board.move_stack]

        # Encode move history tokens
        input_tokens = self.tokenizer.encode(move_history)
        if isinstance(input_tokens, dict):
            input_ids_list = input_tokens.get("input_ids", None)
            if input_ids_list is None:
                raise ValueError("Tokenizer encode output missing 'input_ids'")
        else:
            input_ids_list = input_tokens

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device).unsqueeze(0)  # batch size 1

        generated_tokens = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_move_length):
                outputs = self.model(input_ids=input_ids)
                logits = outputs["logits"]  # (1, seq_len, vocab_size)
                next_token_logits = logits[0, -1, :]

                # Greedy decoding
                next_token_id = torch.argmax(next_token_logits).item()
                generated_tokens.append(next_token_id)

                # Append next token to input_ids for next step
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)

                # Optionally, stop if end-of-move token generated
                # if next_token_id == self.tokenizer.special_tokens.get("<eom>", None):
                #     break

        # Decode full generated token sequence to move string
        next_move_str = self.tokenizer.decode(generated_tokens)

        # Validate and convert to chess.Move
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            if move.uci() == next_move_str or board.san(move) == next_move_str:
                return move

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

from src.data import SimpleChessTokenizer, ChessGameDataset, load_games_for_tokenizer
from src.model import SimpleChessTransformer

import yaml

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run RL agent with chess transformer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Checkpoint path")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer_path = config["paths"]["data_dir"] + "/tokenizer.json"
    tokenizer = SimpleChessTokenizer.load(tokenizer_path)

    # Initialize model with config parameters
    model_cfg = config["model"]
    model = SimpleChessTransformer(
        vocab_size=len(tokenizer.token_to_id),
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg["dropout"],
        pad_token_id=tokenizer.special_tokens["<pad>"]
    )

    checkpoint_path = args.checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"No checkpoint found at {checkpoint_path}, initializing new model.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Initialize RL agent with model and tokenizer
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