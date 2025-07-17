import chess.pgn
import json
from io import StringIO

def is_grandmaster_game(game, min_elo=2500):
    white_title = game.headers.get("WhiteTitle", "")
    black_title = game.headers.get("BlackTitle", "")
    if "GM" in white_title or "GM" in black_title:
        return True

    try:
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        if white_elo >= min_elo and black_elo >= min_elo:
            return True
    except ValueError:
        pass

    return False

def parse_pgn_grandmaster_only(pgn_text: str):
    pgn_io = StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        if is_grandmaster_game(game):
            yield game

def game_to_moves(game):
    board = game.board()
    moves = []
    for move in game.mainline_moves():
        san = board.san(move)
        moves.append(san)
        board.push(move)
    return moves

def save_games_to_jsonl(games, output_path):
    count = 0
    with open(output_path, 'w') as f:
        for game in games:
            moves = game_to_moves(game)
            json_line = json.dumps({"moves": moves})
            f.write(json_line + "\n")
            count += 1
    print(f"Saved {count} grandmaster games to {output_path}")

def main():
    local_pgn_path = "sample_games.pgn"  # Replace with your PGN file path

    print(f"Reading PGN file: {local_pgn_path}")
    with open(local_pgn_path, 'r', encoding='utf-8') as f:
        pgn_text = f.read()

    games = list(parse_pgn_grandmaster_only(pgn_text))
    print(f"Parsed {len(games)} grandmaster games")

    save_games_to_jsonl(games, "grandmaster_games.jsonl")

if __name__ == "__main__":
    main()