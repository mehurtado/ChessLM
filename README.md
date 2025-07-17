
# Simple Chess Transformer

A simplified chess transformer implementation for training a model to predict chess moves using Stockfish-generated data.

## Features

- Single-threaded Stockfish game generation
- Basic chess move tokenizer
- Simple transformer model
- Basic training loop
- Minimal configuration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Stockfish engine (if not already installed):
```bash
# Ubuntu/Debian
sudo apt-get install stockfish

# macOS
brew install stockfish
```

## Usage

1. Generate training data:
```bash
python scripts/generate_data.py --num-games 1000 --output data/games.jsonl
```

2. Train the model:
```bash
python scripts/train.py --config configs/default.yaml
```

## Project Structure

```
simple-chess-transformer/
├── src/
│   ├── data.py          # Tokenizer and dataset
│   ├── generator.py     # Stockfish game generator
│   ├── model.py         # Simple transformer model
│   └── train.py         # Training script
├── scripts/
│   ├── generate_data.py # Data generation script
│   └── train.py         # Training script
├── configs/
│   └── default.yaml     # Basic configuration
├── data/               # Generated data
├── checkpoints/        # Model checkpoints
└── requirements.txt
```

## Configuration

Edit `configs/default.yaml` to adjust:
- Model size (d_model, num_layers, num_heads)
- Training parameters (learning_rate, batch_size, max_steps)
- Data generation settings (num_games, stockfish_depth)
