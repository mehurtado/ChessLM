
"""Simple training script for chess transformer."""

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SimpleChessTokenizer, ChessGameDataset, load_games_for_tokenizer
from model import SimpleChessTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTrainer:
    """Simple trainer for chess transformer."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        Path(config["paths"]["data_dir"]).mkdir(exist_ok=True)
        Path(config["paths"]["checkpoint_dir"]).mkdir(exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.train_loader = None
        
        self.step = 0
        self.best_loss = float('inf')
    
    def setup_tokenizer(self, games_file: str) -> None:
        """Setup tokenizer.
        
        Args:
            games_file: Path to games file
        """
        tokenizer_path = Path(self.config["paths"]["data_dir"]) / "tokenizer.json"
        
        if tokenizer_path.exists():
            logger.info("Loading existing tokenizer...")
            self.tokenizer = SimpleChessTokenizer.load(str(tokenizer_path))
        else:
            logger.info("Creating new tokenizer...")
            self.tokenizer = SimpleChessTokenizer(self.config["model"]["vocab_size"])
            
            # Load games for vocabulary building
            games = load_games_for_tokenizer(games_file, max_games=1000)
            self.tokenizer.build_vocab(games)
            
            # Save tokenizer
            self.tokenizer.save(str(tokenizer_path))
    
    def setup_model(self) -> None:
        """Setup model."""
        model_config = self.config["model"]
        
        self.model = SimpleChessTransformer(
            vocab_size=len(self.tokenizer.token_to_id),
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            max_seq_len=model_config["max_seq_len"],
            dropout=model_config["dropout"],
            pad_token_id=self.tokenizer.special_tokens["<pad>"]
        ).to(self.device)
        
        logger.info(f"Model has {self.model.count_parameters():,} parameters")
    
    def setup_optimizer(self) -> None:
        """Setup optimizer."""
        training_config = self.config["training"]
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=0.01
        )
        
        # Simple learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config["max_steps"],
            eta_min=training_config["learning_rate"] * 0.1
        )
    
    def setup_data(self, games_file: str) -> None:
        """Setup data loader.
        
        Args:
            games_file: Path to games file
        """
        dataset = ChessGameDataset(
            games_file=games_file,
            tokenizer=self.tokenizer,
            max_length=self.config["model"]["max_seq_len"]
        )
        
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=0,  # Single-threaded
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        logger.info(f"Training on {len(dataset)} games")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs["loss"]
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self) -> (float, Dict[str, Any]):
        """Evaluate model on a subset of training data.

        Returns:
            Average loss, and a sample dict with input and output tokens
        """
        self.model.eval()

        total_loss = 0
        num_batches = 0
        sample = None

        with torch.no_grad():
            for batch in self.train_loader:
                if num_batches >= 10:  # Evaluate on 10 batches
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs["loss"].item()
                num_batches += 1

                if sample is None:
                    # Take first batch as sample
                    sample = {
                        "input_ids": input_ids[0].cpu(),
                        "labels": labels[0].cpu(),
                        "logits": outputs["logits"][0].cpu()  # Assuming model returns logits
                    }

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss, sample
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.best_loss = checkpoint["best_loss"]
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def train(self, games_file: str) -> None:
        """Main training loop.
        
        Args:
            games_file: Path to games file
        """
        # Setup components
        self.setup_tokenizer(games_file)
        self.setup_model()
        self.setup_optimizer()
        self.setup_data(games_file)
        
        training_config = self.config["training"]
        max_steps = training_config["max_steps"]
        eval_steps = training_config["eval_steps"]
        save_steps = training_config["save_steps"]
        
        logger.info("Starting training...")
        
        # Training loop
        pbar = tqdm(total=max_steps, desc="Training")
        
        while self.step < max_steps:
            for batch in self.train_loader:
                if self.step >= max_steps:
                    break
                
                # Training step
                loss = self.train_step(batch)
                
                # Update progress
                pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.6f}"})
                pbar.update(1)
                
                self.step += 1
                
                # Evaluation
                if self.step % eval_steps == 0:
                    eval_loss, sample = self.evaluate()
                    logger.info(f"Step {self.step}: train_loss={loss:.4f}, eval_loss={eval_loss:.4f}")
                    print(f"Step {self.step}: train_loss={loss:.4f}, eval_loss={eval_loss:.4f}")

                    # Decode input tokens
                    input_tokens = self.tokenizer.decode(sample["input_ids"].tolist())
                    print(f"Sample input sequence: {input_tokens}")

                    # Get predicted tokens from logits (argmax)
                    predicted_ids = sample["logits"].argmax(dim=-1).tolist()
                    predicted_tokens = self.tokenizer.decode(predicted_ids)
                    print(f"Model output sequence: {predicted_tokens}")

                    # Save best model
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        best_model_path = Path(self.config["paths"]["checkpoint_dir"]) / "best_model.pt"
                        self.save_checkpoint(str(best_model_path))
        
        pbar.close()
        
        # Save final model
        final_model_path = self.config["paths"]["model_save_path"]
        self.save_checkpoint(final_model_path)
        
        logger.info("Training completed!")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train simple chess transformer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--games", type=str, default="data/games.jsonl", help="Games file path")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = SimpleTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(args.games)


if __name__ == "__main__":
    main()
