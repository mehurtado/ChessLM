
"""Simple transformer model for chess."""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        normed_x = self.norm1(x)
        
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to key padding mask
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        attn_output, _ = self.attention(
            normed_x, normed_x, normed_x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)
        
        return x


class SimpleChessTransformer(nn.Module):
    """Simple transformer model for chess move prediction."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """Initialize chess transformer.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pad_token_id: Padding token ID
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels for loss computation (batch_size, seq_len)
            
        Returns:
            Dictionary with logits and optionally loss
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        # Clamp positions to max_seq_len to avoid index errors
        positions = torch.clamp(positions, 0, self.max_seq_len - 1)
        position_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result["loss"] = loss
        
        return result
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """Generate chess moves.
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated token IDs (batch_size, seq_len + max_length)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs["logits"]
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token generated
                if next_token.item() == 3:  # EOS token ID
                    break
        
        return input_ids
