import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from typing import Optional, Tuple

# Transformer Encoder block
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Ensure embedding dimension is divisible by the number of attention heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Layer normalization before multi-head attention
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Identity layer for residual connection
        self.add1 = nn.Identity()

        # Layer normalization before feed-forward network
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.ReLU(inplace=True)
        )
        # Identity layer for residual connection
        self.add2 = nn.Identity()
        # Causal mask for autoregressive tasks
        self.causal_mask = None

    def forward(self, x):
        seq_len = x.shape[1]
        # Create or adjust the causal mask for the current sequence length
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            self.causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Apply multi-head attention with residual connection
        norm_x = self.layer_norm1(x)
        attn_output, _ = self.multihead_attn(norm_x, norm_x, norm_x, attn_mask=mask)
        x = self.add1(x + attn_output)
        
        # Apply feed-forward network with residual connection
        norm_x = self.layer_norm2(x)
        ff_output = self.ff(norm_x)
        x = self.add2(x + ff_output)
        
        return x

# Transformer Model for sequence-to-sequence tasks
class TransformerModel(nn.Module):
    def __init__(self, seq_len, features, embed_dim, num_heads, ff_dim, num_layers=6, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.features = features

        # Input projection layer to map input features to embedding dimension
        self.input_projection = nn.Linear(features, embed_dim)
        # Positional encoding for sequence data
        self.register_buffer('pos_encoding', self.create_positional_encoding(seq_len, embed_dim))

        # Stack of Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final layer normalization and output layer
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc_out = nn.Linear(embed_dim, 1, bias=True)

        # Initialize weights for all layers
        self.apply(self._init_weights)

    def forward(self, x):
        # Add positional encoding to input embeddings
        x = self.input_projection(x) + self.pos_encoding[:, :x.shape[1], :]

        # Pass through each Transformer encoder layer
        for encoder in self.encoder_layers:
            x = encoder(x)

        # Take the last timestep's output instead of global average pooling
        x = x[:, -1, :]  # Shape: (batch_size, embed_dim)

        # Apply final layer normalization and output layer
        x = self.layer_norm(x)
        x = self.fc_out(x)
        x = x.squeeze(-1)

        return x

    def _init_weights(self, module):
        # Initialize weights for different layers
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # Kaiming initialization
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def create_positional_encoding(self, seq_len, embed_dim):
        # Create sinusoidal positional encoding
        position = torch.arange(seq_len, dtype=torch.float, device=torch.device('cpu')).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float, device=torch.device('cpu')) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, seq_len, embed_dim, device=torch.device('cpu'))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    