import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

# Transformer Encoder block
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # Multi-head self-attention + Residual Connection
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.multihead_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)

        # Feedforward + Residual Connection
        y_norm = self.layer_norm2(x)
        y = F.relu(self.fc1(y_norm))
        y = self.dropout2(y)
        y = self.fc2(y)
        return x + y

# Transformer Model block
class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, features, embed_dim, num_heads, ff_dim, num_layers, dropout=0.0):
        super(TransformerModel, self).__init__()
        
        assert input_dim == seq_len * features, f"Expected input_dim={seq_len * features} for {seq_len} time steps of {features} features"        
        self.input_projection = nn.Linear(features, embed_dim)  # Project 5 features to embed_dim
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.seq_len = seq_len
        self.features = features
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = x.view(x.shape[0], self.seq_len, self.features)  # Reshape (batch_size, 60) to (batch_size, seq_len, features)
        x = self.input_projection(x)   # Project features to embed_dim
        
        for encoder in self.encoder_layers:
            x = encoder(x)
        
        x = self.global_avg_pool(x.permute(0, 2, 1)).squeeze(-1)
        x = self.layer_norm(x)
        return self.fc_out(x).squeeze(-1)

