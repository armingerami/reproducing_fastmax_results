import math
import torch
from torch import nn
import kornia

class PerTokenClassifier(nn.Module):
    """Linear classifier that outputs one class prediction per token."""
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, n_classes)
    def forward(self, X):
        
        X = self.layer_norm(X)  # (b, n, d_model)
        logits = self.linear(X)  # (b, n, n_classes)
        return logits
    
class AvgPoolClassifier(nn.Module):
    """Linear classifier that outputs one class prediction by averaging tokens."""
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(d_model, n_classes)
    def forward(self, X):
        X = self.layer_norm(X)  # (b, n, d_model)
        X = X.permute(0, 2, 1)  # (b, d_model, n) to fit AdaptiveMaxPool1d input format
        X = self.pool(X).squeeze(-1)  # (b, d_model)
        logits = self.linear(X)  # (b, n_classes)
        return logits
    
class SingleTokenClassifier(nn.Module):
    """Linear classifier that takes a single token as input"""
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, n_classes)
    def forward(self, X):
        X = self.layer_norm(X)
        X = X[:, 0, :]  # (b, d_model)
        logits = self.linear(X)  # (b, n_classes)
        return logits
    

class ConvEmbedding(nn.Module):
    """Embedding for images where each token is a patch of pixels convolved, n_kernels=d_model."""
    def __init__(self, channels, patch_shape, d_model):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, d_model, kernel_size=patch_shape, stride=patch_shape
        )
    def forward(self, images):
        # images shape: (b, c, h, w)
        X = self.conv(images)  # (b, d_model, h/patch_h, w/patch_w);
        X = X.flatten(2)  # (b, d_model, n)
        return X.transpose(1,2)  # (b, n, d_model)
    
class FlattenEmbedding(nn.Module):
    """Embedding for images where each token is a patch of pixels flattened into a vector."""
    def __init__(self, channels, patch_shape, d_model):
        super().__init__()
        patch_pixels = channels * patch_shape[0] * patch_shape[1] 
        self.linear = nn.Linear(patch_pixels, d_model)
    def forward(self, images):
        # images shape: (b, c, h, w)
        patches = kornia.contrib.extract_tensor_patches(images, window_size=self.patch_shape, stride=self.patch_shape)  # (b, n, c, patch_h, patch_w)
        patches = patches.flatten(2) # (b, n, c*patch_h*patch_w)
        X = self.linear(patches)  # (b, n, d_model)
        return X
    

# Copied from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Use register_buffer to specify that this is a non-learnable parameter.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)