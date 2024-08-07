import math
import torch
from torch import nn
from torch.nn import functional as F
import os, sys

# Manually add the root project folder so python knows where to look for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../fmm-attention')))
from attention_mechanisms.score_function import fastmax
from modules import PerTokenClassifier, AvgPoolClassifier, SingleTokenClassifier, ConvEmbedding, FlattenEmbedding

class Attention(nn.Module):
    """The core attention operation of a transformer."""

    def __init__(self, d_model, d_head, h, dropout_rate=0.2, use_masking=False, fastmax=False, use_flash=True, normalize=True, p=2):
        super().__init__()
        self.d_head = d_head
        self.h = h
        self.dropout_rate = dropout_rate
        self.use_masking = use_masking
        self.fastmax = fastmax
        self.flash = use_flash
        self.normalize = normalize
        self.p = p

        self.query_transform = nn.Linear(d_model, h * d_head, bias=False)
        self.key_transform = nn.Linear(d_model, h * d_head, bias=False)
        self.value_transform = nn.Linear(d_model, h * d_head, bias=False)
        self.linear = nn.Linear(h * d_head, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X, ret_attn=False):
        b, n, d_model = X.shape
        Q = self.query_transform(X)
        K = self.key_transform(X)
        V = self.value_transform(X)

        Q = Q.view(b, n, self.h, self.d_head).transpose(1, 2)
        K = K.view(b, n, self.h, self.d_head).transpose(1, 2)
        V = V.view(b, n, self.h, self.d_head).transpose(1, 2)

        if self.normalize:
            Q = Q / (Q.norm(dim=-1, keepdim=True) + 1e-6)
            K = K / (K.norm(dim=-1, keepdim=True) + 1e-6)

        A = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.use_masking:
            mask = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1).to(X.device)
            A = A.masked_fill(mask, float('-inf'))
        A = F.softmax(A, dim=-1)
        if self.training:
            A = self.dropout(A)
        V_hat = torch.matmul(A, V)
        V_hat = V_hat.transpose(1, 2).contiguous().view(b, n, self.h * self.d_head)
        X_hat = self.dropout(self.linear(V_hat))
        return (X_hat, A) if ret_attn else X_hat

class Transformer(nn.Module):
    """Simple model consisting of repeated blocks of Attention + MLP."""
    def __init__(self, out_dim, max_tokens, d_model=512, d_mlp=2048, heads_per_block=8, num_blocks=6, dropout_rate=0.2, use_masking=False, embedding="patch_conv", classifier="cls_token", patch_shape=None, patch_channels=None, normalize=True):
        super().__init__()
        self.token_embedding = ConvEmbedding(patch_channels, patch_shape, d_model)
        self.pos_embedding = nn.Embedding(max_tokens + 1, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(heads_per_block, d_model, d_mlp, dropout_rate, use_masking, normalize=normalize) for _ in range(num_blocks)
        ])
        self.class_token = nn.Parameter(torch.randn(d_model))
        self.classifier = SingleTokenClassifier(d_model, out_dim)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens):
        tok_emb = self.token_embedding(tokens)
        b, n, _ = tok_emb.shape
        cls_emb = self.class_token.expand(b, 1, -1)
        tok_emb = torch.cat((cls_emb, tok_emb), dim=1)
        pos_emb = self.pos_embedding(torch.arange(n+1, device=tokens.device))
        X = tok_emb + pos_emb
        for block in self.blocks:
            X = block(X)
        logits = self.classifier(X)
        return logits

class TransformerBlock(nn.Module):
    """Basic element of a Transformer: Attention + MLP."""
    def __init__(self, num_heads, d_model, d_mlp, dropout_rate=0.2, use_masking=False, normalize=True):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        d_head = d_model // num_heads
        self.attention = Attention(d_model, d_head, num_heads, dropout_rate, use_masking, normalize=normalize)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout_rate),
        )

    def forward(self, X):
        X = self.layer_norm1(X)
        X = X + self.attention(X)
        X = self.layer_norm2(X)
        X = X + self.mlp(X)
        return X

