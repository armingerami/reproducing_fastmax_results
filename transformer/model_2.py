import math
import torch
from torch import nn
from torch.nn import functional as F
import os, sys

 # Manually add the root project folder so python knows where to look for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../fmm-attention')))
from score_function import fastmax
from .modules import PerTokenClassifier, AvgPoolClassifier, SingleTokenClassifier, ConvEmbedding, FlattenEmbedding

# For more detailed explanations of each component, see model.ipynb.

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
        # X shape: (b, n, d_model)
        # b: batch size
        # n: num tokens/features
        # d_model: dim of vector representation of each token
        b, n, d_model = X.shape
        # Project into Q, K, V spaces
        Q = self.query_transform(X)  # (b, n, h*d_head)
        K = self.key_transform(X)  # (b, n, h*d_head)
        V = self.value_transform(X)  # (b, n, h*d_head)
        # Reshape into h heads to be run in parallel
        Q = Q.view(b, n, self.h, self.d_head).transpose(1, 2)  # (b, h, n, d_head)
        K = K.view(b, n, self.h, self.d_head).transpose(1, 2)  # (b, h, n, d_head)
        V = V.view(b, n, self.h, self.d_head).transpose(1, 2)  # (b, h, n, d_head)
        A = None
        if self.fastmax:
            if ret_attn:
                V_hat, A = fastmax(Q, K, V, mask=self.use_masking, normalize=self.normalize, p=self.p, create_attn_matrix=True)  # (b, h, n, d_head)
            else:
                V_hat = fastmax(Q, K, V, mask=self.use_masking, normalize=self.normalize, p=self.p)
        elif self.flash and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Pytorch's scaled_dot_product_attention will use FlashAttention CUDA kernels if we are 
            # running on a GPU and a non-Flash but MKL-optimized version on CPU. So if we specify 
            # attention_type="flash" but run on a CPU, this will not throw an error, but it will not 
            # use FlashAttention.
            # Note: scaled_dot_product_attention is only available in PyTorch 2.0+
            V_hat = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout_rate if self.training else 0,
                is_causal=self.use_masking,
            )
        else:
            # manual implementation of scaled dot product attention
            A = Q @ K.mT  # (b, h, n, d_head) @ (b, h, d_head, n) -> (b, h, n, n)
            A = A / math.sqrt(self.d_head)
            if self.use_masking:
                # Mask out upper triangular of A so that weightings only apply to
                # previous tokens in the sequence. Filling with -inf before softmax
                # will cause softmax to give weight 0 to these tokens.
                n = X.shape[1]
                upper_tri_mask = torch.triu(
                    torch.ones((n, n), dtype=bool, device=X.device), diagonal=1
                )  # Upper triangular matrix
                A = A.masked_fill(
                    upper_tri_mask, float("-inf")
                )  # Fill upper triangular with -inf
            A = F.softmax(A, dim=-1)
            if self.training:
                A = self.dropout(A)
            V_hat = A @ V  # (b, h, n, n) @ (b, n, d_head) -> (b, h, n, d_head)
        # Combine heads into single dimension and project back to d_model space
        V_hat = V_hat.transpose(1, 2).contiguous().view(b, n, self.h * self.d_head)  # (b, n, h*d_head)
        X_hat = self.dropout(self.linear(V_hat))  # (b, n, d_model)
        if not ret_attn:
            # Save memory by only retaining attention map if requested
            A = None
        return X_hat, A

class TransformerBlock(nn.Module):
    """Basic element of a Transformer: Attention + MLP."""

    def __init__(self, num_heads, d_model, d_mlp, dropout_rate=0.2, use_masking=False, fastmax=False, use_flash=True, normalize=True, p=2, ret_attn=False):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        d_head = d_model // num_heads
        self.attention_subblock = Attention(d_model, d_head, num_heads, dropout_rate, use_masking, fastmax, use_flash, normalize, p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mlp_subblock = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout_rate),
        )

    def forward(self, X, attn_maps, ret_attn=False):
        # X shape: (b, n, d_model)
        attention_output = self.layer_norm1(X)
        attention_output, attn_map = self.attention_subblock(attention_output, ret_attn)  # (b, n, d_model)
        attention_output = attention_output + X  # residual connection
        mlp_output = self.layer_norm2(attention_output)
        mlp_output = self.mlp_subblock(mlp_output)  # (b, n, d_model)
        mlp_output = mlp_output + X  # residual connection
        attn_maps.append(attn_map)
        return mlp_output, attn_maps


class Transformer(nn.Module):
    """
    Simple model consisting of repeated blocks of Attention + MLP.
    out_dim: again, the desired vector length. If classification, just
        the number of classes. If word prediction, the vocab_size so we can
        output a probability over each word in the vocab.
    max_tokens: the maximum number of tokens in the input. 
        This is only used to define a positional encoding that learns to 
        represent spatial positions up to that number. Can be set to a longer
        lenth than the actual intended number of tokens in the input.
    """
        
    def __init__(
        self,
        out_dim,
        max_tokens,
        d_model=512,
        d_mlp=2048,
        heads_per_block=8,
        num_blocks=6,
        dropout_rate=0.2,
        use_masking=False,
        embedding="discrete_set",
        classifier="per_token",
        vocab_size=None,
        patch_shape=None,
        patch_channels=None,
        fastmax=False,
        use_flash=True,
        normalize=True,
        p=2,
        return_attn_maps=True,
        ):
        super().__init__()
        self.return_attn_maps = return_attn_maps

        # Embedding layers: feature space -> d_model space (X)
        if embedding == "discrete_set":
            self.token_embedding = nn.Embedding(vocab_size, d_model)
        elif embedding == "patch_flatten":
            self.token_embedding = FlattenEmbedding(patch_channels, patch_shape, d_model)
        elif embedding == "patch_conv":
            self.token_embedding = ConvEmbedding(patch_channels, patch_shape, d_model)
        else:
            raise ValueError(f"Unknown embedding type: {embedding}")
        self.pos_embedding = nn.Embedding(max_tokens + 1, d_model) # +1 to account for class token if present
        
        # Transformer core: X -> X_hat
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                TransformerBlock(heads_per_block, d_model, d_mlp, dropout_rate, use_masking, fastmax, use_flash, normalize, p, return_attn_maps)
            )
        
        # Classifier head: X_hat -> logits
        if classifier == "per_token":
            self.classifier = PerTokenClassifier(d_model, out_dim)
        elif classifier == "avg_pool":
            self.classifier = AvgPoolClassifier(d_model, out_dim) 
        elif classifier == "cls_token":
            self.class_token = nn.Parameter(torch.randn(d_model))
            self.classifier = SingleTokenClassifier(d_model, out_dim)
        else:
            raise ValueError(f"Unknown classifier type: {classifier}")
        
        # Weight initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens, return_attn_maps=False):
        # tokens shape: (b, n, d_feature) or (b, n) if discrete vocabulary
        # b: batch size
        # n: num tokens/patches
        # d_feature: dim of each token/patch prior to embedding. Accepts integers for discrete feature sets.
        # d_model: dim of vector representation of each token
        tok_emb = self.token_embedding(tokens)  # (b, n, d_model)
        b = tok_emb.shape[0]
        
        if isinstance(self.classifier, SingleTokenClassifier):
            # Add a class token to the beginning of each sequence
            cls_emb = self.class_token.expand(b, 1, -1)  # (b, 1, d_model)
            tok_emb = torch.cat((cls_emb, tok_emb), dim=1)  # (b, n+1, d_model)
        
        n = tok_emb.shape[1]
        position_indices = torch.arange(n, device=tokens.device)  # (n,)
        pos_emb = self.pos_embedding(position_indices)  # (n, d_model)
        X = tok_emb + pos_emb  # (b, n, d_model)
        attn_maps = []
        for block in self.blocks:
            X, attn_maps = block(X, attn_maps, return_attn_maps)  # (b, n, d_model)
        logits = self.classifier(X)
        if return_attn_maps:
            return logits, attn_maps  
        return logits