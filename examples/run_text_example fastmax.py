import glob
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
import math
import torch
from torch import cuda
import fastmax_cuda
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import time
import argparse

 # Manually add the root project folder so python knows where to look for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../fmm-attention')))
from transformer.data import get_data
from transformer.eval import generate_text
from transformer.model import Transformer
from transformer.train import train

# torch.set_default_device('cuda')

# hyperparameters
#######################
# Small model (1 minute to train on CPU)

# Number of tokens to send through model in single forward pass. Transformer is agnostic of this number, so
# can accept different numbers at train and inference time. Used to define single "example" in dataloader,
# and to ensure positional encoding is at least this large. Model is O(n^2) with this number.
embedding = "discrete_set" # Learnable mapping from integers to d_model vectors
classifier = "per_token" # For text generation we want one class prediction per token. See note on label generation in data.py

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=None, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate.')
    parser.add_argument('--token_length', type=int, default=None, help='Toekn length.')
    parser.add_argument('--heads', type=int, default=None, help='Number of heads.')
    parser.add_argument('--dim', type=int, default=None, help='Dimension/number of channels for all of the heads (i.e. dimension per head  = dim/heads).')
    parser.add_argument('--max_iters', type=int, default=None, help='Number of iterations.')
    parser.add_argument('--batch', type=int, default=None, help='batch size.')
    args = parser.parse_args()
    return args

# hyperparameters
args = get_args()

# Model hyperparameters

# Modality-specific parameters (text)
n_tokens_per_inference = args.token_length
max_tokens = n_tokens_per_inference # Used for defining pos_encoding. Must be at least n_tokens, but could be more.
d_model = 256
d_mlp = d_model * 4
heads_per_block = 8
num_blocks = 8
dropout_rate = args.dropout
print("fastmax")
# Training parameters
lr = 1e-4
batch_size = args.batch
epochs = 2
loss_fn = F.cross_entropy
optim = torch.optim.AdamW

# # Large model (1 day to train on CPU)
# n_tokens = 256
# d_model = 384
# d_mlp = d_model * 4
# heads_per_block = 6
# num_blocks = 6
# dropout_rate = 0.2
# lr = 3e-4
# batch_size = 64
# epochs = 4
#######################

# FMM test
fastmax = True
# fastmax = True


if __name__ == "__main__":
    # Create dataloaders
    dataset_name = 'shakespeare'
    train_loader, test_loader = get_data(dataset_name, batch_size, n_tokens_per_inference)
    vocab_size = train_loader.dataset.dataset.vocab_size

    # Create model
    model = Transformer(
        out_dim=vocab_size,
        max_tokens=max_tokens,
        d_model=d_model,
        d_mlp=d_mlp,
        heads_per_block=heads_per_block,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        use_masking=True,
        embedding=embedding,
        classifier=classifier,
        vocab_size=vocab_size,
        fastmax=fastmax,
    )

    # Train
    start_time = time.time()
    model, weights_file = train(
        model,
        train_loader,
        test_loader,
        epochs=epochs,
        loss_fn=loss_fn,
        optim=optim,
        lr=lr,
        name=dataset_name,
    )
    print("Elapsed time:", time.time() - start_time, "seconds")
    print(torch.cuda.memory_allocated())
    # Evaluate
    model.load_state_dict(torch.load(weights_file))
    print(f"Generating text using model saved at {weights_file}:")
    print(generate_text(model, n_tokens_per_inference, "Romeo, what did you do with your shoe?"))
