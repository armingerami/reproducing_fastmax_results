import glob
import os
import sys

import torch
import torch.nn.functional as F

 # Manually add the root project folder so python knows where to look for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../fmm-attention')))
from transformer.data import get_data
from transformer.model import Transformer
from transformer.train import train
from transformer.eval import generate_text, get_ascii_image
2

# Dataset parameters
dataset_name = 'mnist'
n_classes = 10
n_channels = 1
patch_shape_dim = 5
patch_shape = (patch_shape_dim, patch_shape_dim)
n_patches_per_image = int(28*28/(patch_shape_dim*patch_shape_dim))
# Based on 5x5 patches and 28x28 images, we can fit 25 patches per image.
# The core transformer is agnostic of this number, but it learns a conv embedding based on the patch size.
# So the patch shape at training must remain the same at inference, but different size images or different
# conv strides can be used at inference time. Model is O(n^2) with this number.
embedding = "patch_conv" # Learnable conv kernels that map patch pixels to d_model vectors.
classifier = "cls_token" # Apply a linear classifier on a single token that holds a representation of the whole sequence.

# Model hyperparameters
# Small model:
max_features = n_patches_per_image # Used for defining pos_encoding. Must be at least n_patches, but could be more.
d_model = 6
d_mlp = d_model * 2
heads_per_block = 3
num_blocks = 2
dropout_rate = 0.0
# # Large model:
# max_features = n_patches_per_image 
# d_model = 96
# d_mlp = d_model * 2
# heads_per_block = 4
# num_blocks = 6
# dropout_rate = 0.0

# Training parameters
lr = 1e-2 # learning rate
batch_size = 128
epochs = 1
loss_fn = F.cross_entropy
optim = torch.optim.AdamW


# Experiment parameters
attn_type = "Attention"
fastmax = False
use_flash = False
normalize = None
p = None

# attn_type = "FlashAttention"
# fastmax = False
# use_flash = True
# normalize = None
# p = None

# attn_type = "FastAttention, mean-centered, p=2"
# fastmax = True
# use_flash = False
# normalize = True
# p = 2
# lr = lr * 10

name = ", ".join([dataset_name, attn_type])


if __name__ == "__main__":
    # Create dataloaders
    train_loader, test_loader = get_data(dataset_name, batch_size)

    # Create model
    model = Transformer(
        out_dim=n_classes,
        max_tokens=max_features,
        d_model=d_model,
        d_mlp=d_mlp,
        heads_per_block=heads_per_block,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        use_masking=False,
        embedding=embedding,
        classifier=classifier,
        patch_shape=patch_shape,
        patch_channels=n_channels,
        fastmax=fastmax,
        use_flash=use_flash,
    )

    # Train
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

    # Evaluate
    model.load_state_dict(torch.load(weights_file))
    img = test_loader.dataset[0][0]
    logits, attn_map_list = model(img.unsqueeze(0), return_attn_maps=True)
    pred = logits.argmax()
    # attn_map_list shape: [(b, h, n, n), ..., times NumBlocks]
    first_attn_map = attn_map_list[0][0, 0]
    print("First layer, first batch, first head, attention map:")
    print(get_ascii_image(first_attn_map.unsqueeze(0)))

    print("Test image:")
    print(get_ascii_image(img))
    print(f"Prediction: {pred}")
