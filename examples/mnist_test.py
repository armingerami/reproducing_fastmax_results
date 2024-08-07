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

# Experiment parameters
attn_type = "Attention"
fastmax = False
use_flash = False
normalize = None
p = None

# attn_type = "FastAttention, mean-centered, p=2"
# fastmax = True
# use_flash = False
# normalize = True
# p = 2

# attn_type = "FastAttention, zero-centered, p=2"
# fastmax = True
# use_flash = False
# normalize = False
# p = 2

# attn_type = "FastAttention, mean-centered, p=1"
# fastmax = True
# use_flash = False
# normalize = True
# p = 1

# attn_type = "FlashAttention"
# fastmax = False
# use_flash = True
# normalize = None
# p = None

# Dataset parameters
dataset_name = 'mnist'
n_classes = 10
n_channels = 1
patch_shape = (4, 4)
n_patches_per_image = 49
# Based on 5x5 patches and 28x28 images, we can fit 25 patches per image.
# The core transformer is agnostic of this number, but it learns a conv embedding based on the patch size.
# So the patch shape at training must remain the same at inference, but different size images or different
# conv strides can be used at inference time. Model is O(n^2) with this number.
embedding = "patch_conv" # Learnable conv kernels that map patch pixels to d_model vectors.
classifier = "cls_token" # Apply a linear classifier on a single token that holds a representation of the whole sequence.

# Model hyperparameters
max_features = n_patches_per_image 
d_model = 96
d_mlp = d_model * 2
heads_per_block = 4
num_blocks = 6
dropout_rate = 0.0

# Training parameters
lr = 5e-4 # learning rate
batch_size = 128
epochs = 10
loss_fn = F.cross_entropy
optim = torch.optim.AdamW

name = ", ".join([dataset_name, attn_type, f"N={n_patches_per_image}", f"D={d_model}"])

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
        normalize=normalize,
        p=p
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
        name=name,
    )

    # Evaluate
    model.load_state_dict(torch.load(weights_file))
    img = test_loader.dataset[0][0]
    pred = model(img.unsqueeze(0)).argmax()
    print("Test image:")
    print(get_ascii_image(img))
    print(f"Prediction: {pred}")
