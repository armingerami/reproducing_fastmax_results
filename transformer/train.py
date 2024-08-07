import time
from datetime import timedelta

import lightning as L
import torch
from lightning.pytorch import callbacks, loggers
from torch.utils import data

from .model import PerTokenClassifier

def train_val_split(dataloader):
    """Split a dataloader into train and val dataloaders."""
    dataset = data.ConcatDataset([batch for batch in dataloader])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    train_loader = data.DataLoader(train_dataset, batch_size=dataloader.batch_size)
    val_loader = data.DataLoader(val_dataset, batch_size=dataloader.batch_size)
    return train_loader, val_loader


def save_statedict(ckpt_path):
    """Load the statedict from a PyTorch Lightning checkpoint."""
    checkpoint = torch.load(ckpt_path)
    lightning_statedict = checkpoint["state_dict"]
    statedict = {k.replace("model.", ""): v for k, v in lightning_statedict.items()}
    statedict_path = ckpt_path.replace(".ckpt", ".pth")
    torch.save(statedict, statedict_path)
    return statedict_path


class LightingWrapper(L.LightningModule):
    """Wraps a nn.Module with an associated loss function and optimizer."""

    def __init__(self, model, loss_fn, optim, lr, weight_decay=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, X):
        return self.model(X)
    
    def configure_optimizers(self):
        return self.optim(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        # Text modality:
        # x: (b, n) where each token is an integer in [0, n_classes) (vocab_size)
        # y: (b, n) There n labels for n tokens (the predicted next token). Each 
        #    successive label gets a longer preceeding series of tokens to use as its
        #    features due to masking in attention.
        # Image modality:
        # x: (b, n, d_feature) where each token is a flattened image patch
        # y: (b,) There is a single class label for each image.
        # The model handles both input shapes properly, and the train/val loops handle the output shappes properly
        #
        # PerTokenClassifier (text): (b, n, n_classes)
        # ClassTokenClassifier (img): (b, n_classes)

        x, y = batch
        logits = self.model(x)  # (b, n, n_classes) or (b, n_classes)
        if isinstance(self.model.classifier, PerTokenClassifier):
            # Reshape to handle text modality:
            logits = logits.view(-1, logits.shape[-1])  # (b, n, n_classes) -> (b*n, n_classes)
            y = y.view(-1)  # (b, n) -> (b*n)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Identical to training step
        x, y = batch
        logits = self.model(x)  # (b, n, n_classes) or (b, n_classes)
        if isinstance(self.model.classifier, PerTokenClassifier):
            # Reshape to handle text modality:
            logits = logits.view(-1, logits.shape[-1])  # (b, n, n_classes) -> (b*n, n_classes)
            y = y.view(-1)  # (b, n) -> (b*n)
        loss = self.loss_fn(logits, y)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    

def train(
    model,
    train_loader,
    val_loader=None,
    epochs=1,
    loss_fn=torch.nn.CrossEntropyLoss,
    optim=torch.optim.AdamW,
    lr=1e-3,
    name="",
    vals_per_epoch=1,
):
    """
    Given a model, dataset, loss function and optimizer, train the model.
    Uses Pytorch Lighting to handle logging, checkpointing, and GPU accelleration.
    """
    if val_loader is None:
        train_loader, val_loader = train_val_split(train_loader)

    wrapped_model = LightingWrapper(model, loss_fn, optim, lr)
    ckpt = callbacks.ModelCheckpoint(dirpath=f"logs/{name}")
    wandb_logger = loggers.WandbLogger(project="fmm-attention", log_model=False, name=name)
    trainer = L.Trainer(
        max_epochs=epochs,
        val_check_interval=1 / vals_per_epoch,
        accelerator="auto",
        callbacks=[ckpt],
        logger=wandb_logger,
    )

    # Run the training loop
    print(f"Running {name}...")
    start_time = time.time()
    trainer.fit(wrapped_model, train_loader, val_loader)
    end_time = time.time()
    elapsed_time = str(timedelta(seconds=end_time - start_time))
    print(f"Training time: {elapsed_time}")
    print(f"Finished {name}.")
    wandb_logger.experiment.finish()


    weights_file = save_statedict(ckpt.best_model_path)
    return wrapped_model.model, weights_file
