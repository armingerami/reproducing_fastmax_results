import json
import os
from pathlib import Path
from collections import namedtuple
import re
from tqdm import tqdm
import requests
import torch
from torch.utils import data
from torchvision import datasets, transforms
from tokenizers import BertWordPieceTokenizer
tokenizerWiki = BertWordPieceTokenizer("data/wikitext103/bert-base-uncased-vocab.txt", lowercase=True)

def download_shakespear_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    unique_chars = sorted(list(set(text)))
    token_to_int = {ch: i for i, ch in enumerate(unique_chars)}
    os.makedirs("data/shakespeare", exist_ok=True)
    with open("data/shakespeare/input.txt", "w") as f:
        f.write(text)
    with open("data/shakespeare/vocab.json", "w") as f:
        json.dump(token_to_int, f)


def encode(text):
    """create a mapping from unique vocab tokens to integers"""
    # Get vocab data if not already downloaded
    if not os.path.exists("data/shakespeare/input.txt"):
        download_shakespear_data()
    with open("data/shakespeare/vocab.json", "r") as f:
        token_to_int = json.load(f)
    encoded_text = [token_to_int[char] for char in text]
    return torch.tensor(encoded_text, dtype=torch.long)


def decode(tokens):
    """create a mapping from integers back to unique vocab tokens"""
    # Get vocab data if not already downloaded
    if not os.path.exists("data/shakespeare/input.txt"):
        download_shakespear_data()
    with open("data/shakespeare/vocab.json", "r") as f:
        token_to_int = json.load(f)
    int_to_token = {v: k for k, v in token_to_int.items()}
    token_list = [int_to_token[token] for token in tokens]
    return "".join(token_list)


class ShakespeareDataset(data.Dataset):
    def __init__(self, tokens_per_chunk):
        super().__init__()
        # Download data if not already downloaded
        if not os.path.exists("data/shakespeare/input.txt"):
            download_shakespear_data()
        with open("data/shakespeare/input.txt", "r") as f:
            text = f.read()
        with open("data/shakespeare/vocab.json", "r") as f:
            token_to_int = json.load(f)
        
        self.data = encode(text)
        self.vocab_size = len(token_to_int)
        self.block_size = tokens_per_chunk

    def __len__(self):
        # A single example of this text set is a chunk of characters with length block_size
        return len(self.data) // self.block_size

    def __getitem__(self, i):
        # The corresponding label for each example is a chunk of characters of the same size,
        # but shifted one character to the right. Thus the task is to predict the next character
        # given all of the previous characters in a block. In this sense, the model learns to
        # generate predictions based on with varying amounts of preceding characters, ranging
        # from just a single character to the entire block.
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + self.block_size + 1]
        return x, y

class WikiDataset(data.Dataset):
    def __init__(self, tokens_per_chunk):
        super().__init__()
        print('a')
        text = Path('data/wikitext103/wiki.train.tokens').read_text()
        # with open("data/shakespeare/vocab.json", "r") as f:
            # token_to_int = txt.load(f)
        print('b')
        # txt = ""
        # for k in text.split("\n"):
        #     txt += "\n"+re.sub(r"[^a-zA-Z0-9]+", ' ', k)
        # txt = tokenizerWiki.encode(text[:50399960])
        print('c')
        # print(len(text))
        # self.data = encode(txt[:50399960])
        self.data = text
        print('d')
        # self.vocab_size = len(token_to_int)
        self.vocab_size = 30523
        self.block_size = tokens_per_chunk

    def __len__(self):
        # A single example of this text set is a chunk of characters with length block_size
        return len(self.data) // self.block_size

    def __getitem__(self, i):
        # The corresponding label for each example is a chunk of characters of the same size,
        # but shifted one character to the right. Thus the task is to predict the next character
        # given all of the previous characters in a block. In this sense, the model learns to
        # generate predictions based on with varying amounts of preceding characters, ranging
        # from just a single character to the entire block.
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + self.block_size + 1]
        return x, y


def get_data(dataset_name, batch_size, n_features=None, train_ratio=0.9):
    """
    Get dataloaders for a given dataset.
    Returns:
        train_loader: a DataLoader for the training set
        val_loader: a DataLoader for the validation set
        feature_dim: the dimensionality of each feature one example from the dataset
        n_classes: the number of classes in the dataset
    Feature_dim and n_classes are required for initializing the Transformer model.
    """
    if dataset_name == "shakespeare":
        n_features = n_features or 32
        dataset = ShakespeareDataset(n_features)
        n = len(dataset)
        train_size = int(train_ratio * n)
        test_size = n - train_size
        train_data, val_data = data.random_split(dataset, [train_size, test_size])
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_data, batch_size=batch_size)

    elif dataset_name == "wikitext103":
        n_features = n_features or 32
        dataset = WikiDataset(n_features)
        n = len(dataset)
        train_size = int(train_ratio * n)
        test_size = n - train_size
        train_data, val_data = data.random_split(dataset, [train_size, test_size])
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_data, batch_size=batch_size)


        
    elif dataset_name == "mnist":
        train_data = datasets.MNIST(root="data/mnist", train=True, download=True, transform=transforms.ToTensor())
        val_data = datasets.MNIST(root="data/mnist", train=False, download=True, transform=transforms.ToTensor())
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_data, batch_size=batch_size)
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Try 'shakespeare' or 'mnist'.")
    return train_loader, val_loader
