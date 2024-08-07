import torch
import torch.nn.functional as F

from .data import decode, encode

# torch.set_default_device('cuda')

def generate_text(model, max_tokens, prompt, max_new_tokens=500):
    """Given a model and a prompt, generate text from the model."""
    # encode the prompt text to integers
    tokens = encode(prompt).unsqueeze(0)  # (b=1, n)
    for _ in range(max_new_tokens):
        # Retrieve the last max_tokens tokens from the sequence
        token_chunk = tokens[:, -max_tokens:]  # (b=1, n=max_tokens)
        token_chunk = token_chunk.to("cuda")
        logits = model(token_chunk)  # (b, n, d=vocab_size)
        # Each token's logit vector is trained to predict the next token, so extract
        # the last one.
        logits = logits[:, -1, :]  # (b, d)
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # (b, d)
        # sample from the distribution
        # We could alternatively used argmax to return the integer value of the highest probability in probs,
        # but multinomial does the same thing with some non-determinism. In particular, it will not always
        # return the index of the highest probability, it will sometimes return the index of a lower probability,
        # and it weights all of the choices by their relative possibilities. So if you have two synonyms with
        # very close probability, it will sometimes return one, and sometimes the other.
        next_token = torch.multinomial(probs, num_samples=1)  # (b, 1)
        tokens = torch.cat((tokens, next_token), dim=1)  # (b, n+1)
    # decode the tokens back into text
    text = decode(tokens.squeeze().tolist())
    return text

def get_ascii_image(img):
    """Given an image tensor, return a string representation of the image."""
    # img_tensor shape: (c, h, w)
    # Combine channels into a single grayscale channel
    img = torch.mean(img, dim=0)  # (h, w)
    # Normalize pixel values to a range between 0 and 1
    img = img / torch.max(img)
    # Define ASCII characters representing different shades
    ascii_chars = ['`', '.', ':', '-', '=', '+', '*', '#', '%', '@']
    # Convert pixel values to ASCII characters
    ascii_image = [[ascii_chars[int(pixel.item() * 9)] for pixel in row] for row in img]
    # Join rows of 2D list into single string
    ascii_image = [''.join(row) for row in ascii_image]
    ascii_image = '\n'.join(ascii_image)
    
    return ascii_image
