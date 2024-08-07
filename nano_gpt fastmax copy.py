import math
import torch
from torch import cuda
import fastmax_cuda
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import time
import argparse

class FASTMultiHeadAttention_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v, drop_noise, rpe_matrix = None, mask = False, dropout = 0.0, normalize = False, temperature = 1.0, a0 = 1.0, a1 = 1.0, a2 = 0.5,lim = 1.0, p=2):
        b = 0
        if len(q.shape) == 4:
            b = q.shape[0]
            q = q.reshape((q.shape[0]*q.shape[1],q.shape[2],q.shape[3])) # (b,h,n,d) -> (b*h,n,d)
            k = k.reshape((k.shape[0]*k.shape[1],k.shape[2],k.shape[3])) # (b,h,n,d) -> (b*h,n,d)
            v = v.reshape((v.shape[0]*v.shape[1],v.shape[2],v.shape[3])) # (b,h,n,d) -> (b*h,n,d)
            drop_noise = drop_noise.reshape((drop_noise.shape[0]*drop_noise.shape[1],drop_noise.shape[2],drop_noise.shape[3])) # (b,h,n,d) -> (b*h,n,d)
        elif len(q.shape) != 3: print("q, k, and v should be either 3 or 4 dimensional tensors. If 3D: (b*h,n,d), if 4D: (b,h,n,d).")

        if rpe_matrix is None:
            print("Relative Positional Encoding must be given. Send a 2*n-1 by d matrix of all zeros if you don't want to use RPE.")

        q = q.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        k = k.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        v = v.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        drop_noise = drop_noise.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        # print(torch.cuda.memory_allocated())
        o = fastmax_cuda.forwardpass(q,k,v,drop_noise,rpe_matrix,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)
        # print(torch.cuda.memory_allocated())
        # print('a')
        ctx.save_for_backward(q,k,v,o)
        ctx.mask = mask
        ctx.p = p
        ctx.b = b
        ctx.t = temperature
        ctx.a0 = a0
        ctx.a1 = a1
        ctx.a2 = a2
        o = o[:,:,:q.shape[2]]
        o = o.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
        if b != 0: o = o.reshape((b,int(o.shape[0]/b),o.shape[1],o.shape[2])) # (b*h,n,d) -> (b,h,n,d)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q,k,v,o = ctx.saved_tensors
        mask = ctx.mask
        p = ctx.p
        b = ctx.b
        t = ctx.t
        a0 = ctx.a0
        a1 = ctx.a1
        a2 = ctx.a2

        if(b != 0): grad_output = grad_output.reshape((grad_output.shape[0]*grad_output.shape[1],grad_output.shape[2],grad_output.shape[3])).contiguous()
        grad_output = grad_output.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        gradq, gradk, gradv = fastmax_cuda.backwardpass(q,k,v,o,grad_output,mask,a0,a1,a2,p)

        gradq = gradq.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
        gradk = gradk.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
        gradv = gradv.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)

        if(b != 0):
          gradq = gradq.reshape((b,int(gradq.shape[0]/b),gradq.shape[1],gradq.shape[2])).contiguous()
          gradk = gradk.reshape((b,int(gradk.shape[0]/b),gradk.shape[1],gradk.shape[2])).contiguous()
          gradv = gradv.reshape((b,int(gradv.shape[0]/b),gradv.shape[1],gradv.shape[2])).contiguous()
        
        return gradq, gradk/t, gradv, None, None, None, None, None, None, None, None, None, None, None


def fastmax_function(q, k, v, mask=0, dropout_rate = 0.0, normalize=0, temperature=1, a0=1,a1=1,a2=0.5,lim=1,p=2, create_attn_matrix = 0):
    """
    Input: query, key, and value matrices (b, h, n, d)
        b: batch size
        h: number of heads
        n: number of tokens
        d: dimension per attention head (d = d_model / h)
    mask: boolean indicating whether to apply causal masking
    temperature: Hyperparameter to control the standard deviation of <q, k>; stdev(<q, k>) = 1/temperature
        Stdev of <q, k> is important in general with attention, but even more so when using a taylor
        expansion to approximate an exponential because the error increases with the stdev of <q, k>.
        In normal attention, stdev equates to the "temperature" of the softmax function, and with a
        taylor approximation, higher temperature also means we drift further from the true softmax.
        For positive inputs, this drifting error actually lowers the temperature, and for negative inputs
        it raises the temperature.
    Output: The result of Attention matrix * Value (b, h, n, d)
    """
    if create_attn_matrix == 0:
        if normalize == 1:
            temperature = 1
            # q = q - torch.mean(q,dim = 3).unsqueeze(-1)
            # k = k - torch.mean(k,dim = 3).unsqueeze(-1)
            qn = torch.linalg.norm(q, dim = 3)
            kn = torch.linalg.norm(k, dim = 3)
            q = lim*q/torch.linalg.norm(qn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
            k = lim*k/torch.linalg.norm(kn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
        else:
            temperature = temperature*math.sqrt(q.shape[3])
            temperature = 1
        temperatue2 = temperature*temperature

        # Prepare the quadratic terms with respect to k and q:
        if p == 2:
            # Prepare the quadratic terms with respect to k and q:
            k2 = k.unsqueeze(-1) @ k.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
            k2 = k2.flatten(-2)                     # (b, h, n, d*d)
            q2 = q.unsqueeze(-1) @ q.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
            q2 = q2.flatten(-2)                     # (b, h, n, d*d)
            drop_attn = torch.nn.Dropout(p=dropout_rate)
            k2 = drop_attn(k2)
            q2 = drop_attn(q2)

            if mask == 0:
                first_term = a0*torch.sum(v,-2)  # (b, h, d)

                second_term = a1*torch.matmul(k.swapaxes(-2,-1),v)/temperature  # (b, h, d, d)

                third_term = a2*torch.matmul(k2.swapaxes(-2,-1),v)/temperatue2  # (b, h, d^2, d)

                div1 = a0*torch.ones([k.shape[0],k.shape[1],1,1], device=k.device)*k.shape[2] # (b, h, 1, 1)
                div2 = a1*torch.sum(k,-2).unsqueeze(-1) # (b, h, d, 1)
                div3 = a2*torch.sum(k2,-2).unsqueeze(-1) # (b, h, d^2, 1)

                ans2 = torch.matmul(q,second_term)  # (b, h, n, d)
                ans3 = torch.matmul(q2,third_term)  # (b, h, n, d)
                div2 = torch.matmul(q,div2)/(temperature) # (b, h, n, 1)
                div3 = torch.matmul(q2,div3)/(temperatue2) # (b, h, n, 1)

                ans = ans2+ans3 # (b, h, n, d)
                ans = torch.add(ans.permute(2,3,1,0) ,first_term.permute(2,1,0)).permute(3,2,0,1) # (b, h, n, d)
                div = div2+div3 # (b, h, n, d)
                div = torch.add(div.permute(2,3,1,0) ,div1.permute(3,2,1,0)).permute(3,2,0,1) # (b, h, n, 1)
                ans = ans/div # (b, h, n, d)

            else:
                first = a0*torch.cumsum(v,2) # (b, h, n, d)
                second = a1*torch.einsum("bhij,bhijk -> bhik",[q, torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k,v]),2)])/temperature # (b, h, n, d)
                third = a2*torch.einsum("bhij,bhijk -> bhik",[q2,torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k2,v]),2)])/temperatue2 # (b, h, n, d)

                kcs = torch.cumsum(k,-2) # (b, h, n, d)
                k2cs = torch.cumsum(k2,-2) # (b, h, n, d^2)
                div1 = a0*torch.cumsum(torch.ones([q.shape[0],q.shape[1],q.shape[2]], device=k.device),2) # (b, h, 1)
                div2 = a1*torch.einsum("bhij,bhij -> bhi",[q,kcs])/temperature # (b, h, n)
                div3 = a2*torch.einsum("bhij,bhij -> bhi",[q2,k2cs])/temperatue2 # (b, h, n)
                div = (div1 + div2 + div3).unsqueeze(-1) # (b, h, n, 1)

                ans = first + second + third # (b, h, n, d)
                ans /= div # (b, h, n, d)
            
        # Taylor series with constant and linear terms:
        elif p == 1:
            drop_attn = torch.nn.Dropout(p=dropout_rate)
            k = drop_attn(k)
            q = drop_attn(q)
            if mask is None or not mask:
                first_term = a0*torch.sum(v,-2)  # (b, h, d)
                second_term = a1*torch.matmul(k.swapaxes(-2,-1),v)/temperature  # (b, h, d, d)

                div1 = a0*torch.ones([k.shape[0],k.shape[1],1,1], device=k.device)*k.shape[2] # (b, h, 1, 1)
                div2 = a1*torch.sum(k,-2).unsqueeze(-1) # (b, h, d, 1)

                ans2 = torch.matmul(q,second_term)  # (b, h, n, d)
                div2 = torch.matmul(q,div2)/(temperature) # (b, h, n, 1)

                ans = ans2 # (b, h, n, d)
                ans = torch.add(ans.permute(2,3,1,0) ,first_term.permute(2,1,0)).permute(3,2,0,1) # (b, h, n, d)
                div = div2 # (b, h, n, d)
                div = torch.add(div.permute(2,3,1,0) ,div1.permute(3,2,1,0)).permute(3,2,0,1) # (b, h, n, 1)
                ans = ans/div # (b, h, n, d)

            else:
                first = a0*torch.cumsum(v,2) # (b, h, n, d)
                second = a1*torch.einsum("bhij,bhijk -> bhik",[q, torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k,v]),2)])/temperature # (b, h, n, d)

                kcs = torch.cumsum(k,-2) # (b, h, n, d)
                div1 = a0*torch.cumsum(torch.ones([q.shape[0],q.shape[1],q.shape[2]], device=k.device),2) # (b, h, 1)
                div2 = a1*torch.einsum("bhij,bhij -> bhi",[q,kcs])/temperature # (b, h, n)
                div = (div1 + div2).unsqueeze(-1) # (b, h, n, 1)

                ans = first + second # (b, h, n, d)
                ans /= div # (b, h, n, d)
        
        else:
            raise ValueError(f"p must be 1 or 2, got: {p}")
        return ans

    else:
        # temperature = temperature*math.sqrt(q.shape[3])
        temperatue2 = temperature*temperature

        k2 = k.unsqueeze(-1) @ k.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
        k2 = k2.flatten(-2)                     # (b, h, n, d*d)
        q2 = q.unsqueeze(-1) @ q.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
        q2 = q2.flatten(-2)    
        attn = a0 + a1*torch.matmul(q, torch.swapaxes(k, -2, -1))/temperature + a2*torch.matmul(q2, torch.swapaxes(k2, -2, -1))/temperatue2
        if mask is not None:
            attn = torch.where(mask == 0, 0, attn)
        attn /= (torch.sum(attn, axis=3)).unsqueeze(-1)
        ans = torch.matmul(attn,v)
        return ans, attn

class FASTMultiHeadAttention(torch.nn.Module):
    def __init__(self, use_custom_gradient = True):
        super(FASTMultiHeadAttention, self).__init__()
        self.use_custom_gradient = use_custom_gradient

    def forward(self, q,k,v,drop_noise=None,rpe_matrix = None, mask = False, dropout = 0.0, normalize = False, temperature = 1.0, a0 = 1.0, a1 = 1.0, a2 = 0.5,lim = 1.0,p=2):
        if self.use_custom_gradient: return FASTMultiHeadAttention_Function.apply(q,k,v,drop_noise,rpe_matrix,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)
        # else: return fastmax_function(q,k,v,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)
        else: return 0

def rpe_matrix_creator(n, d, device, dtype, structured = False, is_zero = True):
    """
    Creates the relative positional encoding matrix
    Inputs: (assuming query is a (b,h,n,d) or (b*h,n,d) tensor)
      - n (int): number of tokens
      - d (int): dimesion/channel per head
      - data type: must be torch.float32. This input is used to make sure the datatype used by the attention head is torch.float32.
      - Structured (bool): if True, produces sin/cos based RPE, and randomized matrx otherwise.
    Output:
      - rpe: a (2*n-1,d) matrix.
    """
    if(dtype != torch.float32): print("The data type must be float32 in order for Fastmax to work")
    if(structured):
        pe_positive = torch.zeros(n, d,device=device,dtype=dtype)
        pe_negative = torch.zeros(n, d,device=device,dtype=dtype)
        position = torch.arange(0, n, device=device,dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, device=device,dtype=dtype) * -(math.log(10000.0) / d))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0])
        pe_negative = pe_negative[1:]
        rpe = torch.cat([pe_positive, pe_negative], dim=0)
    else: 
        if is_zero:
            rpe = torch.zeros(size=(2*n-1,d),device=device,dtype=dtype)
        else:
            rpe = torch.normal(0,1,size=(2*n-1,d),device=device,dtype=dtype)
    return rpe
fastmax_custom = FASTMultiHeadAttention()

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
batch_size = args.batch # how many independent sequences will we process in parallel?
block_size = args.token_length # what is the maximum context length for predictions?
max_iters = args.max_iters
eval_interval = 2000
learning_rate = args.lr
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = args.dim
n_head = 1
n_heads = args.heads
n_layer = 8
dropout_rate = args.dropout
dim_per_head = int(n_embd/n_heads)
p = 1
dropout = dropout_rate
# ------------

print("block_size = ", block_size)
print("dropout = ", dropout_rate)
print("n_embd = ", n_embd)
print("n_head = ", n_head)
print("learning_rate = ", learning_rate)
print("Fastmax")

torch.manual_seed(42)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('Tiny Shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # B,T,C = x.shape
        # k = self.key(x)   # (B,T,hs)
        # q = self.query(x) # (B,T,hs)
        # # compute attention scores ("affinities")
        # wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # wei = F.softmax(wei, dim=-1) # (B, T, T)
        # wei = self.dropout(wei)
        # # perform the weighted aggregation of the values
        # v = self.value(x) # (B,T,hs)
        # out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        B,T,C = x.shape
        q = self.query(x) # (B,T,hs)
        k = self.key(x)   # (B,T,hs)
        v = self.value(x) # (B,T,hs)
        q = q.reshape((batch_size, n_heads, block_size, dim_per_head))
        k = k.reshape((batch_size, n_heads, block_size, dim_per_head))
        v = v.reshape((batch_size, n_heads, block_size, dim_per_head))
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # print(q.device)
        mask = True
        normalize = True
        temperature = 1.0
        a0 = 1.0
        a1 = 1.0
        a2 = 0.5
        lim = 1.0
        rpe_matrix = rpe_matrix_creator(k.shape[-2],q.shape[-1],q.device,q.dtype,structured = False,is_zero = True).contiguous()
        drop_noise = torch.normal(0,1,size=(q.shape),dtype=q.dtype,device=q.device).contiguous()
        out = fastmax_custom(q,k,v,drop_noise,rpe_matrix,mask,dropout_rate,normalize,temperature,a0,a1,a2,lim,p)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_time = time.time()
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print("Elapsed time:", time.time() - start_time, "seconds")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(torch.cuda.memory_allocated())
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))