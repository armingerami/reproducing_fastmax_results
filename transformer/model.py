import math
import torch
from torch import nn
from torch.nn import functional as F
import fastmax_cuda
import os, sys

# torch.set_default_device('cuda')
 # Manually add the root project folder so python knows where to look for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../fmm-attention')))

from .modules import PerTokenClassifier, AvgPoolClassifier, SingleTokenClassifier, ConvEmbedding, FlattenEmbedding

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
                # V_hat = fastmax(Q, K, V, mask=self.use_masking, normalize=self.normalize, p=self.p)
                mask = True
                normalize = True
                temperature = 1.0
                a0 = 1.0
                a1 = 1.0
                a2 = 0.5
                lim = 1.0
                dropout_rate = 0.1
                p = 1
                rpe_matrix = rpe_matrix_creator(K.shape[-2],Q.shape[-1],Q.device,Q.dtype,structured = False,is_zero = True).contiguous()
                drop_noise = torch.normal(0,1,size=(Q.shape),dtype=Q.dtype,device=Q.device).contiguous()
                V_hat = fastmax_custom(Q,K,V,drop_noise,rpe_matrix,mask,dropout_rate,normalize,temperature,a0,a1,a2,lim,p)
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
        tokens = tokens.to("cuda")
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
