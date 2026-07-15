import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

def precompute_rope_freqs(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rope(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    seq_len = xq_.shape[1]
    freqs_cis = freqs_cis[:seq_len].view(1, seq_len, 1, -1)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.head_dim = config['n_embd'] // self.n_heads
        
        self.wq = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        self.wk = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        self.wv = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        self.wo = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        
        self.resid_dropout = nn.Dropout(config['dropout'])

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch 2.0+ native scaled dot product attention (fast on CPU too)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.wo(y))
        return y

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config['n_embd'])
        self.mlp = SwiGLU(config['n_embd'], config['n_inner'])

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config['vocab_size']
        
        self.token_emb = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.drop = nn.Dropout(config['dropout'])
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])
        self.norm = RMSNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        # Weight tying: saves ~196k parameters, allowing for more layers
        self.token_emb.weight = self.lm_head.weight
        
        self.freqs_cis = precompute_rope_freqs(config['n_embd'] // config['n_heads'], config['block_size'] * 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        x = self.drop(x)
        
        freqs_cis = self.freqs_cis.to(x.device)

        for block in self.blocks:
            x = block(x, freqs_cis)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())