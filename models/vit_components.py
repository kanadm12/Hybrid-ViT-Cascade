"""
ViT Components for Hybrid-ViT Cascade
Essential attention and modulation components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention mechanism"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_patches, embed_dim)
        Returns:
            output: (batch_size, num_patches, embed_dim)
        """
        batch_size, num_patches, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, num_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention for conditioning"""
    
    def __init__(self, embed_dim, context_dim, num_heads=8, dropout=0.1, store_attention=False):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.store_attention = store_attention
        
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(context_dim, embed_dim * 2, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Store attention weights for diagnostic purposes
        self.attention_weights = None
        
    def forward(self, x, context):
        """
        Args:
            x: (batch_size, num_patches, embed_dim) - query
            context: (batch_size, context_len, context_dim) - key/value
        Returns:
            output: (batch_size, num_patches, embed_dim)
        """
        batch_size, num_patches, embed_dim = x.shape
        context_len = context.shape[1]
        
        # Generate Q from x, K and V from context
        q = self.q(x).reshape(batch_size, num_patches, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, num_patches, head_dim)
        
        kv = self.kv(context).reshape(batch_size, context_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, num_heads, context_len, head_dim)
        k, v = kv[0], kv[1]
        
        # Cross-attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Store attention weights if requested (for diagnostics)
        if self.store_attention:
            self.attention_weights = attn.detach()
        
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization with modulation
    Conditions on timestep and X-ray embeddings
    """
    
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        
        self.linear = nn.Linear(cond_dim, embed_dim * 6, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x, cond):
        """
        Args:
            x: (batch_size, num_patches, embed_dim)
            cond: (batch_size, cond_dim) - combined time + xray embedding
        Returns:
            Modulation parameters for two layer norms (before self-attn and MLP)
        """
        # Generate modulation parameters
        params = self.linear(cond).unsqueeze(1)  # (B, 1, embed_dim * 6)
        
        # Split into scale and shift for self-attn norm, cross-attn norm, and MLP norm
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        
        return (shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps
    """
    
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, t):
        """
        Args:
            t: (batch_size,) timesteps
        Returns:
            embeddings: (batch_size, embed_dim)
        """
        device = t.device
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings
