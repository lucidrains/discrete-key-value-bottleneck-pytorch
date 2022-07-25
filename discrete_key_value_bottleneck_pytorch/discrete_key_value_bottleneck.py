import torch
from torch import nn, einsum
from einops import rearrange, repeat

from vector_quantize_pytorch import VectorQuantize

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# main class

class DiscreteKeyValueBottleneck(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_memories,
        dim_memory = None,
        **kwargs
    ):
        super().__init__()

        self.vq = VectorQuantize(
            dim = dim,
            codebook_size = num_memories,
            **kwargs
        )

        dim_memory = default(dim_memory, dim)
        self.values = nn.Parameter(torch.randn(num_memories, dim_memory))

    def forward(self, x):
        _, memory_indices, _ = self.vq(x)

        memories = self.values[memory_indices]
        return memories
