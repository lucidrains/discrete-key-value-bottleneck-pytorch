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
        num_memory_codebooks = 1,
        encoder = None,
        dim_memory = None,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        assert (dim % num_memory_codebooks) == 0, 'embedding dimension must be divisible by number of codes'

        self.vq = VectorQuantize(
            dim = dim,
            codebook_size = num_memories,
            heads = num_memory_codebooks,
            **kwargs
        )

        dim_memory = default(dim_memory, dim // num_memory_codebooks)
        self.values = nn.Parameter(torch.randn(num_memory_codebooks, num_memories, dim_memory))

    def forward(self, x, **kwargs):

        if exists(self.encoder):
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, **kwargs)
                x.detach_()

        _, memory_indices, _ = self.vq(x)

        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, '... -> ... 1')

        memory_indices = rearrange(memory_indices, 'b n h -> b h n')

        values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
        memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])

        memories = values.gather(2, memory_indices)

        return rearrange(memories, 'b h n d -> b n (h d)')
