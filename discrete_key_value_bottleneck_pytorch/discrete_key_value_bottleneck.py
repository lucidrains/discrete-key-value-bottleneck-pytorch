import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce

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
        dim_embed = None,
        num_memory_codebooks = 1,
        encoder = None,
        dim_memory = None,
        average_pool_memories = True,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        dim_embed = default(dim_embed, dim)
        self.dim_embed = dim_embed

        self.vq = VectorQuantize(
            dim = dim * num_memory_codebooks,
            codebook_size = num_memories,
            heads = num_memory_codebooks,
            separate_codebook_per_head = True,
            **kwargs
        )

        dim_memory = default(dim_memory, dim)
        self.values = nn.Parameter(torch.randn(num_memory_codebooks, num_memories, dim_memory))

        rand_proj = torch.empty(num_memory_codebooks, dim_embed, dim)
        nn.init.xavier_normal_(rand_proj)

        self.register_buffer('rand_proj', rand_proj)
        self.average_pool_memories = average_pool_memories

    def forward(
        self,
        x,
        return_intermediates = False,
        average_pool_memories = None,
        **kwargs
    ):
        average_pool_memories = default(average_pool_memories, self.average_pool_memories)

        if exists(self.encoder):
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, **kwargs)
                x.detach_()

        assert x.shape[-1] == self.dim_embed, f'encoding has a dimension of {x.shape[-1]} but dim_embed (defaults to dim) is set to {self.dim_embed} on init'

        x = einsum('b n d, c d e -> b n c e', x, self.rand_proj)
        x = rearrange(x, 'b n c e -> b n (c e)')

        vq_out = self.vq(x)

        quantized, memory_indices, commit_loss = vq_out

        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, '... -> ... 1')

        memory_indices = rearrange(memory_indices, 'b n h -> b h n')

        values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
        memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])

        memories = values.gather(2, memory_indices)

        if average_pool_memories:
            memories = reduce(memories, 'b h n d -> b n d', 'mean')

        if return_intermediates:
            return memories, vq_out

        return memories
