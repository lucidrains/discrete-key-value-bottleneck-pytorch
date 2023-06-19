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
        random_project_embed = True,  # in update to paper, they do a random projection of embedding
        embed_dim = None,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        embed_dim = default(embed_dim, dim)

        assert (embed_dim % num_memory_codebooks) == 0, f'embedding dimension {embed_dim} must be divisible by number of codes {num_memory_codebooks}'

        self.vq = VectorQuantize(
            dim = embed_dim,
            codebook_size = num_memories,
            heads = num_memory_codebooks,
            separate_codebook_per_head = True,
            **kwargs
        )

        dim_memory = default(dim_memory, embed_dim // num_memory_codebooks)
        self.values = nn.Parameter(torch.randn(num_memory_codebooks, num_memories, dim_memory))

        self.random_project_embed = random_project_embed

        if not random_project_embed:
            return

        rand_proj = torch.empty(dim, embed_dim)
        nn.init.xavier_normal_(rand_proj)

        self.register_buffer('rand_proj', rand_proj)

    def forward(
        self,
        x,
        return_intermediates = False,
        **kwargs
    ):

        if exists(self.encoder):
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, **kwargs)
                x.detach_()

        if self.random_project_embed:
            x = einsum('b n d, d e -> b n e', x, self.rand_proj)

        vq_out = self.vq(x)

        quantized, memory_indices, commit_loss = vq_out

        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, '... -> ... 1')

        memory_indices = rearrange(memory_indices, 'b n h -> b h n')

        values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
        memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])

        memories = values.gather(2, memory_indices)

        flattened_memories = rearrange(memories, 'b h n d -> b n (h d)')

        if return_intermediates:
            return flattened_memories, vq_out

        return flattened_memories
