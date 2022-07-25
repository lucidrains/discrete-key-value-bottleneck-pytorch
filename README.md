<img src="./discrete-key-value.png" width="450px"></img>

## Discrete Key / Value Bottleneck - Pytorch

Implementation of <a href="https://arxiv.org/abs/2207.11240">Discrete Key / Value Bottleneck</a>, in Pytorch.

## Install

```bash
$ pip install discrete-key-value-bottleneck-pytorch
```

## Usage

```python
import torch
from discrete_key_value_bottleneck_pytorch import DiscreteKeyValueBottleneck

key_value_bottleneck = DiscreteKeyValueBottleneck(
    dim = 512,             # input dimension
    num_memories = 256,    # number of memories
    dim_memory = 2048,     # dimension of the output memories
    decay = 0.9,           # the exponential moving average decay, lower means the keys will change faster
)

embeds = torch.randn(1, 1024, 512)  # from pretrained encoder

memories = key_value_bottleneck(embeds)

memories.shape # (1, 1024, 2048)  # (batch, seq, memory / values dimension)

# now you can use the memories for the downstream decoder
```

You can also pass the pretrained encoder to the bottleneck and it will automatically invoke it. Example with `vit-pytorch` library

```bash
$ pip install vit-pytorch
```

then

```python
```bash
$ pip install vit-pytorch>=0.35.8
```

Then

```python
import torch

# import vision transformer

from vit_pytorch import SimpleViT
from vit_pytorch.extractor import Extractor

vit = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

# train vit, or load pretrained

vit = Extractor(vit, return_embeddings_only = True)

# then

key_value_bottleneck = DiscreteKeyValueBottleneck(
    encoder = vit,         # pass the frozen encoder into the bottleneck
    dim = 512,             # input dimension
    num_memories = 256,    # number of memories
    dim_memory = 2048,     # dimension of the output memories
    decay = 0.9,           # the exponential moving average decay, lower means the keys will change faster
)

images = torch.randn(1, 3, 256, 256)  # from pretrained encoder

memories = key_value_bottleneck(embeds) # (1, 64, 2048)   # (64 patches)
```

## Citations

```bibtex
@inproceedings{Trauble2022DiscreteKB,
    title   = {Discrete Key-Value Bottleneck},
    author  = {Frederik Trauble and Anirudh Goyal and Nasim Rahaman and Michael Curtis Mozer and Kenji Kawaguchi and Yoshua Bengio and Bernhard Scholkopf},
    year    = {2022}
}
```
