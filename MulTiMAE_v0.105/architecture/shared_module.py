import math
import torch
import numpy as np

from itertools import repeat
import collections.abc
from enum import Enum
from torch import _assert

class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, output_fmt=None, bias=True, strict_img_size=True, dynamic_img_pad=False):
        super().__init__()
        to_2tuple = _ntuple(2)
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def feat_ratio(self, as_scalar=True):
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size):
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input = [batch, seq_len, d_model]
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = torch.permute(x, (1,0,2))
        x = x + self.pe[:x.size(0)]
        x = torch.permute(x, (1,0,2))
        return self.dropout(x)

class MultiheadBlockAttention(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead

        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)

        self.output_linear = torch.nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, padding_mask):
        # Linear transformation
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split head
        batch_size, Q_seq_len, Q_num_modality, d_model = Q.shape
        batch_size, K_seq_len, K_num_modality, d_model = K.shape
        Q = Q.view(batch_size, Q_seq_len, Q_num_modality, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        K = K.view(batch_size, K_seq_len, K_num_modality, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)
        V = V.view(batch_size, K_seq_len, K_num_modality, self.nhead, d_model//self.nhead).permute(0,3,1,2,4)

        # Scaled dot product attention
        QK = Q @ K.permute(0,1,2,4,3)
        logits = QK / math.sqrt(d_model//self.nhead)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(-2).expand(-1, logits.shape[1], -1, logits.shape[-2], -1)
        logits += padding_mask
        attn_weight = torch.nn.functional.softmax(logits, dim=-1)
        assert torch.isnan(attn_weight).sum() == 0
        attn_output = attn_weight @ V

        # Concat head
        batch_size, nhead, seq_len, num_modality, _ = attn_output.shape
        attn_output = attn_output.permute(0,2,3,1,4).reshape(batch_size, seq_len, num_modality, -1)
        attn_output = self.output_linear(attn_output)

        
        return attn_output, attn_weight


def patchify(imgs, patch_size=16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def unpatchify(x, patch_size=16):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs
1==1