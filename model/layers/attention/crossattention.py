# TAKEN FROM :
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cross_vit.py

from collections import OrderedDict

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Callable, Union

from model.layers.torch_utils import get_activation_fn
from model.layers.module.module_utils import get_normalization_fn


__all__ = ['CrossViT',
           'ImageEmbedder',
           'MultiScaleEncoder',
           'CrossTransformer',
           'Transformer',
           'PreNorm',
           'Attention',
           'FeedForward']


# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Normalize(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor):
        return self.fn(x.transpose(1, 2)).transpose(1, 2)


# pre-layer norm
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, norm: str = 'ln'):
        super().__init__()
        self.norm = Normalize(get_normalization_fn(norm)[0](dim))
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs):

        return self.fn(self.norm(x), **kwargs)


# feedforward
class FeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 dropout: float = 0.,
                 output_dim: int = 0,
                 activation: str = 'gelu'):
        super().__init__()
        if output_dim == 0:
            output_dim = dim
        self.net = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(dim, hidden_dim)),
            (activation, get_activation_fn(activation)()),
            ('dropout1', nn.Dropout(dropout)),
            ('linear2', nn.Linear(hidden_dim, output_dim)),
            ('dropout2', nn.Dropout(dropout))])
        )
        if dim == output_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(dim, output_dim)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def res(self, x: torch.Tensor):
        return self.residual(x)


# attention
class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.,
                 v_proj: bool = True,
                 res_proj: bool = False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.v_proj = v_proj
        if v_proj:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        else:
            self.to_kv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(inner_dim, dim)),
            ('dropout', nn.Dropout(dropout))])
        )

        if res_proj:
            self.residual = nn.Linear(dim, dim)
        else:
            self.residual = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor = None,
                kv_include_self: bool = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        if self.v_proj:
            qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        else:
            qkv = (self.to_q(x), self.to_kv(context), context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

    def res(self, x: torch.Tensor):
        return self.residual(x)


# transformer encoder, for small and large patches
class Transformer(nn.Module):
    def __init__(self,
                 dim: Union[int, list],
                 depth: int,
                 heads: Union[int, list],
                 dim_head: Union[int, list],
                 mlp_dim: Union[int, list],
                 dropout: Union[float, list] = 0.,
                 mlp_out_dim: Union[int, list] = 0,
                 activation: str = 'gelu',
                 norm: str = 'ln',
                 global_norm: bool = True,
                 **kwargs):
        super().__init__()

        if isinstance(dim, int):
            dim = [dim] * depth
        if isinstance(heads, int):
            heads = [heads] * depth
        if isinstance(dim_head, int):
            dim_head = [dim_head] * depth
        if isinstance(mlp_dim, int):
            mlp_dim = [mlp_dim] * depth
        if isinstance(mlp_out_dim, int):
            mlp_out_dim = [mlp_out_dim] * depth
        if isinstance(dropout, float):
            dropout = [dropout] * depth

        assert isinstance(dim, list)
        assert isinstance(heads, list)
        assert isinstance(dim_head, list)
        assert isinstance(mlp_dim, list)
        assert isinstance(mlp_out_dim, list)
        assert isinstance(dropout, list)

        self.layers = nn.ModuleDict({})
        for i in range(depth):

            attn_kwargs = dict(dim=dim[i],
                               heads=heads[i],
                               dim_head=dim_head[i],
                               dropout=dropout[i],
                               v_proj=kwargs['v_proj'],
                               res_proj=kwargs['res_proj'])
            ffnn_kwargs = dict(dim=dim[i],
                               hidden_dim=mlp_dim[i],
                               dropout=dropout[i],
                               activation=activation,
                               output_dim=mlp_out_dim[i])

            self.layers.update(
                {
                    f'l{i+1}':
                    nn.ModuleDict(
                        OrderedDict({
                            'attn': PreNorm(dim=dim[i],
                                            fn=Attention(**attn_kwargs),
                                            norm=norm),
                            'ffn': PreNorm(dim=dim[i],
                                           fn=FeedForward(**ffnn_kwargs),
                                           norm=norm)
                        })
                    )
                }
            )

        if global_norm:
            _dim = dim[-1] if mlp_out_dim[-1] == 0 else mlp_out_dim[-1]
            self.norm = Normalize(get_normalization_fn(norm)[0](_dim))
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        attn_list = []
        for _, layer in self.layers.items():
            x1, attn = layer['attn'](x)
            x = x1 + layer['attn'].fn.res(x)
            x = layer['ffn'](x) + layer['ffn'].fn.res(x)
            attn_list.append(attn)
        return self.norm(x), attn_list


# projecting CLS tokens, in the case that small and large patch tokens
# have different dimensions
class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(
            dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(
            dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x, attn = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x, attn


# cross attention transformer
class CrossTransformer(nn.Module):
    def __init__(self, depth,
                 sm_dim, sm_heads, sm_dim_head, sm_dropout,
                 lg_dim, lg_heads, lg_dim_head, lg_dropout,
                 **kwargs):
        super().__init__()
        sm_attn_kwargs = dict(dim=lg_dim,
                              heads=sm_heads,
                              dim_head=sm_dim_head,
                              dropout=sm_dropout)
        lg_attn_kwargs = dict(dim=sm_dim,
                              heads=lg_heads,
                              dim_head=lg_dim_head,
                              dropout=lg_dropout)
        self.layers = nn.ModuleDict({})
        for i in range(depth):
            self.layers.update(
                {
                    f'l{i+1}':
                    nn.ModuleDict(
                        OrderedDict({
                            'sm_lg':
                            ProjectInOut(
                                dim_in=sm_dim,
                                dim_out=lg_dim,
                                fn=PreNorm(dim=lg_dim,
                                           fn=Attention(**sm_attn_kwargs))
                            ),
                            'lg_sm':
                            ProjectInOut(
                                dim_in=lg_dim,
                                dim_out=sm_dim,
                                fn=PreNorm(dim=sm_dim,
                                           fn=Attention(**lg_attn_kwargs))
                            )
                        })
                    )
                }
            )

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(
            lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        attn_list = []
        for _, layer in self.layers.items():
            sm_attend_lg, lg_attend_sm = layer['sm_lg'], layer['lg_sm']
            sm_cls_1, sm_attn = sm_attend_lg(
                sm_cls, context=lg_patch_tokens, kv_include_self=True)
            sm_cls += sm_cls_1
            lg_cls_1, lg_attn = lg_attend_sm(
                lg_cls, context=sm_patch_tokens, kv_include_self=True)
            lg_cls += lg_cls_1
            attn_list.append((sm_attn, lg_attn))

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)

        return sm_tokens, lg_tokens, attn_list


class CrossTransformerIdentity(nn.Identity):
    def __init__(self, *args, **kwargs):
        super(CrossTransformerIdentity, self).__init__()

    def forward(self, input1, input2):
        return input1, input2, []


# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head=64,
        dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                CrossTransformer(sm_dim=sm_dim,
                                 lg_dim=lg_dim,
                                 depth=cross_attn_depth,
                                 heads=cross_attn_heads,
                                 dim_head=cross_attn_dim_head,
                                 dropout=dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens


# patch-based image to token embedder
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout=0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'  # noqa
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)


# cross ViT class
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size=12,
        sm_enc_depth=1,
        sm_enc_heads=8,
        sm_enc_mlp_dim=2048,
        sm_enc_dim_head=64,
        lg_patch_size=16,
        lg_enc_depth=4,
        lg_enc_heads=8,
        lg_enc_mlp_dim=2048,
        lg_enc_dim_head=64,
        cross_attn_depth=2,
        cross_attn_heads=8,
        cross_attn_dim_head=64,
        depth=3,
        dropout=0.1,
        emb_dropout=0.1,
        sm_enc_mlp_out_dim=0,
        lg_enc_mlp_out_dim=0,
        activation='gelu',
        norm='ln'
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim=sm_dim,
                                               image_size=image_size,
                                               patch_size=sm_patch_size,
                                               dropout=emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim=lg_dim,
                                               image_size=image_size,
                                               patch_size=lg_patch_size,
                                               dropout=emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(
                depth=sm_enc_depth,
                heads=sm_enc_heads,
                mlp_dim=sm_enc_mlp_dim,
                dim_head=sm_enc_dim_head,
                activation=activation,
                norm=norm,
                mlp_out_dim=sm_enc_mlp_out_dim
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth,
                heads=lg_enc_heads,
                mlp_dim=lg_enc_mlp_dim,
                dim_head=lg_enc_dim_head,
                activation=activation,
                norm=norm,
                mlp_out_dim=lg_enc_mlp_out_dim
            ),
            dropout=dropout
        )

        self.sm_mlp_head = nn.Sequential(
            Normalize(get_normalization_fn(norm)[0](sm_dim)),
            nn.Linear(sm_dim, num_classes)
        )
        self.lg_mlp_head = nn.Sequential(
            Normalize(get_normalization_fn(norm)[0](lg_dim)),
            nn.Linear(lg_dim, num_classes)
        )

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + lg_logits


if __name__ == '__main__':
    m = Transformer(8, 2, 4, 4, 16, 0, 32)
    m(torch.rand(2, 5, 8))
