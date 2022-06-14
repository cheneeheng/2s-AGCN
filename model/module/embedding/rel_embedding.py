# Based on:
# https://arxiv.org/pdf/1904.09925.pdf
# https://arxiv.org/pdf/1803.02155.pdf
# https://arxiv.org/pdf/2101.11605.pdf

# FROM https://github.com/The-AI-Summer/self-attention-cv/blob/main/self_attention_cv/pos_embeddings/relative_embeddings_1D.py  # noqa
import torch
import torch.nn as nn
from einops import rearrange


__all__ = ['RelPosEmb1D']


# borrowed from
# https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py#L21  # noqa
# i will try to reimplement the function
# as soon as i understand how it works
# not clear to me how it works yet
def relative_to_absolute(q):
    """
    Converts the dimension that is specified from the axis
    from relative distances (with length 2*tokens-1)
    to absolute distance (length tokens)

    The final output can be viewed as taking (L) sections diagonally
    from top right to bottom left (through skewing)
    from a 2L matrix.
    The final row of the matrix is discarded because it takes element
    from the sides of the 2L matrix/start and end of L.

    Input: [bs, heads, length, 2*length - 1]
    Output: [bs, heads, length, length]
    """
    b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l, right pad
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    # zero pad l*2l to (l*2l) + (l-1), right pad
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    # reshape to b,h,l+1,2*l-1
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def rel_pos_emb_1d(q, rel_emb, shared_heads):
    """
    Same functionality as RelPosEmb1D

    Args:
        q: a 4d tensor of shape [batch, heads, tokens, dim]
        rel_emb: a 2D or 3D tensor
        of shape [ 2*tokens-1 , dim] or [ heads, 2*tokens-1 , dim]
    """
    if shared_heads:
        emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
    else:
        emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
    return relative_to_absolute(emb)


class RelPosEmb1D(nn.Module):
    def __init__(self, tokens, dim_head, heads=None):
        """
        Output: [batch head tokens tokens]

        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q

            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        scale = dim_head ** -0.5
        self.tokens = tokens
        self.shared_heads = heads if heads is not None else True
        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(
                torch.randn(2 * tokens - 1, dim_head) * scale)
        else:
            self.rel_pos_emb = nn.Parameter(
                torch.randn(heads, 2 * tokens - 1, dim_head) * scale)

    def forward(self, q):
        return rel_pos_emb_1d(q, self.rel_pos_emb, self.shared_heads)


# a = RelPosEmb1D(5, 8)
# b = a.forward(torch.ones((2, 3, 5, 8)))
# print(1)
