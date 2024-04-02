from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Function


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply


def gumbel_sigmoid_sub(x, tau=1e-12, training=True):
    if not training:
        return (x / tau).sigmoid()

    y = x.sigmoid()

    g1 = -torch.empty_like(x).exponential_().log()
    y_hard = ((x + g1 - g1) / tau).sigmoid()

    y_hard = (y_hard - y).detach() + y
    return y_hard


def divergence(query, target):
    if query.dim() <= 2 and query.shape[1] == 1:
        query = query.view(-1)
        target = target.view(-1)

    div_fct = nn.KLDivLoss(reduction="batchmean")
    q2t = div_fct(torch.log_softmax(query, dim=-1), torch.softmax(target, dim=-1))
    t2q = div_fct(torch.log_softmax(target, dim=-1), torch.softmax(query, dim=-1))
    return 1 / 2 * (q2t + t2q)


def prepare_for_asa_attack(attention_mask=None, num_choices=1, head_mask=None):
    """
    attention mask should be 4 dimension [num_layer, batch, seq_length, seq_length]
    """
    if isinstance(attention_mask, list):
        # [3,128,128]
        # attention_mask = [
        #     a.view((-1,) +a.shape[2:]) if a is not None else None
        #     for a in attention_mask
        # ]
        pass
    else:
        # [1,3,128]
        if attention_mask.dim() < 4:
            attention_mask = (
                attention_mask.view(-1, attention_mask.size(-1))
                if attention_mask is not None
                else None
            )
        elif attention_mask.dim() == 4:
            # [12,12,1,128] -> [12,1,12,128] -> [12,12,128]
            # [1,3,128,128]
            attention_mask = attention_mask.permute(0, 2, 1, 3)
            attention_mask = (
                attention_mask.reshape(
                    -1, attention_mask.size(-2), attention_mask.size(-1)
                )
                if attention_mask is not None
                else None
            )
            # [12,12,1,128]
            # attention_mask = attention_mask.permute(0,2,1,3)
        elif attention_mask.dim() == 6:  # training
            bs = attention_mask.size(0)
            # [16,12,4,12,128,128] -> [12,4*16,12,128,128]
            attention_mask = attention_mask.permute(
                1, 0, 2, 3, 4, 5
            )  # [12,16,4,12,128,128]
            attention_mask = (
                attention_mask.reshape(
                    -1,
                    num_choices * bs,
                    attention_mask.size(-3),
                    attention_mask.size(-2),
                    attention_mask.size(-1),
                )
                if attention_mask is not None
                else None
            )
            # [12,64,12,128,128]
        # elif attention_mask.dim() == 4:
        #     attention_mask = attention_mask.permute(0, 2, 1, 3)
        else:  # dim =4
            pass

    if head_mask is not None and num_choices > 1:
        if head_mask.shape[-1] == 1:
            head_mask = head_mask.repeat(1, num_choices, 1).view(
                head_mask.shape[0], -1, 1
            )
        elif head_mask.dim() == 2:
            pass
        else:
            head_mask = head_mask.repeat(1, num_choices, 1)
        # [num_hidden_layers x batch x num_heads[]

    return attention_mask, head_mask


def get_head_mask(
    head_mask: Optional[torch.Tensor],
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
    dtype=None,
) -> torch.Tensor:
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers, dtype)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers

    return head_mask


def _convert_head_mask_to_5d(head_mask, num_hidden_layers, dtype):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = (
            head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )  # We can specify head_mask for each layer
    elif head_mask.dim() == 3:  # added [num_hidden_layers x batch x num_heads]
        #     # [12,12,12]
        head_mask = (
            head_mask
            #         .unsqueeze(0)
            .unsqueeze(-1).unsqueeze(-1)
        )
    #     # [1,36,12,1,1]
    elif head_mask.dim() == 4:  # added
        # [12,3,12,12] -> [12,36,12] -> [12,36,12,1,1]
        # head_mask = head_mask.unsqueeze(-1)
        # head_mask = head_mask.unsqueeze(-2)
        head_mask = (
            head_mask.view(head_mask.shape[0], -1, head_mask.shape[-1])
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
    elif head_mask.dim() == 5:
        pass
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(
        dtype=dtype
    )  # switch to float if need + fp16 compatibility
    return head_mask


def get_extended_attention_mask(attention_mask, input_shape, device=None, dtype=None):
    # if dtype is None:
    #     dtype = self.dtype

    if attention_mask.dim() >= 4:  # added
        extended_attention_mask = attention_mask

    elif attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask
