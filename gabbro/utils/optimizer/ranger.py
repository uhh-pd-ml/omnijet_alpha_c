"""Implementation of the Ranger optimizer.

Code taken from https://github.com/hqucms/weaver-core/blob/main/weaver/utils/nn/optimizer/ranger.py
"""

from .lookahead import Lookahead
from .radam import RAdam


def Ranger(
    params,
    lr=1e-3,  # lr
    betas=(0.95, 0.999),
    eps=1e-5,
    weight_decay=0,  # RAdam options
    alpha=0.5,
    k=6,  # LookAhead options
):
    radam = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return Lookahead(radam, alpha, k)
