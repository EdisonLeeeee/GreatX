from typing import Any, Optional

import torch
import torch.nn as nn

act_dict = dict(relu="ReLU",
                relu6="ReLU6",
                sigmoid="Sigmoid",
                celu="CELU",
                elu="ELU",
                gelu="GELU",
                leakyrelu="LeakyReLU",
                prelu="PReLU",
                selu="SELU",
                silu="SiLU",
                softmax="Softmax",
                tanh="Tanh")


def get(act: Optional[str] = None, inplace: bool = False) -> nn.Module:
    """Get activation functions by input `string`

    Parameters
    ----------
    act : string or None
        the string to get activations, if None, return :class:`nn.Identity()` 
        that returns the input as output, by default None
    inplace : bool, optional
        the inplace argument in activation functions
        currently it is not work since not all the functions 
        take this argument, by default False

    Example
    -------
    >>> from graphwar.nn.layers import activations
    >>> activations.get('relu')
    ReLU()

    >>> activations.get(None)
    Identity()

    NOTE
    ----
    We currently do not support :obj:`inplace=True` since
    not all activation functions in PyTorch support argument :obj:`inplace=True`.

    Returns
    -------
    torch.nn.Module
        the activation function

    Raises
    ------
    ValueError
        unknown or invalid activation string.
    """
    if act is None:
        return nn.Identity()

    if isinstance(act, nn.Module):
        return act

    out = act_dict.get(act, None)
    if out:
        return getattr(nn, out)()
    else:
        raise ValueError(
            f"Unknown activation {act}. The allowed activation functions are {tuple(act_dict.keys())} or `None`.")
