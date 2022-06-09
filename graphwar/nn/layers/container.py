import inspect

import torch.nn as nn


class Sequential(nn.Sequential):
    """A modified :class:`torch.nn.Sequential` which can accept multiple inputs.

    Parameters
    ----------
    loc : int, optional
        the location of feature input :obj:`x`, by default 0    

    Example
    -------
    >>> import torch
    >>> from graphwar.nn.layers import Sequential, GCNConv

    >>> edge_index = torch.LongTensor([[1, 2], [3,4]]) # size [2, M]
    >>> x = torch.randn(5, 20)

    >>> conv1 = GCNConv(20, 50)
    >>> conv2 = GCNConv(50, 5)
    >>> dropout1 = torch.nn.Dropout(0.5)
    >>> dropout2 = torch.nn.Dropout(0.6)


    >>> # Case 1: standard usage
    >>> sequential = Sequential(dropout1, conv1, dropout2, conv2)
    >>> sequential(x, edge_index)
    tensor([[ 0.6738, -0.9032, -0.9628,  0.0670,  0.0252],
        [ 0.4909, -1.2430, -0.6029,  0.0510,  0.2107],
        [ 0.6338, -0.2760, -0.9112, -0.3197,  0.2689],
        [ 0.4909, -1.2430, -0.6029,  0.0510,  0.2107],
        [ 0.3876, -0.6385, -0.5521, -0.2753,  0.6713]], grad_fn=<AddBackward0>)

    >>> # which is equivalent to:
    >>> h1 = dropout1(x)
    >>> h2 = conv1(h1, edge_index)
    >>> h3 = dropout2(h2)
    >>> h4 = conv2(h3, edge_index)

    >>> # Case 2: with keyword argument
    >>> sequential(x, edge_index, edge_weight=torch.ones(20))
    tensor([[ 0.6738, -0.9032, -0.9628,  0.0670,  0.0252],
        [ 0.4909, -1.2430, -0.6029,  0.0510,  0.2107],
        [ 0.6338, -0.2760, -0.9112, -0.3197,  0.2689],
        [ 0.4909, -1.2430, -0.6029,  0.0510,  0.2107],
        [ 0.3876, -0.6385, -0.5521, -0.2753,  0.6713]], grad_fn=<AddBackward0>)    

    >>> # which is equivalent to:
    >>> h1 = dropout1(x)
    >>> h2 = conv1(x, edge_index, edge_weight=torch.ones(20))
    >>> h3 = dropout2(h2)
    >>> h4 = conv2(x, edge_index, edge_weight=torch.ones(20))  

    Note
    ----
    * The argument :obj:`loc` must be specified as the location of feature input :obj:`x`,  which would walk through the whole layers.
    * The usage of keyword argument must be matched with that of the layers with more than one arguments required.
    """

    def __init__(self, *args, loc: int = 0):
        super().__init__(*args)
        self.loc = loc
        para_required = []
        for module in self:
            assert hasattr(module, "forward"), module
            para_required.append(inspect.signature(module.forward).parameters)
        self._para_required = para_required

    def forward(self, *inputs, **kwargs):
        loc = self.loc
        assert loc <= len(inputs)
        output = inputs[loc]

        for module, para_required in zip(self, self._para_required):
            if len(para_required) == 1:
                input = inputs[loc]
                if isinstance(input, tuple):
                    output = tuple(module(_input) for _input in input)
                else:
                    output = module(input)
            else:
                output = module(*inputs, **kwargs)
            inputs = inputs[:loc] + (output,) + inputs[loc + 1:]
        return output

    def reset_parameters(self):
        for layer in self:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
