import inspect
import torch.nn as nn


class Sequential(nn.Sequential):
    """A modified `torch.nn.Sequential` which can take multiple inputs.

    Example
    -------
    >>> import torch
    >>> import dgl
    >>> from graphwar.nn import Sequential
    >>>
    >>> g = dgl.rand_graph(5, 20)
    >>> feat = torch.randn(5, 20)

    >>> conv1 = dgl.nn.GraphConv(20, 50)
    >>> conv2 = dgl.nn.GraphConv(50, 5)
    >>> dropout1 = torch.nn.Dropout(0.5)
    >>> dropout2 = torch.nn.Dropout(0.6)

    >>> sequential = Sequential(dropout1, conv1, dropout2, conv2, loc=1)
    >>> sequential(g, feat)
    tensor([[ 0.6738, -0.9032, -0.9628,  0.0670,  0.0252],
        [ 0.4909, -1.2430, -0.6029,  0.0510,  0.2107],
        [ 0.6338, -0.2760, -0.9112, -0.3197,  0.2689],
        [ 0.4909, -1.2430, -0.6029,  0.0510,  0.2107],
        [ 0.3876, -0.6385, -0.5521, -0.2753,  0.6713]], grad_fn=<AddBackward0>)


    >>> # which is equivalent to:
    >>> feat = dropout1(feat)
    >>> feat = conv1(g, feat)
    >>> feat = dropout2(feat)
    >>> feat = conv2(g, feat)

    Note
    ----
    The argument `loc` must be specified as the location of `feat`, 
    which walk through the whole layers.

    """

    def __init__(self, *args, loc=0):
        super().__init__(*args)
        self.loc = loc

    def forward(self, *inputs):
        loc = self.loc
        assert loc <= len(inputs)
        output = inputs[loc]

        for module in self:
            assert hasattr(module, 'forward')
            para_required = len(inspect.signature(module.forward).parameters)
            if para_required == 1:
                input = inputs[loc]
                if isinstance(input, tuple):
                    output = tuple(module(_input) for _input in input)
                else:
                    output = module(input)
            else:
                output = module(*inputs)
            inputs = inputs[:loc] + (output,) + inputs[loc + 1:]
        return output
