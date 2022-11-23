from math import pi
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


def heaviside(x: Tensor) -> Tensor:
    return x.ge(0).float()


class BaseSpike(torch.autograd.Function):
    """Base spiking function.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class SuperSpike(BaseSpike):
    """Spike function with SuperSpike surrogate gradient from
    "SuperSpike: Supervised Learning in Multilayer
    Spiking Neural Networks", Zenke et al. 2018.

    Design choices: (1) Height of 1 ("The Remarkable Robustness of
    Surrogate Gradient...", Zenke et al. 2021) (2) alpha scaled by 10
    ("Training Deep Spiking Neural Networks", Ledinauskas et al. 2020)
    """
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output  # .clone()
        sg = 1 / (1 + alpha * x.abs())**2
        return grad_input * sg, None


class MultiGaussSpike(BaseSpike):
    """Spike function with multi-Gaussian surrogate gradient from
    "Accurate and efficient time-domain classification", Yin et al. 2021.

    Design choices:
    - Hyperparameters determined through grid search (Yin et al. 2021)
    """
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output  # .clone()
        zero = torch.tensor(0.0)  # no need to specify device for 0-d tensors
        sg = (1.15 * gaussian(x, zero, alpha) -
              0.15 * gaussian(x, alpha, 6 * alpha) -
              0.15 * gaussian(x, -alpha, 6 * alpha))
        return grad_input * sg, None


def gaussian(x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Gaussian PDF with broadcasting.
    """
    return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (
        sigma * torch.sqrt(2 * torch.tensor(pi)))  # noqa


class TriangleSpike(BaseSpike):
    """Spike function with triangular surrogate gradient
    as in Bellec et al. 2020.
    """
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output  # .clone()
        sg = torch.nn.functional.relu(1 - alpha * x.abs())
        return grad_input * sg, None


class ArctanSpike(BaseSpike):
    """Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output  # .clone()
        sg = 1 / (1 + alpha * x * x)
        return grad_input * sg, None


class SigmoidSpike(BaseSpike):
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output  # .clone()
        sgax = (x * alpha).sigmoid_()
        sg = (1. - sgax) * sgax * alpha
        return grad_input * sg, None


def superspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return SuperSpike.apply(x - thresh, alpha)


def mgspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(0.5)):
    return MultiGaussSpike.apply(x - thresh, alpha)


def sigmoidspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    return SigmoidSpike.apply(x - thresh, alpha)


def trianglespike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    return TriangleSpike.apply(x - thresh, alpha)


def arctanspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return ArctanSpike.apply(x - thresh, alpha)


SURROGATE = {
    'sigmoid': sigmoidspike,
    'triangle': trianglespike,
    'arctan': arctanspike,
    'mg': mgspike,
    'super': superspike,
}


class PoissonEncoder(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """"""
        out_spike = torch.rand_like(x).le(x).to(x)
        return out_spike


def get_surrogate(name: str) -> Callable:
    assert name in SURROGATE
    surrogate = SURROGATE.get(name, None)
    if surrogate is None:
        raise ValueError(f"Unsupported surrogate function {name}")
    return surrogate


class IF(nn.Module):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float = 0.,
        alpha: float = 1.0,
        gamma: float = 0.,
        thresh_decay: float = 1.0,
        surrogate: str = 'sigmoid',
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.gamma = gamma
        self.thresh_decay = thresh_decay
        self.surrogate = get_surrogate(surrogate)
        self.register_buffer("alpha", torch.as_tensor(alpha,
                                                      dtype=torch.float))
        self.reset()

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold

    def forward(self, dv: Tensor) -> Tensor:
        """"""
        # 1. charge
        self.v += dv
        # 2. fire
        spike = self.surrogate(self.v, self.v_threshold, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        # 4. threhold updates
        self.v_th = self.gamma * spike + self.v_th * self.thresh_decay
        return spike


class LIF(nn.Module):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float = 0.,
        tau: float = 1.0,
        alpha: float = 1.0,
        gamma: float = 0.,
        thresh_decay: float = 1.0,
        surrogate: str = 'sigmoid',
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.gamma = gamma
        self.thresh_decay = thresh_decay
        self.surrogate = get_surrogate(surrogate)
        self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float))
        self.register_buffer("alpha", torch.as_tensor(alpha,
                                                      dtype=torch.float))
        self.reset()

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold

    def forward(self, dv: Tensor) -> Tensor:
        """"""
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_th, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        # 4. threhold updates
        self.v_th = self.gamma * spike + self.v_th * self.thresh_decay
        return spike


class PLIF(nn.Module):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float = 0.,
        tau: float = 1.0,
        alpha: float = 1.0,
        gamma: float = 0.,
        thresh_decay: float = 1.0,
        surrogate: str = 'sigmoid',
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.gamma = gamma
        self.thresh_decay = thresh_decay
        self.surrogate = get_surrogate(surrogate)
        self.register_parameter(
            "tau", nn.Parameter(torch.as_tensor(tau, dtype=torch.float)))
        self.register_buffer("alpha", torch.as_tensor(alpha,
                                                      dtype=torch.float))
        self.reset()

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold

    def forward(self, dv: Tensor) -> Tensor:
        """"""
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_th, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        # 4. threhold updates
        self.v_th = self.gamma * spike + self.v_th * self.thresh_decay
        return spike
