import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Noisy Networks for Exploration (Fortunato et al., 2018).
    Factorized Gaussian noise (Rainbow-style).

    Usage:
        layer = NoisyLinear(in_features, out_features, sigma0=0.5)
        y = layer(x)  # in training mode uses noise; in eval mode deterministic
        layer.reset_noise()  # optionally resample noise (otherwise resampled each forward in train mode)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.sigma0 = float(sigma0)

        # Learnable parameters: mean and stddev
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_sigma", None)

        # Non-learnable noise buffers (factorized)
        self.register_buffer("eps_in", torch.empty(in_features))
        self.register_buffer("eps_out", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        # As in the paper: mu ~ U(-1/sqrt(p), 1/sqrt(p)), sigma = sigma0 / sqrt(p)
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        if self.use_bias:
            self.bias_mu.data.uniform_(-bound, bound)
            self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        # f(eps) = sign(eps) * sqrt(|eps|)
        return x.sign() * x.abs().sqrt()

    @torch.no_grad()
    def reset_noise(self) -> None:
        self.eps_in.normal_()
        self.eps_out.normal_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Common pattern: resample each forward pass during training.
            # If you want manual control, comment this out and call reset_noise() externally.
            self.reset_noise()

            eps_in = self._f(self.eps_in)
            eps_out = self._f(self.eps_out)

            # Factorized noise: outer product gives weight noise
            weight_eps = eps_out.unsqueeze(1) * eps_in.unsqueeze(0)
            weight = self.weight_mu + self.weight_sigma * weight_eps

            if self.use_bias:
                bias = self.bias_mu + self.bias_sigma * eps_out
            else:
                bias = None
            return F.linear(x, weight, bias)

        # Eval: deterministic
        return F.linear(x, self.weight_mu, self.bias_mu if self.use_bias else None)
