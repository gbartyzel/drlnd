import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = input_features
        self.out_features = output_features

        mu_val = 1 / np.sqrt(input_features)
        sigma_val = 0.5 / np.sqrt(input_features)

        self.w_mu = nn.Parameter(torch.Tensor(output_features, input_features))
        self.w_sigma = nn.Parameter(
            sigma_val * torch.ones(output_features, input_features))
        self.w_mu.data.uniform_(-mu_val, mu_val)

        self.b_mu = nn.Parameter(torch.Tensor(output_features))
        self.b_sigma = nn.Parameter(sigma_val * torch.ones(output_features))
        self.b_mu.data.uniform_(-mu_val, mu_val)

    def forward(self, input):
        w_eps, b_eps = self._create_noise()
        weight = self.w_mu + self.w_sigma * w_eps
        bias = self.b_mu + self.b_sigma * b_eps
        return F.linear(input, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def _create_noise(self):
        noise_i = torch.normal(mean=torch.zeros(1, self.in_features), std=1.0)
        noise_j = torch.normal(mean=torch.zeros(self.out_features, 1), std=1.0)
        w_eps = self._noising(noise_i) * self._noising(noise_j)
        b_eps = torch.squeeze(self._noising(noise_j))
        return w_eps, b_eps

    @staticmethod
    def _noising(x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

