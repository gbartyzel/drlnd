import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FactorizedNoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FactorizedNoisyLinear, self).__init__(in_features, out_features, bias)
        mu_init = 1.0 / np.sqrt(self.in_features)
        sigma_init = 0.5 / np.sqrt(self.in_features)

        self.weight.data.uniform_(-mu_init, mu_init)
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        if bias:
            self.bias.data.uniform_(-mu_init, mu_init)
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

        self.register_buffer('noise_in', torch.zeros(1, in_features))
        self.register_buffer('noise_out', torch.zeros(out_features, 1))

    def forward(self, input):
        noising = lambda x: torch.mul(torch.sign(x), torch.sqrt(torch.abs(x)))
        torch.randn(1, self.in_features, out=self.noise_in)
        torch.randn(self.out_features, 1, out=self.noise_out)

        bias = self.bias
        if bias is not None:
            bias_eps = noising(self.noise_out)
            bias = self.bias + self.bias_sigma * Variable(bias_eps.t())
        weight_eps = torch.mul(noising(self.noise_in), noising(self.noise_out))
        return F.linear(
            input, self.weight + self.weight_sigma * Variable(weight_eps), bias)


class IndependentNoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(IndependentNoisyLinear, self).__init__(in_features, out_features, bias)
        mu_init = np.sqrt(3.0 / self.in_features)
        sigma_init = 0.017

        self.weight.data.uniform_(-mu_init, mu_init)
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        if bias:
            self.bias.data.uniform_(-mu_init, mu_init)
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

        self.register_buffer('weight_eps', torch.zeros(out_features, in_features))
        self.register_buffer('bias_eps', torch.zeros(out_features))

    def forward(self, input):
        torch.randn(self.out_features, self.in_features, out=self.weight_eps)
        torch.randn(self.out_features, out=self.bias_eps)

        bias = self.bias
        if bias is not None:
            bias = self.bias + self.bias_sigma * Variable(self.bias_eps)
        return F.linear(
            input, self.weight + self.weight_sigma * Variable(self.weight_eps), bias)
