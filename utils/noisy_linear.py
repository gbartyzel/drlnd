import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias)
        mu_init = 1.0 / np.sqrt(self.in_features)
        sigma_init = 0.5 / np.sqrt(self.in_features)
        
        self.weight.data.uniform_(-mu_init, mu_init)
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        if bias:
            self.bias.data.uniform_(-mu_init, mu_init)
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
        
        self.register_buffer('weight_eps', torch.zeros(out_features, in_features, requires_grad=True))
        self.register_buffer('bias_eps', torch.zeros(out_features, requires_grad=True))
 
    def forward(self, input):
        self.sample_noise()
        bias = self.bias
        if bias is not None:
            bias = self.bias + self.bias_sigma * self.bias_eps.t().cuda()
        return F.linear(
            input, self.weight + self.weight_sigma * self.weight_eps.cuda(), bias)
    
    def sample_noise(self):
        noise_i = torch.randn(1, self.in_features)
        noise_j = torch.randn(self.out_features, 1)
        noising = lambda x: torch.mul(torch.sign(x), torch.sqrt(torch.abs(x)))
        self.weight_eps = torch.mul(noising(noise_i), noising(noise_j))
        self.bias_eps = noising(noise_j)
