import torch
import torch.nn as nn
import numpy as np

class KANLayer(nn.Module):
            def __init__(self, input_dim, output_dim, degree, poly_type, alpha=None, beta=None):
                super(KANLayer, self).__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.degree = degree
                self.poly_type = poly_type
                self.alpha = alpha
                self.beta = beta

                self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
                nn.init.normal_(self.coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

                if poly_type == 'chebyshev':
                    self.register_buffer("arange", torch.arange(0, degree + 1, 1))

            def forward(self, x):
                x = x.view(-1, self.input_dim)
                x = torch.tanh(x)

                if self.poly_type == 'chebyshev':
                    x = x.view((-1, self.input_dim, 1)).expand(-1, -1, self.degree + 1)
                    x = x.acos()
                    x *= self.arange
                    x = x.cos()
                else:
                    poly = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
                    if self.poly_type == 'bessel':
                        if self.degree > 0:
                            poly[:, :, 1] = x + 1
                        for i in range(2, self.degree + 1):
                            poly[:, :, i] = (2 * i - 1) * x * poly[:, :, i - 1].clone() + poly[:, :, i - 2].clone()
                    elif self.poly_type == 'fibonacci':
                        poly[:, :, 0] = 0
                        if self.degree > 0:
                            poly[:, :, 1] = 1
                        for i in range(2, self.degree + 1):
                            poly[:, :, i] = x * poly[:, :, i - 1].clone() + poly[:, :, i - 2].clone()
                    elif self.poly_type == 'gegenbauer':
                        if self.degree > 0:
                            poly[:, :, 1] = 2 * self.alpha * x
                        for n in range(1, self.degree):
                            term1 = 2 * (n + self.alpha) * x * poly[:, :, n].clone()
                            term2 = (n + 2 * self.alpha - 1) * poly[:, :, n - 1].clone()
                            poly[:, :, n + 1] = (term1 - term2) / (n + 1)
                    elif self.poly_type == 'hermite':
                        if self.degree > 0:
                            poly[:, :, 1] = 2 * x
                        for i in range(2, self.degree + 1):
                            poly[:, :, i] = 2 * x * poly[:, :, i - 1].clone() - 2 * (i - 1) * poly[:, :, i - 2].clone()
                    elif self.poly_type == 'jacobi':
                        if self.degree > 0:
                            poly[:, :, 1] = (0.5 * (self.alpha - self.beta) + (self.alpha + self.beta + 2) * x / 2)
                        for n in range(2, self.degree + 1):
                            A_n = 2 * n * (n + self.alpha + self.beta) * (2 * n + self.alpha + self.beta - 2)
                            term1 = (2 * n + self.alpha + self.beta - 1) * (2 * n + self.alpha + self.beta) * \
                                    (2 * n + self.alpha + self.beta - 2) * x * poly[:, :, n-1].clone()
                            term2 = (2 * n + self.alpha + self.beta - 1) * (self.alpha ** 2 - self.beta ** 2) * poly[:, :, n-1].clone()
                            term3 = (n + self.alpha + self.beta - 1) * (n + self.alpha - 1) * (n + self.beta - 1) * \
                                    (2 * n + self.alpha + self.beta) * poly[:, :, n-2].clone()
                            poly[:, :, n] = (term1 - term2 - term3) / A_n
                    elif self.poly_type == 'laguerre':
                        poly[:, :, 0] = 1
                        if self.degree > 0:
                            poly[:, :, 1] = 1 + self.alpha - x
                        for k in range(2, self.degree + 1):
                            term1 = ((2 * (k-1) + 1 + self.alpha - x) * poly[:, :, k - 1].clone())
                            term2 = (k - 1 + self.alpha) * poly[:, :, k - 2].clone()
                            poly[:, :, k] = (term1 - term2) / k
                    elif self.poly_type == 'legendre':
                        poly[:, :, 0] = 1
                        if self.degree > 0:
                            poly[:, :, 1] = x
                        for n in range(2, self.degree + 1):
                            poly[:, :, n] = ((2 * (n-1) + 1) / n) * x * poly[:, :, n-1].clone() - ((n-1) / n) * poly[:, :, n-2].clone()
                    x = poly

                y = torch.einsum('bid,iod->bo', x, self.coeffs)
                return y.view(-1, self.output_dim)