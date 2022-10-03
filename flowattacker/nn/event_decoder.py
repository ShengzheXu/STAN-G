import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical
from .behavior_encoder import AttributeEmbedding

def softmax_sample(h, num_samples=1):
    probs = F.softmax(h, dim=1)
    sample = torch.multinomial(probs, num_samples)
    return sample #, dist


class EventDiscreteDecoder(nn.Module):

    def __init__(self, dim_in, 
                 disc_num_list: list,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 seq_decode: bool = False):
        super().__init__()

        self.seq_decode = seq_decode
        self.n_disc = len(disc_num_list)
        if seq_decode:
            self.seq_encoder = AttributeEmbedding(n_cont=0, n_disc_list=disc_num_list)
            self.mask_seq = 1 - torch.triu(torch.ones(self.n_disc, self.n_disc), diagonal=0)
            dim_in += self.seq_encoder.out_dim

        self.layers = nn.ModuleList([EventDiscreteDecoderLayer(dim_in, dim_out, num_layers, dropout) for dim_out in disc_num_list])
        

    def forward(self, h, seq_x=None, return_sample=False):
        seq_filling = False
        if seq_x is None:
            seq_x = torch.zeros(1, self.n_disc).to(h.device)
            seq_filling = True
        self.mask_seq = self.mask_seq.to(h.device)

        out = []
        if self.seq_decode is False:
            for mod in self.layers:
                out.append(mod(h))
        else:
            for idx, mod in enumerate(self.layers):
                masked_seq_x = seq_x * self.mask_seq[idx]

                seq_h = self.seq_encoder(masked_seq_x)
                p = mod(torch.cat([h, seq_h.squeeze(1)], 1))
                out.append(p)
                if seq_filling:
                    seq_x[:, idx] = softmax_sample(p).squeeze(1)
                
                # if self.training is False:
                #     print(idx, seq_x.shape, seq_x, self.mask_seq[idx], masked_seq_x)
                #     print('adding', idx, seq_x[:, idx])
                #     input()

        # out = torch.cat(out, dim=1)
        return (out, seq_x) if return_sample is True else out

    def loss(self, x, y):
        '''
        if self.seq_decode is True, the masked_seq_x should be inputed.
        '''
        pr = self.forward(x, y)

        y = y.long()
        loss = [torch.nn.CrossEntropyLoss()(p, y[:,i]) for i, p in enumerate(pr)]
        # print([lo.item() for lo in loss])
        return torch.stack(loss, dim=0).mean()
    
    def sample(self, x):
        if self.seq_decode:
            pr, result = self.forward(x, return_sample=True)
        else:
            pr = self.forward(x, return_sample=True)
            result = [softmax_sample(p) for p in pr]
            result = torch.cat(result, dim=1)
        return result

class EventDiscreteDecoderLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, 
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_dim, in_dim))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(in_dim, in_dim))
        self.lins.append(torch.nn.Linear(in_dim, out_dim))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = F.relu(lin(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class EventContinuousDecoder(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, 
                 n_components: int = 3,
                 seq_decode: int = None):
        super().__init__()

        if seq_decode is not None:
            self.seq_encoder = AttributeEmbedding(n_cont=0, n_disc_list=seq_decode)
            dim_in += self.seq_encoder.out_dim

        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y, seq_x=None):
        if seq_x is not None:
            seq_h = self.seq_encoder(seq_x)
            x = torch.cat([x, seq_h.squeeze(1)], dim=1)

        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        
        return torch.mean(loss)

    def sample(self, x, seq_x=None):
        if seq_x is not None:
            seq_h = self.seq_encoder(seq_x)
            x = torch.cat([x, seq_h.squeeze(1)], dim=1)
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples #, normal


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )

    def forward(self, x):
        params = self.network(x)
        # print(params)
        # input()
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)

import numpy as np
import torch
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--n-iterations", type=int, default=1)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    x = torch.randn(5, 1)
    y = torch.randn(5, 1)

    model = EventContinuousDecoder(1, 1, n_components=3)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for i in range(args.n_iterations):
        optimizer.zero_grad()
        loss = model.loss(x, y).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")

    samples = model.sample(x)
