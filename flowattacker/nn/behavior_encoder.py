
import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable


class AttributeEmbedding(nn.Module):

    def __init__(self, n_cont: int, n_disc_list: list, n_out=32, dropout=0.2) -> None:
        super().__init__()
        
        # calc for [disc | cont]
        emb_szs = [(nc, min(8, (nc+1)//2)) for nc in n_disc_list]
        # emb_szs = [(nc, min(50, (nc+1)//2)) for nc in column_disc_num]
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in emb_szs])
        
        n_emb = sum(e.embedding_dim for e in self.embeddings) # length of all embeddings combined
        self.n_emb, self.n_cont, self.n_disc = n_emb, n_cont, len(n_disc_list)
        
        self.emb_drop = nn.Dropout(dropout)
        self.cont_bn = nn.BatchNorm1d(self.n_cont)
        
        self.out_dim = (self.n_emb + self.n_cont + 1) // 2
        self.out_dim = max(i for i in [16, 32, 64, 128] if i <= self.out_dim)
        self.out_lin = nn.Linear(self.n_emb + self.n_cont, self.out_dim )
        self.out_drop = nn.Dropout(dropout)
        self.out_bn = nn.BatchNorm1d(self.out_dim)

    def forward(self, x):
        '''
        Args:
        - x: [batch, seq, dim]
        '''
        # x_cat, x_cont = x[:,:,:self.n_disc], x[:,:,self.n_disc:].long()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        x_cat, x_cont = torch.tensor_split(x, [self.n_disc], dim=2)
        x_cat = x_cat.long()
    
        # pass x_cont # [1024, 5, 1] -> [1024, 1, 5] for bn
        if x_cont.shape[-1] > 0:
            x_cont = torch.transpose(x_cont,1,2)
            x = self.cont_bn(x_cont)       
            x = torch.transpose(x,1,2)
 
        # pass x_cat # [1024, 5, 21] -> [1024*5, 21] for emb
        bs, seq_len, row_len = x_cat_shape =  x_cat.size()
        x_cat = x_cat.view(-1, row_len)                              # (bs*seq_len, row_len)
        x2 = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]   # [*, row_len] 
        x2 = torch.cat(x2, 1)                                        # -> [*, emb_size]
        x2 = x2.view(bs, seq_len, -1)                                # (bs, seq_len, embd_size)
        x2 = self.emb_drop(x2)
        
        # merge to [1024, 10, 6+410] / [1024, 10, 80] => [1024, 10, 40]
        x = torch.cat([x, x2], 2) if x_cont.shape[-1] > 0 else x2

        x = F.relu(self.out_lin(x))
        x = self.out_drop(x)

        x = torch.transpose(x,1,2)
        x = self.out_bn(x)     
        x = torch.transpose(x,1,2)

        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term) # if d_model%2==0 else torch.cos(position * div_term)[:,0:-1]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            -> x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0,1)
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class BehaviorEncoding(nn.Module):

    def __init__(self, n_cont: int, n_disc_list: list, seq_len: int = 10, # ntoken: int, 
                 nhead: int = 2, d_hidden: int = 200,
                 nlayers: int = 2,
                 dropout: float = 0.2,):
        super().__init__()

        self.model_type = 'Transformer'
        self.row_encoder = AttributeEmbedding(n_cont, n_disc_list, dropout)
        
        d_model = self.row_encoder.out_dim
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hidden, dropout, batch_first=True)
        self.context_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        
        self.d_model = d_model
        # self.decoder = nn.Linear(d_model, ntoken)

        # self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.row_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, feature]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, ntoken]
        """
        src = self.row_encoder(x) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # src: [1024, 10, row_emb] -> output: [1024, behavior_emb]
        # output = self.context_encoder(src, src_mask)
        output = self.context_encoder(src)
        output = output.sum(1)
    
        # output = self.decoder(output)
        return output

class STANEncoding(nn.Module):

    def __init__(self, n_cont: int, n_disc_list: list, seq_len: int = 10,
                 dropout: float = 0.2,):
        super().__init__()

        self.row_encoder = AttributeEmbedding(n_cont, n_disc_list, dropout)
        d_model = self.row_encoder.out_dim
        
        self.model_type = 'CNN'
        cnn_config = [64, 64, 'M', 128, 128, 'M']

        self.context_encoder = self.make_layers(cnn_config)

        self.d_model = d_model // 2**cnn_config.count('M')
        # self.decoder = nn.Linear(d_model, ntoken)
        # self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.row_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 1
        # print(cfg)
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, feature]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, ntoken]
        """
        src = self.row_encoder(x).unsqueeze(1)
        # src: [1024, seq, row_emb] -> src: [1024, 1, seq, row_emb]
        # output: [1024, behavior_emb]
        output = self.context_encoder(src)
        output = output.sum(1).sum(1)
        
        # output = self.decoder(output)
        return output