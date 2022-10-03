from turtle import forward
import numpy as np
import time
import os
import glob
import copy

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from ..nn import BehaviorEncoding, STANEncoding , EventContinuousDecoder, EventDiscreteDecoder
from ..nn import generate_square_subsequent_mask


class Behavior2Event(nn.Module):

    def __init__(self, encoder_type, decoder_type, column_cont, column_disc, column_disc_num, short_len, long_len) -> None:
        super().__init__()

        if encoder_type == 'attn_long':
            self.encoder = BehaviorEncoding(n_cont=len(column_cont), n_disc_list=column_disc_num, seq_len=short_len)
            self.encoder_long = BehaviorEncoding(n_cont=len(column_cont), n_disc_list=column_disc_num, seq_len=long_len)
            self.d_model = self.encoder.d_model + self.encoder_long.d_model
        elif encoder_type == 'attn':
            self.encoder = BehaviorEncoding(n_cont=len(column_cont), n_disc_list=column_disc_num, seq_len=short_len)
            self.d_model = self.encoder.d_model
        elif encoder_type == 'cnn':
            self.encoder = STANEncoding(n_cont=len(column_cont), n_disc_list=column_disc_num)
            self.d_model = self.encoder.d_model
        else:
            assert encoder_type in ['attn_long', 'attn', 'cnn'], "Unknown Encoder Type"
        self.encoder_type = encoder_type

        if decoder_type == 'one': 
            self.decoder_cont = EventContinuousDecoder(self.d_model, len(column_cont))
            self.decoder_disc = EventDiscreteDecoder(self.d_model, column_disc_num)
        elif decoder_type == 'seq':
            self.decoder_cont = EventContinuousDecoder(self.d_model, len(column_cont), seq_decode=column_disc_num)
            self.decoder_disc = EventDiscreteDecoder(self.d_model, column_disc_num, seq_decode=True)
        else:
            assert decoder_type in ['one', 'seq'], "Unknown Decoder Type"
        self.decoder_type = decoder_type

        self.n_cont = len(column_cont)
        self.n_disc = len(column_disc)
        self.short_len = short_len
        self.long_len = long_len

        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01) 
            # m.bias.data.zero_()
            initrange = 0.1
            m.weight.data.uniform_(-initrange, initrange)

    def encode_long(self, x, src_mask=None):
        return self.encoder_long(x, src_mask)
    
    def forward_h(self, short, long, next=None, src_mask=None):
        seq_len = short.size(1) # [1024, 10, 22]
        h = self.encoder(short, src_mask[:seq_len, :seq_len])

        if self.encoder_type == 'attn_long':
            seq_len = long.size(1) # [1024, 10, 22]
            h_long = self.encoder_long(long, src_mask[:seq_len, :seq_len])
            h = torch.cat([h, h_long], 1)

        return h
    
    def loss(self, short, long, next, src_mask=None):
        h = self.forward_h(short, long, next, src_mask=src_mask)

        loss_disc = self.decoder_disc.loss(h, next[:,:self.n_disc])
        loss_cont = self.decoder_cont.loss(h, next[:,self.n_disc:], seq_x=next[:,:self.n_disc])

        # loss = loss_cont + loss_disc
        return loss_cont, loss_disc
    
    def sample(self, short, long, src_mask=None):
        h = self.forward_h(short, long, src_mask=src_mask)

        # [1, 16] | [1, 6]
        out_disc = self.decoder_disc.sample(h)
        out_cont = self.decoder_cont.sample(h, seq_x=out_disc)
    
        return torch.cat([out_disc, out_cont], dim=1)


class NetflowAttacker(object):
    def __init__(self, data_description, device, encoder_type='attn_long', decoder_type='seq', save_to='tmp') -> None:
        '''
        Args:
        - encoder_type: 'cnn', 'attn', 'attn_long'
        '''
        # data_description += [10, 300]
        column_cont, column_disc, column_disc_num, short_len, long_len = data_description
        
        self.model = Behavior2Event(encoder_type, decoder_type, column_cont, column_disc, column_disc_num, short_len, long_len).to(device)

        self.src_mask = generate_square_subsequent_mask(max(short_len, long_len)).to(device)
        # self.src_mask_long = generate_square_subsequent_mask(long_len).to(device)

        self.n_cont = len(column_cont)
        self.n_disc = len(column_disc)
        self.short_len = short_len
        self.long_len = long_len

        self.device = device
        self.save_to = save_to
    
    def save(self, epoch):
        file = os.path.join(os.getcwd(), self.save_to, 'saved_model', 'checkpoint_%d.pth' % epoch)
        head, tail = os.path.split(file)
        os.makedirs(head, exist_ok=True)
        torch.save(self.model.state_dict(), file)

    def load(self, load_epoch=None):
        if load_epoch is None:
            list_of_files = glob.glob(os.path.join(os.getcwd(), self.save_to, 'saved_model', 'checkpoint_*.pth'))
            latest_file = max(list_of_files, key=os.path.getctime)
            print('loading: ',latest_file)
            # self.load(latest_file)
            self.model.load_state_dict(torch.load(latest_file))
        else:
            # self.load(os.path.join(os.getcwd(), 'saved_model', 'checkpoint_%d.pth' % load_epoch))
            self.model.load_state_dict(torch.load(
                os.path.join(os.getcwd(), self.save_to, 'saved_model', 'checkpoint_%d.pth' % load_epoch)))

    def train(self, data_loader, optimizer):
        self.model.train()

        total_cont_loss = total_disc_loss = total_loss = total_examples = 0
        for i, (long, short, next) in enumerate(data_loader):
            optimizer.zero_grad()

            long = long.to(self.device)
            short = short.to(self.device)
            next = next.to(self.device)            

            loss_cont, loss_disc = self.model.loss(short, long, next, self.src_mask)
            loss = loss_disc + loss_cont
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()

            num_examples = next.size(0)
            total_loss += loss.item() * num_examples
            total_cont_loss += loss_cont.item() * num_examples
            total_disc_loss += loss_disc.item() * num_examples
            total_examples += num_examples
        
        return total_loss / total_examples, total_cont_loss/total_examples, total_disc_loss/total_examples
        
    def fit(self, dataset, num_epochs: int = 50, batch_size: int = 1024, learning_rate = 1e-2):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs//(5*2)))
        
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, \
            num_workers=12)

        for epoch in range(1, 1+num_epochs):
            start_time = time.time()
            loss, loss_cont, loss_disc = self.train(data_loader, optimizer)  
            scheduler.step()          
            self.save(epoch)
            end_time = time.time()

            print(f'Epoch: {epoch:02d}, '
                    f'Time: {end_time - start_time:.2f}, '
                    f'Loss Cont: {loss_cont:.4f}, '
                    f'Loss Disc: {loss_disc:.4f}, '
                    f'Loss: {loss:.4f}')
        
    @torch.no_grad()
    def sample(self, n: int = 10000, n_time=None, num_scenario = 1, load_epoch: int = None, bcl=None):
        print('Sampling:')
        self.load(load_epoch)
        self.model.eval()

        # initial marginal starter
        temp_x = torch.zeros(num_scenario, self.short_len, self.n_cont+self.n_disc).to(self.device)
        temp_long = torch.zeros(num_scenario, self.long_len, self.n_cont+self.n_disc).to(self.device)
        time_delta_col = 0

        gen_buff = []
        mem_bcl = [copy.deepcopy(bcl) for i in range(num_scenario)]
        check_time = time.time()

        for i in range(n):
            # print('in', temp_x.shape, temp_long.shape)
            temp_next = self.model.sample(temp_x, temp_long, self.src_mask)

            sc_long = []
            for sc in range(num_scenario):
                temp_long = mem_bcl[sc].sampling_loading_push(temp_next[sc][time_delta_col].to("cpu").numpy(), temp_next[sc].to("cpu").numpy())
                sc_long.append(torch.from_numpy(temp_long).unsqueeze(0))
            temp_long = torch.cat(sc_long, 0)
            temp_next = temp_next.unsqueeze(1) # [batch, dim] -> [batch, 1, dim] for context rolling and batch output
            
            gen_buff.append(temp_next)
            temp_x = torch.cat([temp_x[:,1:,:], temp_next], dim=1)

            temp_x = temp_x.float().to(self.device)
            temp_long = temp_long.float().to(self.device)

            if i%5000==0:
                print(f'scenario {0} * row {i} generated, '
                     f'Time: {time.time() - check_time:.2f}, '
                    )
                check_time = time.time()

        # [bs, n, dim] -> [[n, dim], ...]
        out = torch.cat(gen_buff, dim=1).detach().to("cpu").numpy()
        out = np.split(out, num_scenario)
        out = [np.squeeze(o, axis=0) for o in out]
        
        return out

    @torch.no_grad()
    def encode_long(self, dataset, load_epoch: int = None, batch_size: int = 1024 * 4):
        self.load(load_epoch)
        self.model.eval()

        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, \
            num_workers=12)
        
        data_long_emb = []
        for i, (long, short, next) in enumerate(data_loader):
            long = long.to(self.device)
            short = short.to(self.device)
            next = next.to(self.device)            

            seq_len = long.size(1) # [1024, 10, 22]
            src_mask = self.src_mask_long if seq_len == self.long_len else self.src_mask_long[:seq_len, :seq_len]

            h = self.model.encode(long, src_mask)
            data_long_emb.append(h)

        data_long_emb = torch.cat(data_long_emb, dim=0).detach().to("cpu").numpy()
        return data_long_emb
