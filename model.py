import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(__file__))

from layers import GraphAttentionLayer, GraphConvolution


# user layers
class HeteroLayer(nn.Module):
    def __init__(self, nfeat, nhid, dropout) -> None:
        super().__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gc2(x, adj))
        return x

# user layer
class NodeLayer1(nn.Module):
    def __init__(self, nfeat, nout, dropout, device):
        super().__init__()
        self.nout = nout
        self.complaintGat = HeteroLayer(nfeat, nout, dropout)
        self.rev_complaintGat = HeteroLayer(nfeat, nout, dropout)
        self.payGat = HeteroLayer(nfeat, nout, dropout)
        self.rev_payGat = HeteroLayer(nfeat, nout, dropout)
        self.cooperateGat = HeteroLayer(nfeat, nout, dropout)
        self.rev_cooperateGat = HeteroLayer(nfeat, nout, dropout)
        self.HeteroConv = [self.complaintGat, self.rev_complaintGat, self.payGat, self.rev_payGat, self.cooperateGat, self.rev_cooperateGat]

        self.device = device

    def forward(self, x, adj):
        """
            x: [batch_size, n, nx]
            adj: [batch_size, 6, n, n, ne]
        """
        h = torch.FloatTensor([]).to(self.device)

        for i in range(6):
            et = adj[:, i, :, :].squeeze()
            h = torch.cat((h, self.HeteroConv[i](x, et)), dim=2)

        if torch.any(torch.isnan(h)):
            print('self.HeteroConv', h)
            exit()
        return h

# trans layer
class TransLayer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, rate=False):
        super().__init__()
        self.dropout = dropout
        self.rate = rate

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        
        if rate:
            self.lastLayer = nn.Linear(nclass + 1, nclass)
            nn.init.xavier_uniform_(self.lastLayer.weight, gain=1.414)
        
    def forward(self, x, adj, rate=None):
        """
            x: [batch_size, n, ne+ne+nx]
            adj: [batch_size, n, n]
        """
        x = F.dropout(x, self.dropout, training=self.training)
        p = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(p, self.dropout, training=self.training)

        x = F.elu(self.out_att(x, adj))

        x = x[:, 0, :].squeeze()
        p = p[:, 0, :].squeeze()

        # neighbor label rate
        if self.rate:
            rate = rate.unsqueeze(-1)
            x = torch.cat([x, rate], dim=1)
            x = F.elu(self.lastLayer(x))

        return x, p

# main model
class MainModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, device) -> None:
        super().__init__()
        # user layer
        self.nodelayer = NodeLayer1(nfeat, nhid, dropout, device)
        # trans layer
        self.translayer = TransLayer(nhid*18 + 6, 8, nclass, dropout, alpha, nheads, rate=False)

    def forward(self, nx, tx, ti, nadj, tadj, mask=None):
        x = self.nodelayer(nx, nadj)

        # concat user representation
        ex_uin = ti[:, :, 0].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        bee_uin = ti[:, :, 1].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        frc_uin = ti[:, :, 2].unsqueeze(2).repeat(1, 1, x.shape[2]).long()

        ex_x = torch.gather(x, 1, ex_uin)
        bee_x = torch.gather(x, 1, bee_uin)
        frc_x = torch.gather(x, 1, frc_uin)
         
        x = torch.cat((ex_x, bee_x, frc_x, tx), dim=2)

        return self.translayer(x, tadj)
        

# adv
class AdvModel(nn.Module):
    def __init__(self, nfeat, nhid, ndomain, dropout) -> None:
        super().__init__()
        self.dropout = dropout

        self.linear1 = nn.Linear(nfeat + ndomain, nhid)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.414)

        self.linear2 = nn.Linear(nhid, 2)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1.414)

    def forward(self, x, domian):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([x, domian], dim=1)
        x = F.elu(self.linear1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.linear2(x))

        return x
