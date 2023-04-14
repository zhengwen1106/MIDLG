import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer
from torch.nn.init import kaiming_uniform_, xavier_uniform_




class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nedge, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass*3+nedge, nclass)
        kaiming_uniform_(self.linear.weight, nonlinearity='relu')

    def forward(self, x, adj, tx, tid):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        ex_uin = tid[:, :, 0].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        bee_uin = tid[:, :, 1].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        frc_uin = tid[:, :, 2].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        ex_x = torch.gather(x, 1, ex_uin)
        bee_x = torch.gather(x, 1, bee_uin)
        frc_x = torch.gather(x, 1, frc_uin)

        x = torch.cat((ex_x, bee_x, frc_x, tx), dim=2).squeeze()
        x = F.relu(self.linear(x))

        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nedge, dropout, alpha=0.2, nheads=4):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.linear = nn.Linear(nclass*3+nedge, nclass)
        kaiming_uniform_(self.linear.weight, nonlinearity='relu')

    def forward(self, x, adj, tx, tid):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        ex_uin = tid[:, :, 0].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        bee_uin = tid[:, :, 1].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        frc_uin = tid[:, :, 2].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        ex_x = torch.gather(x, 1, ex_uin)
        bee_x = torch.gather(x, 1, bee_uin)
        frc_x = torch.gather(x, 1, frc_uin)

        x = torch.cat((ex_x, bee_x, frc_x, tx), dim=2).squeeze()
        x = F.relu(self.linear(x))

        return F.log_softmax(x, dim=1)



class HeteroLayer(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha=0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.gc1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.gc2 = GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=True)

    def forward(self, x, adj):
        x = F.elu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gc2(x, adj))
        return x

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
        # x: [batch_size, n, nx]
        # adj: [batch_size, 6, n, n, ne]
        h = torch.FloatTensor([]).to(self.device)

        for i in range(6):
            et = adj[:, i, :, :].squeeze()
            h = torch.cat((h, self.HeteroConv[i](x, et)), dim=2)

        if torch.any(torch.isnan(h)):
            print('self.HeteroConv', h)
            exit()
    
        return h