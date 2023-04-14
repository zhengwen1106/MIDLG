import argparse
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from scipy.sparse import csr_matrix

import warnings

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)
        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()

class Prop1(torch.nn.Module):
    def __init__(self, num_classes, K):
        super(Prop1, self).__init__()
        self.K = K
        self.proj = Linear(num_classes, 1)
        
    def forward(self, x, adj, edge_weight=None):

        preds = []
        preds.append(x)
        for k in range(self.K):
            x = torch.matmul(adj, x)
            preds.append(x)
           
        pps = torch.stack(preds, dim=2)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(2)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def reset_parameters(self):
        self.proj.reset_parameters()
    
class DaGnn(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nedge, dropout, K):
        super(DaGnn, self).__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        self.prop = Prop1(nclass, K)
        self.dropout = dropout
        self.linear = nn.Linear(nclass*3+nedge, nclass)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, adj, tx, tid):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, adj)
        ex_uin = tid[:, :, 0].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        bee_uin = tid[:, :, 1].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        frc_uin = tid[:, :, 2].unsqueeze(2).repeat(1, 1, x.shape[2]).long()
        ex_x = torch.gather(x, 1, ex_uin)
        bee_x = torch.gather(x, 1, bee_uin)
        frc_x = torch.gather(x, 1, frc_uin)

        x = torch.cat((ex_x, bee_x, frc_x, tx), dim=2).squeeze()
        x = F.relu(self.linear(x))

        return F.log_softmax(x, dim=1)



