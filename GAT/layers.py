import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, edges):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        adj = torch.sparse_coo_tensor(edges, torch.ones((len(edges[0]))), (N,N), device=torch.device("cuda")).to_dense()

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.mm(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = (a.t()).mm(grad_output)
        assert not torch.isnan(grad_values).any()
        try:
            assert not torch.isnan(grad_b).any()
        except:
            print(a)
            print(grad_output)
            print(grad_b)
            assert False
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()


    def forward(self, input, edge):

        N = input.size()[0]

        # input = input.masked_fill(input==0,1e-6)

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # print('h:')
        # print(h)

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = self.leakyrelu(self.a.mm(edge_h).squeeze())
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        # print('edge_e:')
        # print(edge_e)

        edge_att = self.sparse_softmax(edge, edge_e, (N,N))
        edge_att = F.dropout(edge_att, self.dropout, training=self.training)
        # edge_att: N x N
        
        # print('edge_att:')
        # print(edge_att)
       
        h_prime = self.special_spmm(edge, edge_att, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        return F.elu(h_prime)
    
    def sparse_softmax(self,indices,values,size):
        assert indices.requires_grad == False
        # print(indices)
        # print(values)
        # print(size)
        # a = torch.sparse_coo_tensor(indices, values, size,requires_grad=True)
        # return torch.softmax(a.to_dense(),dim=0)
        rows = {}
        inds = {}
        x = indices[0]
        for i,_x in enumerate(x):
            _x = _x.item()
            if not _x in rows:
                rows[_x] = []
                inds[_x] = []
            rows[_x].append(values[i].view(1))
            inds[_x].append(i)
        out = torch.empty_like(values)
        # print(rows[list(rows.keys())[0]])
        for i in rows:
            if not rows[i]:
                continue
            # print(rows[i])
            s = F.softmax(torch.cat(rows[i]),dim=0)
            # print(s)
            for j,ind in enumerate(inds[i]):
                out[ind] = s[j] 
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# def sparse_row_sum(indices,values,size):
#         rows = {}
#         inds = {}
#         x = indices[0]
#         for i,_x in enumerate(x):
#             _x = _x.item()
#             if not _x in rows:
#                 rows[_x] = []
#                 inds[_x] = []
#             rows[_x].append(values[i].view(1))
#             inds[_x].append(i)
#         out = []
#         for i in rows:
#             if not rows[i]:
#                 continue
#             s = torch.cat(rows[i]).sum().view(1)
#             # print(s)
#             out.append(s)
#         return torch.cat(out)

if __name__ == '__main__':
    
    cuda = False
    device = torch.device("cuda") if cuda else torch.device("cpu")

    n_user = 50
    h_dim = 32
    users = torch.normal(0,0.2,(n_user,h_dim),device=device,requires_grad=True)
    e1 = torch.cat((torch.arange(n_user,device=device),torch.arange(n_user,device=device))).view(1,-1)
    e2 = torch.cat((torch.arange(n_user,device=device),torch.randint(0,n_user,(1,n_user),device=device).view(-1))).view(1,-1)
    user_edges = torch.cat((e1,e2))
    # values = torch.normal(0,0.2,(2*n_user,),device=device,requires_grad=True)


    layer = SpGraphAttentionLayer(h_dim,h_dim,0.6,0.2)
    layer.train()
    linear = nn.Linear(h_dim, 1)
    loss = nn.BCELoss()

    u = F.dropout(users,0.6)

    out = torch.sigmoid(linear(layer(u,user_edges)))
    target = torch.cat((torch.zeros((n_user//2,1),device=device),torch.ones((n_user//2,1),device=device)))
    
    print(out)

    l = loss(out,target)

    l.backward()

    print(layer.a.grad)
    print(layer.W.grad)