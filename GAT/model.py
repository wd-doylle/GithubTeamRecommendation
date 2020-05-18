import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer,SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nhid, dropout, alpha, nheads, alt=0):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = {}
        self.out_att = {}
        phases = ['repo','user','team'] if alt == 0 else ['repo_user','team'] if alt == 1 else ['repo_user_team']
        for i,hier in enumerate(phases):
            self.attentions[hier] = [SpGraphAttentionLayer(nhid[i], nhid[i+1], dropout=dropout, alpha=alpha) for _ in range(nheads)]
            for j, attention in enumerate(self.attentions[hier]):
                self.add_module('attention_%s_%d'%(hier,j), attention)
            self.out_att[hier] =  SpGraphAttentionLayer(nhid[i+1]*nheads, nhid[i+1], dropout=dropout, alpha=alpha)
            self.add_module('out_attention_%s'%(hier), self.out_att[hier])

        self.out = nn.Linear(nhid[-1], 1)
        nn.init.xavier_normal_(self.out.weight.data, gain=1.414)
        self.add_module('out', self.out)
        self.forward = self.fwd if alt == 0 else self.fwd_alt1 if alt == 1 else self.fwd_alt2

    def fwd(self,repo,repo_users,users,user_edges,teams,team_users):
        repo_h = self.repo_forward(repo,repo_users,users) # U,nhid[1]
        user_h = self.user_forward(repo_h,user_edges) # U,nhid[2]
        team_h = self.team_forward(user_h,teams,team_users) # T,nhid[3]
        out = self.out(team_h) # T
        return torch.sigmoid(out)
    
    def fwd_alt1(self,repo,repo_users,users,user_edges,teams,team_users):
        user_h = self.repo_user_forward(repo,repo_users,users,user_edges) # U,nhid[1]
        team_h = self.team_forward(user_h,teams,team_users) # T,nhid[3]
        out = self.out(team_h) # T
        return torch.sigmoid(out)

    def fwd_alt2(self,repo,repo_users,users,user_edges,teams,team_users):
        team_h = self.repo_user_team_forward(repo,repo_users,users,user_edges,teams,team_users) # T,nhid[3]
        out = self.out(team_h) # T
        return torch.sigmoid(out)

    def repo_forward(self,repo,repo_users,users):
        input = torch.cat((users,repo.view(1,-1)),dim=0)
        N = users.size()[0]+1
        M = len(repo_users)
        e1 = torch.cat((torch.tensor(repo_users,dtype=torch.long).view(1,M), torch.full((1,M),N-1,dtype=torch.long)))
        e2 = torch.cat((torch.arange(N,dtype=torch.long).view(1,N), torch.arange(N,dtype=torch.long).view(1,N)))
        edges = torch.cat((e1,e2),dim=1)
        x = torch.cat([att(input, edges) for att in self.attentions['repo']], dim=1) # [N, nhid*nheads]
        x = self.out_att['repo'](x,edges)[:-1] # [N, nhid]

        return x
    
    def user_forward(self,users,edges):
        x = torch.cat([att(users, edges) for att in self.attentions['user']], dim=1) # [nheads]
        x = self.out_att['user'](x,edges) # [N, nhid]
        return x

    def repo_user_forward(self,repo,repo_users,users,user_edges):
        input = torch.cat((users,repo.view(1,-1)),dim=0)
        N = users.size()[0]+1
        M = len(repo_users)
        e1 = torch.cat((torch.tensor(repo_users,dtype=torch.long).view(1,M), torch.full((1,M),N-1,dtype=torch.long)))
        e2 = torch.cat((torch.arange(N,dtype=torch.long).view(1,N), torch.arange(N,dtype=torch.long).view(1,N)))
        edges = torch.cat((e1,e2),dim=1)
        edges = torch.cat((user_edges,edges),dim=1)
        x = torch.cat([att(input, edges) for att in self.attentions['repo_user']], dim=1) # [N, nhid*nheads]

        x = self.out_att['repo_user'](x,edges)[:-1] # [N, nhid]

        return x

    def team_forward(self,users,teams,team_users):
        input = torch.cat((users,teams),dim=0)
        M = users.size()[0]
        N = M + len(teams)
        e_x = []
        e_y = []
        for i,usrs in enumerate(team_users):
            e_x.extend([M+i]*len(usrs))
            e_y.extend(usrs)
        e1 = torch.cat((torch.tensor(e_x,dtype=torch.long).view(1,-1), torch.tensor(e_y,dtype=torch.long).view(1,-1)))
        e2 = torch.cat((torch.arange(N,dtype=torch.long).view(1,N), torch.arange(N,dtype=torch.long).view(1,N)))
        edges = torch.cat((e1,e2),dim=1)
        x = torch.cat([att(input, edges) for att in self.attentions['team']], dim=1) # [N, nhid*nheads]
        x = self.out_att['team'](x,edges)[M:N] # [N, nhid]
        return x

    def repo_user_team_forward(self,repo,repo_users,users,user_edges,teams,team_users):
        input = torch.cat((users,teams,repo.view(1,-1)),dim=0)
        N = users.size()[0]+teams.size()[0]+1
        M = len(repo_users)
        K = users.size()[0]
        repo_edges = torch.cat((torch.tensor(repo_users,dtype=torch.long).view(1,M), torch.full((1,M),N-1,dtype=torch.long)))
        e_x = []
        e_y = []
        for i,usrs in enumerate(team_users):
            e_x.extend([K+i]*len(usrs))
            e_y.extend(usrs)
        team_edges = torch.cat((torch.tensor(e_x,dtype=torch.long).view(1,-1), torch.tensor(e_y,dtype=torch.long).view(1,-1)))
        self_edges = torch.cat((torch.arange(N,dtype=torch.long).view(1,N), torch.arange(N,dtype=torch.long).view(1,N)))
        edges = torch.cat((user_edges,repo_edges,team_edges,self_edges),dim=1)
        x = torch.cat([att(input, edges) for att in self.attentions['repo_user_team']], dim=1) # [N, nhid*nheads]

        x = self.out_att['repo_user_team'](x,edges)[K:-1] # [N, nhid]

        return x

if __name__ == '__main__':
    import argparse
    from dataset import GATDataset
    import torch.optim as optim
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nb_heads', type=int, default=2, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--alt', type=int, default=0, help='Alternative architectures')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    
    nhid = [128,128,128,128]

    model = GAT(nhid=nhid, 
            dropout=args.dropout, 
            nheads=args.nb_heads, 
            alpha=args.alpha,
            alt=args.alt)
    
    # model.load_state_dict(torch.load("gat_1.pth"))
    # with open("gat_2.json",'w') as gj:
    #     gj.write(str(model.state_dict()))

    if args.cuda:
        model.cuda()
    
    device = torch.device("cuda") if args.cuda else torch.device("cpu")


    n_user = 50
    n_team = 60

    repo = torch.normal(0,2,(nhid[0],),device=device)
    repo_users = torch.randperm(n_user,device=device)[:int(0.5*n_user)]
    users = torch.normal(0,2,(n_user,nhid[0]),device=device)
    e1 = torch.cat((torch.arange(n_user,device=device),torch.arange(n_user,device=device))).view(1,-1)
    e2 = torch.cat((torch.arange(n_user,device=device),torch.randint(0,n_user,(1,n_user),device=device).view(-1))).view(1,-1)
    user_edges = torch.cat((e1,e2))
    teams = torch.normal(0,2,(n_team,nhid[0]),device=device)
    team_users = [torch.randperm(n_user,device=device)[:m] for m in torch.randint(3,n_user//2,(n_team,))]
    target = torch.cat((torch.zeros((n_team//2,1),device=device),torch.ones((n_team//2,1),device=device)))

    # print("REPO:")
    # repo_out = model.repo_forward(repo,repo_users,users)
    # print(repo_out)

    # print("USER:")
    # user_out = model.user_forward(repo_out,user_edges)
    # print(user_out)

    # print("TEAM:")
    # team_out = model.team_forward(user_out,teams,team_users)
    # print(team_out)

    # print("OUT:")
    # out = torch.sigmoid(model.out(team_out))
    # print(out)


    output = model(repo,repo_users,users,user_edges,teams,team_users)

    # a1 = model.state_dict()["attention_repo_1.a"].clone()
    # loss = nn.BCELoss()
    # l = loss(output,target)
    # l.backward()
    # a2 = model.state_dict()["attention_repo_1.a"]

    # print(a1==a2)
    print(output)



