import json
import torch
import numpy as np

user_repo = {}
repo_inds = {}
repos = []
with open('../contributors.json') as tmj:
    i = 0
    for tml in tmj.readlines():
        line = json.loads(tml)
        repo_inds[line['repo']] = i
        repos.append(line['repo'])
        i += 1
        for user in line['contributors']:
            if not user['login'] in user_repo:
                user_repo[user['login']] = {}
            user_repo[user['login']][line['repo']] = user['contributions']

for user in user_repo:
    total_contri = 0
    for repo in user_repo[user]:
        total_contri += user_repo[user][repo]
    for repo in user_repo[user]:
        user_repo[user][repo] /= total_contri


class SAE(torch.nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = torch.nn.Linear(len(repo_inds), 64)
        self.fc2 = torch.nn.Linear(64, len(repo_inds))
        self.activation = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
cuda0 = torch.device('cuda:0')
sae = SAE()
sae.cuda(device=cuda0)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(sae.parameters(), lr = 0.001, weight_decay = 0.5)


nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for user in user_repo:
        input = torch.zeros(len(repo_inds),device=cuda0)
        for repo in user_repo[user]:
            input[repo_inds[repo]] = user_repo[user][repo]
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = len(repo_inds)/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))



graph = {}
with open("../network.json") as nj:
    j = json.load(nj)
    nodes = j['nodes']
    aware = j['aware']
    for u1,g in enumerate(aware):
        n1 = nodes[u1]
        if not n1 in user_repo:
            continue
        if not n1 in graph:
            graph[n1] = {}
        for u in g:
            u2 = int(u)
            n2 = nodes[u2]
            if not n2 in user_repo:
                continue
            if not n2 in graph:
                graph[n2] = {}
            if not n2 in graph[n1]:
                graph[n1][n2] = 0
            if not n1 in graph[n2]:
                graph[n2][n1] = 0
            graph[n1][n2] += len(g[u])
            graph[n2][n1] += len(g[u])

for n1 in graph:
    ss = 0
    for n2 in graph[n1]:
        ss += graph[n1][n2]
    for n2 in graph[n1]:
        graph[n1][n2] /= ss


def sort_to_k(ary,k,key=lambda x:x,reversed=False):
    k = min(k,len(ary))
    for i in range(k):
        for j in range(len(ary)-1-i):
            if not reversed:
                if key(ary[len(ary)-1-j]) < key(ary[len(ary)-2-j]):
                    ary[len(ary)-1-j],ary[len(ary)-2-j] = ary[len(ary)-2-j],ary[len(ary)-1-j]
            else:
                if key(ary[len(ary)-1-j]) > key(ary[len(ary)-2-j]):
                    ary[len(ary)-1-j],ary[len(ary)-2-j] = ary[len(ary)-2-j],ary[len(ary)-1-j]
    return ary

k = 30

with open("sae_user_score.json",'w') as f:
    for user in user_repo:
        input = torch.zeros(len(repo_inds),device=cuda0)
        for repo in user_repo[user]:
            input[repo_inds[repo]] = user_repo[user][repo]
        pred = sae(input)
        for n2 in graph[user]:
            input = torch.zeros(len(repo_inds),device=cuda0)
            for repo in user_repo[n2]:
                input[repo_inds[repo]] = user_repo[n2][repo]
            output = sae(input)
            pred += output*graph[user][n2]
        pred = pred.cpu()
        recs = sort_to_k(list(range(len(pred))),k,key=lambda i:pred[i],reversed=True)
        f.write(user)
        for rec in recs[:k]:
            f.write("\t%s",json.dumps((repos[rec],pred[rec].item())))
        f.write('\n')