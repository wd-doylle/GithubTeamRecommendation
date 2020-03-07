import json
import torch
import numpy as np

user_interests = {}
users_train = set()
with open('../user_interests_train.json') as tmj:
    for tml in tmj.readlines():
        line = json.loads(tml)
        user_interests[line['user']] = {int(r):line['interests'][r] for r in line['interests']}
        users_train.add(line['user'])

with open('../user_interests_target.json') as tmj:
    for tml in tmj.readlines():
        line = json.loads(tml)
        user_interests[line['user']] = {int(r):line['interests'][r] for r in line['interests']}

print(len(user_interests))
print(len(users_train))

num_repos = 0
for user in user_interests:
    total_contri = 0
    for repo in user_interests[user]:
        num_repos = max(num_repos,repo+1)
        total_contri += user_interests[user][repo]
    for repo in user_interests[user]:
        user_interests[user][repo] /= total_contri


class SAE(torch.nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = torch.nn.Linear(num_repos, 64)
        self.fc2 = torch.nn.Linear(64, num_repos)
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


nb_epoch = 1
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for user in users_train:
        input = torch.zeros(num_repos,device=cuda0)
        for repo in user_interests[user]:
            input[repo] = user_interests[user][repo]
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = num_repos/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step()
        del input
        del target
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))



team_members = set()
with open("../team_members.json") as tj:
    for l in tj.readlines():
        line = json.loads(l)
        team_members.update(line['members'])

gg = {}
with open("../user_social.json") as nj:
    gg = json.load(nj)

graph = {}
for n1 in gg:
    ss = 0
    graph[int(n1)] = {}
    for n2 in gg[n1]:
        ss += gg[n1][n2]
    for n2 in gg[n1]:
        graph[int(n1)][int(n2)] = gg[n1][n2]/ss


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


user_pred = {}
with torch.no_grad():
    with open("sae_user_score.json",'w') as f:
        for i,user in enumerate(user_interests):
            if not user in team_members:
                continue
            print(i)
            if not user in user_pred:
                input = torch.zeros(num_repos,device=cuda0)
                for repo in user_interests[user]:
                    input[repo] = user_interests[user][repo]
                user_pred[user] = sae(input).cpu()
                del input
            pred = user_pred[user]
            for n2 in graph[user]:
                if not n2 in user_interests:
                    continue
                if not n2 in user_pred:
                    input = torch.zeros(num_repos,device=cuda0)
                    for repo in user_interests[n2]:
                        input[repo] = user_interests[n2][repo]
                    user_pred[n2] = sae(input).cpu()
                    del input
                pred += user_pred[n2]*graph[user][n2]
            recs = sort_to_k(list(range(len(pred))),k,key=lambda i:pred[i],reversed=True)
            f.write(str(user))
            for rec in recs[:k]:
                f.write("\t%s"%json.dumps((rec,pred[rec].item())))
            f.write('\n')