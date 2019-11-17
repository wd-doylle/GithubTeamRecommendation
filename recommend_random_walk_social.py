import json
import pandas as pd
import torch

cuda0 = torch.device('cuda:0')

t_g = []
teams = []
team_ind = {}
with open('team_graph.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        team = line[0]
        teams.append(team)
        team_ind[team] = len(teams)-1
        t_g.append(json.loads(line[1]))

repos = []
repo_teams = {}
with open('repo_profiles_new.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        profile = json.loads(line[1])
        repos.append(repo)
        repo_teams[repo] = profile['teams']

team_graph = []
for g in t_g:
    team_graph.append([0]*len(teams))
    for i in g:
        team_graph[-1][int(i)] = g[i]

for i in range(len(team_graph)):
    for j in range(i):
        team_graph[i][j] = team_graph[i][j]+team_graph[j][i]
        team_graph[j][i] = team_graph[i][j]
    team_graph[i][i] = 0


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



cnt = 0
steps = 100
alpha = 0.85
transfer = torch.tensor(team_graph,requires_grad=False,dtype=torch.float32,device=cuda0)
transfer /= transfer.sum(0)

k=50
with open('recommend_random_walk_social.json','w') as rj:
    for repo in repos:
        print(cnt)
        page_rank = torch.zeros(len(teams),requires_grad=False,device=cuda0)
        for team in repo_teams[repo]:
            page_rank[team_ind[team]] = 1/len(repo_teams[repo])
        for i in range(steps):
            page_rank = alpha*page_rank.matmul(transfer) + (1-alpha)/len(teams)
        page_rank = page_rank.cpu().numpy()

        tms = sort_to_k(list(range(len(teams))),k,key=lambda i:page_rank[i],reversed=True)
        rj.write(repo)
        for i in range(k):
            rj.write("\t%s"%(json.dumps((teams[tms[i]],page_rank[tms[i]].item()))))
        rj.write('\n')
        cnt += 1