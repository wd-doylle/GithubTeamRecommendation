import json
import pandas as pd
import torch

cuda0 = torch.device('cuda:0')

repo_teams = {}
with open('repo_profiles_new.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        profile = json.loads(line[1])
        repo_teams[repo] = profile['teams']

repo_graph = []
repos = []
cnt = 0
with open('repo_graph_sim.json') as rj:
    for rl in rj.readlines():
        cnt +=1
        line = rl.split('\t')
        repo = line[0]
        repos.append(repo)
        repo_graph.append(json.loads(line[1]))

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
steps = 1
alpha = 0.95
transfer = 1/torch.tensor(repo_graph,requires_grad=False,device=cuda0)
for i in range(len(repos)):
    transfer[i][i] = 0
transfer /= transfer.sum(0)
transfer[transfer!=transfer] = 0

k=50
with open('recommend_random_walk_sim.json','w') as rj:
    for repo in repos:
        print(cnt)
        page_rank = torch.zeros(len(repos),requires_grad=False,device=cuda0)
        page_rank[cnt] = 1
        page_rank = page_rank.matmul(transfer) + (1-alpha)/len(repos)
        page_rank = page_rank.cpu().numpy()
        team_rank = {}
        for i,p_r in enumerate(page_rank):
            for team in repo_teams[repos[i]]:
                if not team in team_rank:
                    team_rank[team] = 0
                team_rank[team] += p_r
        teams = sort_to_k(list(team_rank.keys()),k,key=lambda i:team_rank[i],reversed=True)
        rj.write(repo)
        for i in range(k):
            rj.write("\t%s"%(json.dumps((teams[i],team_rank[teams[i]]))))
        rj.write('\n')
        cnt += 1