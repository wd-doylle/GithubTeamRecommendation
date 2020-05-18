import torch
import torch.nn as nn
import json
from pymongo import MongoClient

client = MongoClient()
db = client['gtr']

mins = {}
maxs = {}

team_ind = {}
team_profiles = []
mins['team'] = [None,None,None,None]
maxs['team'] = [None,None,None,None]
t_p = db['team_profiles']
for i,row in enumerate(t_p.find()):
    team_ind[row['team']] = i
    team_profiles.append([row['repo_size'],row['repo_watchers'],row['repo_forks'],row['repo_subscribers']])
    for i,f in enumerate(team_profiles[-1]):
        if not mins['team'][i] or mins['team'][i] > f:
            mins['team'][i] = f
        if not maxs['team'][i] or maxs['team'][i] < f:
            maxs['team'][i] = f

user_ind = {}
user_profiles = []
mins['user'] = [None,None,None,None]
maxs['user'] = [None,None,None,None]
u_p = db['user_profiles']
for i,row in enumerate(u_p.find()):
    user_ind[row['user']] = i
    user_profiles.append([row['repo_size'],row['repo_watchers'],row['repo_forks'],row['repo_subscribers']])
    for i,f in enumerate(user_profiles[-1]):
        if not mins['user'][i] or mins['user'][i] > f:
            mins['user'][i] = f
        if not maxs['user'][i] or maxs['user'][i] < f:
            maxs['user'][i] = f

repo_ind = {}
team_repo_contri = {}
team_repo_contri_target = {}
user_repo_contri = {}
user_repo_contri_target = {}
repo_core_targets = db['repo_core_targets']
max_contri_user, max_contri_team = 0, 0
for i,row in enumerate(repo_core_targets.find()):
    repo = row['repo']
    repo_ind[repo] = i
    repo = i
    for tm in row['core_teams']:
        tm_ind = team_ind[tm]
        user_contri = row['core_users']
        if not tm_ind in team_repo_contri:
            team_repo_contri[tm_ind] = {}
        team_repo_contri[tm_ind][repo] = 0
        for mem in json.loads(tm):
            if not mem in user_contri:
                continue
            team_repo_contri[tm_ind][repo] += user_contri[mem]
        max_contri_team = max(max_contri_team,team_repo_contri[tm_ind][repo])
    for tm in row['target_teams']:
        tm_ind = team_ind[tm]
        user_contri = row['core_users'].copy()
        user_contri.update(row['target_users'])
        if not tm_ind in team_repo_contri_target:
            team_repo_contri_target[tm_ind] = {}
        team_repo_contri_target[tm_ind][repo] = 0
        for mem in json.loads(tm):
            if not mem in user_contri:
                continue
            team_repo_contri_target[tm_ind][repo] += user_contri[mem]
        max_contri_team = max(max_contri_team,team_repo_contri_target[tm_ind][repo])
    for user in row['core_users']:
        usr_ind = user_ind[user]
        if not usr_ind in user_repo_contri:
            user_repo_contri[usr_ind] = {}
        user_repo_contri[usr_ind][repo] = row['core_users'][user]
        max_contri_user = max(max_contri_user,user_repo_contri[usr_ind][repo])
    for user in row['target_users']:
        usr_ind = user_ind[user]
        if not usr_ind in user_repo_contri_target:
            user_repo_contri_target[usr_ind] = {}
        user_repo_contri_target[usr_ind][repo] = row['target_users'][user]
        max_contri_user = max(max_contri_user,user_repo_contri_target[usr_ind][repo])


# Training Labels
with open("team_interests_train.json",'w') as tj:
    for tm in team_repo_contri:
        tm_interest = {}
        for repo in team_repo_contri[tm]:
            # tm_interest[repo] = team_repo_contri[tm][repo]/max_contri_team
            tm_interest[repo] = 1

        tj.write("%s\n"%json.dumps({
                'team':tm,
                "interests":tm_interest
            }))
with open("user_interests_train.json",'w') as tj:
    for user in user_repo_contri:
        usr_interest = {}
        for repo in user_repo_contri[user]:
            # usr_interest[repo] = user_repo_contri[user][repo]/max_contri_user
            usr_interest[repo] = 1

        tj.write("%s\n"%json.dumps({
                'user':user,
                "interests":usr_interest
            }))

# Validation Labels
with open("team_interests_target.json",'w') as tj:
    for tm in team_repo_contri_target:
        tm_interest = {}
        for repo in team_repo_contri_target[tm]:
            # tm_interest[repo] = team_repo_contri_target[tm][repo]/max_contri_team
            tm_interest[repo] = 1

        tj.write("%s\n"%json.dumps({
                'team':tm,
                "interests":tm_interest
            }))
with open("user_interests_target.json",'w') as tj:
    for user in user_repo_contri_target:
        usr_interest = {}
        for repo in user_repo_contri_target[user]:
            # usr_interest[repo] = user_repo_contri_target[user][repo]/max_contri_user
            usr_interest[repo] = 1

        tj.write("%s\n"%json.dumps({
                'user':user,
                "interests":usr_interest
            }))


# User Network
with open("user_social.json",'w') as sj:
    with open("network.json") as nj:
        j = json.load(nj)
        nodes = j['nodes']
        aware = j['aware']
        graph = {}
        for u1,g in enumerate(aware):
            if not nodes[u1] in user_ind:
                continue
            n1 = user_ind[nodes[u1]]
            if not n1 in graph:
                graph[n1] = {}
            for u in g:
                u2 = int(u)
                if not nodes[u2] in user_ind:
                    continue
                n2 = user_ind[nodes[u2]]
                if not n2 in graph:
                    graph[n2] = {}
                if not n2 in graph[n1]:
                    graph[n1][n2] = 0
                if not n1 in graph[n2]:
                    graph[n2][n1] = 0
                graph[n1][n2] += len(g[u])
                graph[n2][n1] += len(g[u])
        json.dump(graph,sj)


# Team Member List
with open("team_members.json",'w') as tj:
    for tm in team_ind:
        tj.write("%s\n"%json.dumps({
                'team':team_ind[tm],
                'members':[user_ind[u] for u in json.loads(tm) if u in user_ind]
            }))


repo_profiles = {}
mins['repo'] = [None,None,None,None]
maxs['repo'] = [None,None,None,None]
r_p = db['repo_profiles']
for i,row in enumerate(r_p.find()):
    if row['repo'] in repo_ind:
        repo_profiles[repo_ind[row['repo']]] = [row['size'],row['watchers'],row['forks'],row['subscribers']]
        for i,f in enumerate(repo_profiles[repo_ind[row['repo']]]):
            if not mins['repo'][i] or mins['repo'][i] > f:
                mins['repo'][i] = f
            if not maxs['repo'][i] or maxs['repo'][i] < f:
                maxs['repo'][i] = f


embedding_dim = 124
repo_embedding = nn.Embedding(len(repo_ind), embedding_dim)
with open("GAT/repo_embedding.json",'w') as rj:
    for i in range(len(repo_ind)):
        repo_emb = repo_profiles[i] if i in repo_profiles else [0,0,0,0]
        repo_emb = [2*(repo_emb[i]-(maxs['repo'][i]+mins['repo'][i])/2)/(maxs['repo'][i]-mins['repo'][i]) for i in range(len(repo_emb))]
        repo_emb = repo_emb + [float(n) for n in repo_embedding(torch.tensor([i]))[0]]
        # print(repo_emb)
        rj.write(json.dumps(repo_emb))
        rj.write('\n')

user_embedding = nn.Embedding(len(user_profiles), embedding_dim)
with open("GAT/user_embedding.json",'w') as rj:
    for i in range(len(user_profiles)):
        user_emb = user_profiles[i]
        user_emb = [2*(user_emb[i]-(maxs['user'][i]+mins['user'][i])/2)/(maxs['user'][i]-mins['user'][i]) for i in range(len(user_emb))]
        user_emb = user_emb + [float(n) for n in user_embedding(torch.tensor([i]))[0]]
        rj.write(json.dumps(user_emb))
        rj.write('\n')

team_embedding = nn.Embedding(len(repo_ind), embedding_dim)
with open("GAT/team_embedding.json",'w') as rj:
    for i in range(len(team_profiles)):
        team_emb = team_profiles[i]
        team_emb = [2*(team_emb[i]-(maxs['team'][i]+mins['team'][i])/2)/(maxs['team'][i]-mins['team'][i]) for i in range(len(team_emb))]
        team_emb = team_emb + [float(n) for n in team_embedding(torch.tensor([i]))[0]]
        rj.write(json.dumps(team_emb))
        rj.write('\n')