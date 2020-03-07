import json
from pymongo import MongoClient

client = MongoClient()
db = client['gtr']


team_ind = {}
team_profiles = db['team_profiles']
for i,row in enumerate(team_profiles.find()):
    team_ind[row['team']] = i

user_ind = {}
user_profiles = db['user_profiles']
for i,row in enumerate(user_profiles.find()):
    user_ind[row['user']] = i

repo_ind = {}
team_repo_contri = {}
team_repo_contri_target = {}
user_repo_contri = {}
user_repo_contri_target = {}
repo_core_targets = db['repo_core_targets']
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
    for user in row['core_users']:
        usr_ind = user_ind[user]
        if not usr_ind in user_repo_contri:
            user_repo_contri[usr_ind] = {}
        user_repo_contri[usr_ind][repo] = row['core_users'][user]
    for user in row['target_users']:
        usr_ind = user_ind[user]
        if not usr_ind in user_repo_contri_target:
            user_repo_contri_target[usr_ind] = {}
        user_repo_contri_target[usr_ind][repo] = row['target_users'][user]


# Training Labels
with open("team_interests_train.json",'w') as tj:
    for tm in team_repo_contri:
        tj.write("%s\n"%json.dumps({
                'team':tm,
                "interests":team_repo_contri[tm]
            }))
with open("user_interests_train.json",'w') as tj:
    for user in user_repo_contri:
        tj.write("%s\n"%json.dumps({
                'user':user,
                "interests":user_repo_contri[user]
            }))

# Validation Labels
with open("team_interests_target.json",'w') as tj:
    for tm in team_repo_contri_target:
        tj.write("%s\n"%json.dumps({
                'team':tm,
                "interests":team_repo_contri_target[tm]
            }))
with open("user_interests_target.json",'w') as tj:
    for user in user_repo_contri_target:
        tj.write("%s\n"%json.dumps({
                'user':user,
                "interests":user_repo_contri_target[user]
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