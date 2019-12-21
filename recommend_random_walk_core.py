import json
from pymongo import MongoClient


client = MongoClient()
db = client['gtr']

# repos = []
# with open('repo_profiles_new.json') as rj:
#     for rl in rj.readlines():
#         line = rl.strip().split('\t')
#         repos.append(line[0])

# teams = []
# user_team_ind = {}
# repo_team_ind = {}
# cnt = 0
# with open('team_tags.txt') as tmj:
#     for tml in tmj.readlines():
#         tm,dur,topics,lang,contr,center,aspl,ac,cen,sizes,repo_contributors,lang_diff,topic_diff,size_diff,wtch_diff,fork_diff,sbscrb_diff,feature_diff = tml.split('\t')
#         teams.append(tm)
#         rps = list(json.loads(sizes).keys())
#         for repo in rps:
#             if not repo in repo_team_ind:
#                 repo_team_ind[repo] = []
#             repo_team_ind[repo].append(cnt)
#         for user in json.loads(tm):
#             if not user in user_team_ind:
#                 user_team_ind[user] = []
#             user_team_ind[user].append(cnt)
#         cnt += 1

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

# repo_core = {}
# repo_core_team_ind = {}
# with open('repo_target_core.json','w') as tj:
#     with open("contributors.json") as rj:
#         for l in rj.readlines():
#             j = json.loads(l)
#             repo = j['repo']
#             if not repo in repos:
#                 continue
#             contribution = {}
#             for c in j['contributors']:
#                 contribution[c['login']] = c['contributions']
#             t_c = {}
#             # repo_team_mem = set()
#             for tm in repo_team_ind[repo]:
#                 t_c[tm] = 0
#                 team = json.loads(teams[tm])
#                 # repo_team_mem.update(team)
#                 for member in team:
#                     if member in contribution:
#                         t_c[tm] += contribution[member]
#             if len(repo_team_ind[repo])>1:
#                 core_team_ind = max(t_c,key=lambda x: t_c[x])
#                 repo_core[repo] = set(json.loads(teams[core_team_ind]))
#             else:
#                 core_team_ind = -1
#                 repo_core[repo] = set()
#             repo_core_team_ind[repo] = core_team_ind
#             tj.write(repo)
#             for tm in repo_team_ind[repo]:
#                 if tm == repo_core_team_ind[repo]:
#                     continue
#                 tj.write('\t%s'%(teams[tm]))
#             tj.write('\n')
#             kk = len(contribution)//10
#             cntr = sort_to_k(list(contribution),key = lambda x: contribution[x], reversed=True,k=kk)
#             for i in range(kk):
#                 # if cntr[i] in repo_team_mem:
#                 #     continue
#                 repo_core[repo].add(cntr[i])

graph = []
node_ind = {}
with open("network.json") as nj:
    j = json.load(nj)
    nodes = j['nodes']
    for i,node in enumerate(nodes):
        node_ind[node] = i
        graph.append({})
    aware = j['aware']
    for u1,g in enumerate(aware):
        for u in g:
            u2 = int(u)
            if not u2 in graph[u1]:
                graph[u1][u2] = 0
            if not u1 in graph[u2]:
                graph[u2][u1] = 0
            graph[u1][u2] += len(g[u])
            graph[u2][u1] += len(g[u])



cnt = 0
k=50
with open('recommend_random_walk_core.json','w') as rj:
    for repo in repo_core:
        print(cnt)
        team_score = {}
        for user in repo_core[repo]:
            if not user in node_ind:
                continue
            for u2 in graph[node_ind[user]]:
                if not nodes[u2] in user_team_ind:
                    continue
                for tm in user_team_ind[nodes[u2]]:
                    if tm == repo_core_team_ind[repo]:
                        continue
                    if not tm in team_score:
                        team_score[tm] = 0
                    team_score[tm] += graph[node_ind[user]][u2]
        tms = sort_to_k(list(team_score),k,key=lambda i:team_score[i],reversed=True)
        rj.write(repo)
        for i in range(min(k,len(tms))):
            rj.write("\t%s"%(json.dumps((teams[tms[i]],team_score[tms[i]]))))
        rj.write('\n')
        cnt += 1