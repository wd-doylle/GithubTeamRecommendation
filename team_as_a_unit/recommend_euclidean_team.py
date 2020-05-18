import numpy as np
import json
import pandas as pd
import torch

repo_teams = {}
with open('repo_core_targets.json') as rj:
    for l in rj.readlines()[62546:]:
        doc = json.loads(l)
        repo_teams[doc['repo']] = set(doc['core_teams'])
repo_profiles = {}
with open('repo_profiles.json') as rj:
    for l in rj.readlines():
        doc = json.loads(l)
        if not doc['repo'] in repo_teams:
            continue
        repo_profiles[doc['repo']] = doc
repo_profiles_df = pd.DataFrame(repo_profiles,index={'size','forks','subscribers','watchers','languages','topics'}).transpose()
repo_profiles_df.fillna('',inplace=True)

team_profiles = []
teams = []
with open('team_interest.json') as rj:
    for l in rj.readlines():
        doc = json.loads(l)
        tm = doc.pop('team')
        teams.append(tm)
        team_profiles.append(doc)

team_profiles_df = pd.DataFrame(team_profiles,index=teams)
team_profiles_df.rename(columns={
                                   'repo_size':'size',
                                   'repo_forks':'forks',
                                   'repo_subscribers':'subscribers',
                                   'repo_watchers':'watchers',
                                   'languages':'languages',
                                   'topics':'topics'
                               },inplace=True)

numerics = ['size','forks','subscribers','watchers']
non_numerics = ['languages','topics']

minn = repo_profiles_df[numerics].min()
maxx = repo_profiles_df[numerics].max()
repo_profiles_df[numerics] = (repo_profiles_df[numerics]-minn)/(maxx-minn)
team_profiles_df[numerics] = (team_profiles_df[numerics]-minn)/(maxx-minn)



def euclidean_non_numerics(p1,p2):
    if p1[0] and p2[0]:
        c = len(set(p1[0]).intersection(set(p2[0])))
        dis_langs = 1-c/(len(p1[0])+len(p2[0])-c)
    else:
        dis_langs = 0
    if p1[1] and p2[1]:
        c = len(set(p1[1]).intersection(set(p2[1])))
        dis_topics = 1-c/(len(p1[1])+len(p2[1])-c)
    else:
        dis_topics = 1
    return dis_langs**2 + dis_topics**2
euclidean_non_numerics_v =  np.vectorize(euclidean_non_numerics,signature="(n),(n)->()")

device = torch.device('cpu')
team_num = torch.tensor(team_profiles_df[numerics].values,device=device,requires_grad=False)
batch_size = 5000
dis_num = []
for i in range(repo_profiles_df.shape[0]//batch_size+1):
    print(i)
    r_n = torch.tensor(repo_profiles_df.iloc[i*batch_size:(i+1)*batch_size][numerics].values,device=device,requires_grad=False)
    d_n = (r_n**2).sum(1,keepdim=True)+(team_num**2).sum(1,keepdim=True).transpose(0,1)-2*r_n.matmul(team_num.transpose(0,1))
    del r_n
    # dis_num.extend(d_n.cpu())
    dis_num.extend(d_n)
    del d_n


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
ks = [5,10,30,50]
f = [open("recommend_euclidean_team_%d.json"%k,'w') for k in ks]
for repo,repo_profile in repo_profiles_df.iterrows():
    print(cnt)
    dis_non_num = torch.tensor(euclidean_non_numerics_v(repo_profile[non_numerics],team_profiles_df[non_numerics]),device=device,dtype=torch.float16,requires_grad=False)
    # dis = dis_num[cnt-1].cuda(device=device)+dis_non_num
    dis = dis_num[cnt-1] + dis_non_num
    # dis = dis.cpu().numpy()
    dis = dis.numpy()
    team_scores = {}
    for i,tm in enumerate(team_profiles_df.index):
        if tm in repo_teams[repo]:
            continue
        if not tm in team_scores:
            team_scores[tm] = dis[i]

    tms = sort_to_k(list(team_scores),ks[-1],key=lambda i:team_scores[i])
    
    for i,k in enumerate(ks):
        f[i].write(repo)
        for tm in tms[:ks[i]]:
            f[i].write("\t%s"%(json.dumps((tm,float(team_scores[tm])))))
        f[i].write('\n')

    cnt += 1
