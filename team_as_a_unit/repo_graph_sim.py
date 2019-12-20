import json
import pandas as pd
import numpy as np
import difflib
import torch

repo_profiles = {}
repo_teams = {}
with open('repo_profiles_new.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        profile = json.loads(line[1])
        repo_profiles[repo] = profile
        repo_teams[repo] = repo_profiles[repo].pop('teams')
        repo_profiles[repo]

repo_profiles = pd.DataFrame(repo_profiles).transpose()
repo_profiles.fillna('',inplace=True)

numerics = ['size','forks','subscribers','watchers']
non_numerics = ['languages','topics']

minn = repo_profiles[numerics].min()
maxx = repo_profiles[numerics].max()
repo_profiles[numerics] = (repo_profiles[numerics]-minn)/(maxx-minn)


def euclidean_non_numerics(p1,p2):
    c = len(set(p1[0]).intersection(set(p2[0])))
    sm_langs = c/(len(p1[0])+len(p2[0])-c)
    c = len(set(p1[1]).intersection(set(p2[1])))
    sm_topics = c/(len(p1[1])+len(p2[1])-c)
    return sm_langs**2 + sm_topics**2
euclidean_non_numerics_v =  np.vectorize(euclidean_non_numerics,signature="(n),(n)->()")

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

repos = repo_profiles.index

cuda0 = torch.device('cuda:0')
repo_num = torch.tensor(repo_profiles[numerics].values,device=cuda0,requires_grad=False)
dis_num = (repo_num**2).sum(1,keepdim=True)+(repo_num**2).sum(1,keepdim=True).transpose(0,1)-2*repo_num.matmul(repo_num.transpose(0,1))

cnt = 0
dis_non_num = []
k = 50
with open('repo_graph_sim.json','w') as rj:
    for repo,repo_profile in repo_profiles.iterrows():
        cnt += 1
        print(cnt)
        dis_non_num = torch.tensor(euclidean_non_numerics_v(repo_profile[non_numerics],repo_profiles[non_numerics]),device=cuda0,requires_grad=False)
        dis = dis_num[cnt-1]+dis_non_num
        dis = dis.cpu().numpy()

        rj.write("%s\t%s\n"%(repo,json.dumps(list(dis))))