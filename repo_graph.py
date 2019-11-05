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
repo_profiles[numerics] = (repo_profiles[numerics]-minn)/(maxx-minn)


def euclidean_non_numerics(p1,p2):
    sm_langs = difflib.SequenceMatcher(None,p1[0],p2[0])
    sm_topics = difflib.SequenceMatcher(None,p1[1],p2[1])
    return sm_langs.ratio()**2 + sm_topics.ratio()**2
euclidean_non_numerics_v =  np.vectorize(euclidean_non_numerics,signature="(n),(n)->()")

repos = repo_profiles.index

cuda0 = torch.device('cuda:0')
repo_num = torch.tensor(repo_profiles[numerics].values,device=cuda0,requires_grad=False)
dis_num = (repo_num**2).sum(1,keepdim=True)+(repo_num**2).sum(1,keepdim=True).transpose(0,1)-2*repo_num.matmul(repo_num.transpose(0,1))

cnt = 0
dis_non_num = []
for repo,repo_profile in repo_profiles.iterrows():
    cnt += 1
    print(cnt)
    dis_non_num.append(euclidean_non_numerics_v(repo_profile[non_numerics],repo_profiles[non_numerics]))
    dis_non_num = torch.tensor(dis_non_num,device=cuda0,requires_grad=False)
dis = dis_num+dis_non_num
dis = dis.cpu().numpy()
k = 50
with open('repo_graph.json','w') as rj:
    for i,d in enumerate(dis):
        rj.write("%s\t%s\n"%(repos[i],json.dumps(d)))