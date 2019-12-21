import json
import pandas as pd

repo_profiles = {}
repo_teams = {}
with open('repo_profiles_new.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        profile = json.loads(line[1])
        repo_profiles[repo] = profile
    
teams = []
with open('team_profiles.json') as tj:
    for tl in tj.readlines():
        line = tl.split('\t')
        team = line[0]
        teams.append(team)

users = []
user_profiles = []
with open('user_profiles.json') as tj:
    for tl in tj.readlines():
        line = tl.split('\t')
        user = line[0]
        profile = json.loads(line[1])
        user_profiles.append(profile)
        users.append(user)

repo_profiles_df = pd.DataFrame(repo_profiles).transpose()
repo_profiles_df.fillna('',inplace=True)
user_profiles_df = pd.DataFrame(user_profiles,index=users)
user_profiles_df.rename(columns={
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
user_profiles_df[numerics] = (user_profiles_df[numerics]-minn)/(maxx-minn)

import numpy as np
import scipy

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


cuda0 = torch.device('cuda:0')
repo_num = torch.tensor(repo_profiles[numerics].values,device=cuda0,requires_grad=False)
user_num = torch.tensor(user_profiles_df[numerics].values,device=cuda0,requires_grad=False)
dis_num = (repo_num**2).sum(1,keepdim=True)+(user_num**2).sum(1,keepdim=True).transpose(0,1)-2*repo_num.matmul(user_num.transpose(0,1))

cnt = 0
k = 50
with open('recommend_euclidean_user.json','a') as oj:
    with open('recommend_euclidean_user.bp','r+') as rb:
        bp = int(rb.read())
        cnt = 0
        k = 50
        for repo,repo_profile in repo_profiles_df.iterrows():
            if cnt < bp:
                cnt += 1
                continue
            print(cnt)
            
            dis_non_num = torch.tensor(euclidean_non_numerics_v(repo_profiles[non_numerics],user_profiles[non_numerics]),device=cuda0,requires_grad=False)
            dis = dis_num[cnt-1]+dis_non_num
            dis = dis.cpu().numpy()



            idxs = sort_to_k(list(range(len(teams))),k,key=lambda i:dis[i])
            oj.write(repo+'\t')
            for i in range(k):
                oj.write(json.dumps(teams[idxs[i]])+'\t')
            oj.write('\n')
                
            cnt += 1
            rb.seek(0)
            rb.write(str(cnt))
