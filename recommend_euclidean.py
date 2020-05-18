import numpy as np
import json
import torch
import pandas as pd



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

repo_core_teams = {}
with open('repo_core_targets.json') as rj:
    for l in rj.readlines()[62546:]:
        doc = json.loads(l)
        repo_core_teams[doc['repo']] = set(doc['core_teams'])

repo_profiles = {}
with open('repo_profiles.json') as rj:
    for l in rj.readlines():
        doc = json.loads(l)
        if not doc['repo'] in repo_core_teams:
            continue
        repo_profiles[doc['repo']] = doc
repo_profiles_df = pd.DataFrame(repo_profiles,index={'size','forks','subscribers','watchers','languages','topics'}).transpose()
repo_profiles_df.fillna('',inplace=True)


user_teams = {}
with open('team_profiles.json') as rj:
    for l in rj.readlines():
        doc = json.loads(l)
        tm = doc['team']
        for user in json.loads(tm):
            if not user in user_teams:
                user_teams[user] = {}
            contris = sum(doc['member_contributions'].values())
            user_teams[user][tm] = {
                'contri':doc['member_contributions'][user],
                'degree':doc['member_degrees'][user]
            }

user_profiles = {}
with open('user_profiles.json') as rj:
    for l in rj.readlines():
        doc = json.loads(l)
        user = doc['user']
        if not user in user_teams:
            continue
        user_profiles[doc['user']] = doc
user_profiles_df = pd.DataFrame(user_profiles,index={'repo_size','repo_forks','repo_subscribers','repo_watchers','languages','topics'}).transpose()
user_profiles_df.rename(columns={
                                   'repo_size':'size',
                                   'repo_forks':'forks',
                                   'repo_subscribers':'subscribers',
                                   'repo_watchers':'watchers',
                                   'languages':'languages',
                                   'topics':'topics'
                               },inplace=True)
user_profiles_df.fillna('',inplace=True)


numerics = ['size','forks','subscribers','watchers']
non_numerics = ['languages','topics']

minn = repo_profiles_df[numerics].min()
maxx = repo_profiles_df[numerics].max()
repo_profiles_df[numerics] = (repo_profiles_df[numerics]-minn)/(maxx-minn)
user_profiles_df[numerics] = (user_profiles_df[numerics]-minn)/(maxx-minn)


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

# device = torch.device('cuda:0')
device = torch.device('cpu')
user_num = torch.tensor(user_profiles_df[numerics].values,device=device,requires_grad=False)
batch_size = 5000
dis_num = []
for i in range(repo_profiles_df.shape[0]//batch_size+1):
    print(i)
    r_n = torch.tensor(repo_profiles_df.iloc[i*batch_size:(i+1)*batch_size][numerics].values,device=device,requires_grad=False)
    d_n = (r_n**2).sum(1,keepdim=True)+(user_num**2).sum(1,keepdim=True).transpose(0,1)-2*r_n.matmul(user_num.transpose(0,1))
    del r_n
    # dis_num.extend(d_n.cpu())
    dis_num.extend(d_n)
    del d_n

cnt = 0
ks = [5,10,30,50]
f_min = [open("recommend_euclidean_min_%d.json"%k,'w') for k in ks]
f_max = [open("recommend_euclidean_max_%d.json"%k,'w') for k in ks]
f_mean = [open("recommend_euclidean_mean_%d.json"%k,'w') for k in ks]
f_contri = [open("recommend_euclidean_contri_%d.json"%k,'w') for k in ks]
f_degree = [open("recommend_euclidean_degree_%d.json"%k,'w') for k in ks]
for repo,repo_profile in repo_profiles_df.iterrows():
    print(cnt)
    dis_non_num = torch.tensor(euclidean_non_numerics_v(repo_profile[non_numerics],user_profiles_df[non_numerics]),device=device,dtype=torch.float16,requires_grad=False)
    # dis = dis_num[cnt-1].cuda(device=device)+dis_non_num
    dis = dis_num[cnt-1] + dis_non_num
    # dis = dis.cpu().numpy()
    dis = dis.numpy()
    team_scores = {}
    team_score_contri = {}
    team_score_degree = {}
    for i,user in enumerate(user_profiles_df.index):
        for tm in user_teams[user]:
            if tm in repo_core_teams[repo]:
                continue
            if not tm in team_scores:
                team_scores[tm] = []
                team_score_contri[tm] = 0
                team_score_degree[tm] = 0
            team_scores[tm].append(dis[i])
            team_score_contri[tm] += user_teams[user][tm]['contri']*dis[i]
            team_score_degree[tm] += user_teams[user][tm]['degree']*dis[i]
    team_score_min = {}
    team_score_mean = {}
    team_score_max = {}
    for tm in team_scores:
        team_score_min[tm] = min(team_scores[tm])
        team_score_max[tm] = max(team_scores[tm])
        team_score_mean[tm] = sum(team_scores[tm])/len(team_scores[tm])
    tms_min = sort_to_k(list(team_score_min),ks[-1],key=lambda i:team_score_min[i])
    tms_mean = sort_to_k(list(team_score_mean),ks[-1],key=lambda i:team_score_mean[i])
    tms_max = sort_to_k(list(team_score_min),ks[-1],key=lambda i:team_score_max[i])
    tms_contri = sort_to_k(list(team_score_min),ks[-1],key=lambda i:team_score_contri[i])
    tms_degree = sort_to_k(list(team_score_min),ks[-1],key=lambda i:team_score_degree[i])
    
    for i,k in enumerate(ks):
        f_min[i].write(repo)
        for tm in tms_min[:ks[i]]:
            f_min[i].write("\t%s"%(json.dumps((tm,float(team_score_min[tm])))))
        f_min[i].write('\n')

        f_max[i].write(repo)
        for tm in tms_max[:ks[i]]:
            f_max[i].write("\t%s"%(json.dumps((tm,float(team_score_max[tm])))))
        f_max[i].write('\n')
        
        f_mean[i].write(repo)
        for tm in tms_mean[:ks[i]]:
            f_mean[i].write("\t%s"%(json.dumps((tm,float(team_score_mean[tm])))))
        f_mean[i].write('\n')
        
        f_contri[i].write(repo)
        for tm in tms_contri[:ks[i]]:
            f_contri[i].write("\t%s"%(json.dumps((tm,float(team_score_contri[tm])))))
        f_contri[i].write('\n')

        f_degree[i].write(repo)
        for tm in tms_degree[:ks[i]]:
            f_degree[i].write("\t%s"%(json.dumps((tm,float(team_score_degree[tm])))))
        f_degree[i].write('\n')
    
    cnt += 1
