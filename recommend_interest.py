import numpy as np
import json
# from pymongo import MongoClient
import torch
import pandas as pd


# client = MongoClient()
# db = client['gtr']

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
# for doc in db['repo_core_targets'].find():
with open('repo_core_targets.json') as rj:
    for l in rj.readlines():
        doc = json.loads(l)
        repo_core_teams[doc['repo']] = set(doc['core_team'])

repo_profiles = {}
# for repo in repo_core_teams:
#     doc = db['repo_profiles'].find_one({'repo':repo})
with open('repo_profiles.json') as rj:
    for l in rj.readlines():
        doc = json.loads(l)
        if not doc['repo'] in repo_core_teams:
            continue
        repo_profiles[doc['repo']] = doc
repo_profiles_df = pd.DataFrame(repo_profiles,index={'size','forks','subscribers','watchers','languages','topics'}).transpose()
repo_profiles_df.fillna('',inplace=True)


user_profiles = {}
# for doc in db['user_profiles'].find():
with open('user_profiles.json') as rj:
    for l in rj.readlines():
        doc = json.loads(l)
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

user_teams = {}
# for tm in db['team_profiles'].distinct('team'):
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

cuda0 = torch.device('cuda:0')
user_num = torch.tensor(user_profiles_df[numerics].values,device=cuda0,dtype=torch.float16,requires_grad=False)
batch_size = 10000
dis_num = []
for i in range(repo_profiles_df.shape[0]//batch_size+1):
    print(i)
    r_n = torch.tensor(repo_profiles_df.iloc[i*batch_size:(i+1)*batch_size][numerics].values,device=cuda0,dtype=torch.float16,requires_grad=False)
    d_n = (r_n**2).sum(1,keepdim=True)+(user_num**2).sum(1,keepdim=True).transpose(0,1)-2*r_n.matmul(user_num.transpose(0,1))
    del r_n
    dis_num.extend(d_n.cpu())
    del d_n

cnt = 0
k = 50
f_min = open("recommend_interest_min.json",'w')
f_max = open("recommend_interest_max.json",'w')
f_mean = open("recommend_interest_mean.json",'w')
f_contri = open("recommend_interest_contri.json",'w')
f_degree = open("recommend_interest_degree.json",'w')
for repo,repo_profile in repo_profiles_df.iterrows():
    print(cnt)
    dis_non_num = torch.tensor(euclidean_non_numerics_v(repo_profile[non_numerics],user_profiles_df[non_numerics]),device=cuda0,dtype=torch.float16,requires_grad=False)
    dis = dis_num[cnt-1].cuda(device=cuda0)+dis_non_num
    dis = dis.cpu().numpy()
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
    tms_min = sort_to_k(list(team_score_min),k,key=lambda i:team_score_min[i])
    tms_mean = sort_to_k(list(team_score_mean),k,key=lambda i:team_score_mean[i])
    tms_max = sort_to_k(list(team_score_min),k,key=lambda i:team_score_max[i])
    tms_contri = sort_to_k(list(team_score_min),k,key=lambda i:team_score_contri[i])
    tms_degree = sort_to_k(list(team_score_min),k,key=lambda i:team_score_degree[i])
    
    f_min.write(repo)
    for tm in tms_min[:k]:
        f_min.write("\t%s"%(json.dumps((tm,float(team_score_min[tm])))))
    f_min.write('\n')

    f_max.write(repo)
    for tm in tms_max[:k]:
        f_max.write("\t%s"%(json.dumps((tm,float(team_score_max[tm])))))
    f_max.write('\n')
    
    f_mean.write(repo)
    for tm in tms_mean[:k]:
        f_mean.write("\t%s"%(json.dumps((tm,float(team_score_mean[tm])))))
    f_mean.write('\n')
    
    f_contri.write(repo)
    for tm in tms_contri[:k]:
        f_contri.write("\t%s"%(json.dumps((tm,float(team_score_contri[tm])))))
    f_contri.write('\n')

    f_degree.write(repo)
    for tm in tms_degree[:k]:
        f_degree.write("\t%s"%(json.dumps((tm,float(team_score_degree[tm])))))
    f_degree.write('\n')
    
    cnt += 1
