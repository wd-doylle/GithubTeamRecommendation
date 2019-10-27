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
        repo_teams[repo] = repo_profiles[repo].pop('teams')
    
team_profiles = []
teams = []
with open('team_profiles.json') as tj:
    for tl in tj.readlines():
        line = tl.split('\t')
        team = line[0]
        profile = json.loads(line[1])
        team_profiles.append(profile)
        teams.append(team)

repo_profiles_df = pd.DataFrame(repo_profiles).transpose()
repo_profiles_df.fillna('',inplace=True)
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

min = repo_profiles_df[numerics].min()
max = repo_profiles_df[numerics].max()
repo_profiles_df[numerics] = (repo_profiles_df[numerics]-min)/(max-min)
team_profiles_df[numerics] = (team_profiles_df[numerics]-min)/(max-min)

import numpy as np


euclidean_numerics_v = np.vectorize(
    lambda p1,p2: np.linalg.norm(p1-p2)**2,signature="(n),(n)->()"
)
def euclidean_non_numerics(p1,p2):
    langs = set(p1[0]).union(p2[0])
    topics = set(p1[1]).union(p2[1])
    p1_langs = pd.Series([lang in p1[0] for lang in langs])/np.sqrt(len(langs))
    p2_langs = pd.Series([lang in p2[0] for lang in langs])/np.sqrt(len(langs))
    p1_topics = pd.Series([topic in p1[1] for topic in topics])/np.sqrt(len(topics))
    p2_topics = pd.Series([topic in p2[1] for topic in topics])/np.sqrt(len(topics))
    return np.linalg.norm(p1_langs-p2_langs)**2 + np.linalg.norm(p1_topics-p2_topics)**2
euclidean_non_numerics_v =  np.vectorize(euclidean_non_numerics,signature="(n),(n)->()")
euclidean_v = np.vectorize(lambda num,non_num:np.sqrt(num+non_num))


with open('recommend_euclidean.json','a') as oj:
    with open('recommend_euclidean.bp','r+') as rb:
        bp = int(rb.read())
        cnt = 0
        for repo,repo_profile in repo_profiles_df.iterrows():
            if cnt < bp:
                cnt += 1
                continue
            print(cnt)
            
            teams = [t for t,p in team_profiles_df.iterrows() if not t in repo_teams[repo]]
            dis_num = euclidean_numerics_v(repo_profile[numerics],team_profiles_df.loc[teams,numerics])
            dis_non_num = euclidean_non_numerics_v(repo_profile[non_numerics],team_profiles_df.loc[teams,non_numerics])
            dis = euclidean_v(dis_num,dis_non_num)
            idxs = sorted(list(range(len(teams))),key=lambda i:dis[i])
            oj.write(repo+'\t')
            for i in range(50):
                oj.write(json.dumps(teams[idxs[i]])+'\t')
            oj.write('\n')
                
            cnt += 1
            rb.seek(0)
            rb.write(str(cnt))
