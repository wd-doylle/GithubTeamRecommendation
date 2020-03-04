import json
import networkx as nx
from pymongo import MongoClient

client = MongoClient()
db = client['gtr']

repo_profiles = {}
r_ps = db['repo_profiles']
for rl in r_ps.find():
    repo_profiles[rl['repo']] = rl


# Team Contribution Breakdown
team_repos = {}
# team_repos_train = {}
team_member_contri = {}
repo_core_targets = db['repo_core_targets']
for row in repo_core_targets.find():
    repo = row['repo']
    for tm in row['core_teams']+row['target_teams']:
        if not tm in team_repos:
            team_repos[tm] = []
        team_repos[tm].append(repo)
        user_contri = row['core_users']
        user_contri.update(row['target_users'])
        if not tm in team_member_contri:
            team_member_contri[tm] = {}
        for mem in json.loads(tm):
            if not mem in team_member_contri[tm]:
                team_member_contri[tm][mem] = 0
            if not mem in user_contri:
                continue
            team_member_contri[tm][mem] += user_contri[mem]
    # for tm in row['core_teams']:
    #     if not tm in team_repos_train:
    #         team_repos_train[tm] = []
    #     team_repos_train[tm].append(repo)


# Team Structure
links = {}
team_member_degrees = {}
with open('Edges.link') as lkj:
    for line in lkj.readlines():
        dep,ter,repos = line.split('\t')
        if not dep in links:
            links[dep] = {}
        if not ter in links[dep]:
            links[dep][ter] = 0 
        links[dep][ter] += 1
for tm in team_repos:
    team = json.loads(tm)
    repos = team_repos[tm]
    G = nx.Graph()
    for member in team:
        if not member in links:
            continue
        for ter in links[member]:
            if ter in tm:
                if not member in G or not ter in G[member]:
                    G.add_edge(member,ter)
    m_d = {}
    for member in team:
        m_d[member] = G.degree[member]
    team_member_degrees[tm] = m_d


def team_feature(team_repos,repo_profiles):
    topics = set()
    languages = set()
    sizes = []
    forks = []
    watchers = []
    subscribers = []
    for repo in team_repos:
        if 'topics' in repo_profiles[repo]:
            topics.update(repo_profiles[repo]['topics'])
        if 'languages' in repo_profiles[repo]:
            languages.update(repo_profiles[repo]['languages'])
        sizes.append(repo_profiles[repo]['size'])
        forks.append(repo_profiles[repo]['forks'])
        watchers.append(repo_profiles[repo]['watchers'])
        subscribers.append(repo_profiles[repo]['subscribers'])
        
    repo_size = sum(sizes)/len(sizes)
    repo_forks = sum(forks)/len(forks)
    repo_watchers = sum(watchers)/len(watchers)
    repo_subscribers = sum(subscribers)/len(subscribers)
    
    profile = {
        'topics':list(topics),
        'languages':list(languages),
        'repo_size':repo_size,
        'repo_forks':repo_forks,
        'repo_watchers':repo_watchers,
        'repo_subscribers':repo_subscribers
    }
    return profile


# Repo Features & Team Profiles
db.drop_collection('team_profiles')
team_profiles = db['team_profiles']
cnt = 0
for tm in team_repos:
    profile = team_feature(team_repos[tm],repo_profiles)
    profile['team'] = tm
    profile['member_contributions'] = team_member_contri[tm]
    profile['member_degrees'] = team_member_degrees[tm]
    team_profiles.insert_one(profile)

    print(cnt)
    cnt += 1


# db.drop_collection('team_profiles_train')
# team_profiles = db['team_profiles_train']
# cnt = 0
# for tm in team_repos_train:
#     profile = team_feature(team_repos_train[tm],repo_profiles)
#     profile['team'] = tm
#     profile['member_contributions'] = team_member_contri[tm]
#     profile['member_degrees'] = team_member_degrees[tm]
#     team_profiles.insert_one(profile)

#     print(cnt)
#     cnt += 1