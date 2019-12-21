import json
import networkx as nx
from pymongo import MongoClient

client = MongoClient()
db = client['gtr']

repo_profiles = {}
with open('repo_profiles.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        profiles = json.loads(line[1])
        repo_profiles[repo] = profiles


db.drop_collection('team_profiles')
team_profiles = db['team_profiles']
t_ps = {}
team_repos = {}
with open('team_tags.txt') as tmj:
    for tml in tmj.readlines():
        tm,dur,topics,lang,contr,center,aspl,ac,cen,sizes,repo_contributors,lang_diff,topic_diff,size_diff,wtch_diff,fork_diff,sbscrb_diff,feature_diff = tml.split('\t')
        repos = list(json.loads(sizes).keys())
        topics = set()
        languages = set()
        sizes = []
        forks = []
        watchers = []
        subscribers = []
        for repo in repos:
            if not repo in repo_profiles:
                continue
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
        t_ps[tm] = ({
            'team':tm,
            'topics':list(topics),
            'languages':list(languages),
            'repo_size':repo_size,
            'repo_forks':repo_forks,
            'repo_watchers':repo_watchers,
            'repo_subscribers':repo_subscribers,
        })
        team_repos[tm] = repos


links = {}
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
    t_ps[tm]['member_degrees'] = m_d


repo_teams = db['repo_teams']
r_ts = {}
member_contri = {}
for doc in repo_teams.find():
    r_ts[doc['repo']] = doc['teams']
with open("contributors.json") as rj:
    for l in rj.readlines():
        j = json.loads(l)
        repo = j['repo']
        if not repo in r_ts:
            continue
        user_contri = {}
        for c in j['contributors']:
            user_contri[c['login']] = c['contributions']
        for tm in r_ts[repo]:
            if not tm in member_contri:
                member_contri[tm] = {}
            for mem in json.loads(tm):
                if not mem in member_contri[tm]:
                    member_contri[tm][mem] = 0
                if not mem in user_contri:
                    continue
                member_contri[tm][mem] += user_contri[mem]

cnt = 0
for tm in t_ps:
    print(cnt)
    t_ps[tm]['member_contributions'] = member_contri[tm]
    team_profiles.insert_one(t_ps[tm])
    cnt += 1
