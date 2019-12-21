import json
from pymongo import MongoClient


client = MongoClient()
db = client['gtr']

repo_profiles = {}
with open('repo_features.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        features = json.loads(line[1])
        repo_profiles[repo] = features
        repo_profiles[repo]['subscribers'] = repo_profiles[repo].pop('subscribers_count')
with open('repo_language.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        if not repo in repo_profiles:
            continue
        languages = list(json.loads(line[1]).keys())
        repo_profiles[repo]['languages'] = languages
with open('repo_topics.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        if not repo in repo_profiles:
            continue
        topics = json.loads(line[1])
        repo_profiles[repo]['topics'] = topics

db.drop_collection('repo_profiles')
r_ps = db['repo_profiles']
cnt = 0
for repo in repo_profiles:
    print(cnt)
    profile = repo_profiles[repo]
    profile['repo'] = repo
    r_ps.insert_one(profile)
    cnt += 1

repo_teams = {}
with open('team_tags.txt') as tmj:
    for tml in tmj.readlines():
        tm,dur,topics,lang,contr,center,aspl,ac,cen,sizes,repo_contributors,lang_diff,topic_diff,size_diff,wtch_diff,fork_diff,sbscrb_diff,feature_diff = tml.split('\t')
        # dur = int(dur)
        # aspl = json.loads(aspl)['all']
        # ac = json.loads(ac)['all']
        # cen = json.loads(cen)['all']
        # size = len(tm)
        for repo in json.loads(sizes):
            if not repo in repo_teams:
                repo_teams[repo] = []
            repo_teams[repo].append(tm)

db.drop_collection('repo_teams')
hr_ps = db['repo_teams']
cnt = 0
for repo in repo_teams:
    print(cnt)
    hr_ps.insert_one({
        'repo':repo,
        'teams':repo_teams[repo]
    })
    cnt += 1