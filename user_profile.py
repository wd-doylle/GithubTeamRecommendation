import json
from pymongo import MongoClient

client = MongoClient()
db = client['gtr']
team_profiles = db['team_profiles']

user_repo = {}
for tm in team_profiles.distinct('team'):
    mems = json.loads(tm)
    for user in mems:
        user_repo[user] = set()

with open('contributors.json') as tmj:
    for tml in tmj.readlines():
        line = json.loads(tml)
        for user in line['contributors']:
            if not user['login'] in user_repo:
                continue
            user_repo[user['login']].add(line['repo'])

repo_profiles = db['repo_profiles']
r_ps = {}
for doc in repo_profiles.find():
    r_ps[doc['repo']] = doc
db.drop_collection('user_profiles')
user_profiles = db['user_profiles']
cnt = 0

with open("user_profiles.json",'w') as uf:
    for user in user_repo:
        print(cnt)
        topics = set()
        languages = set()
        sizes = []
        forks = []
        watchers = []
        subscribers = []
        for repo in user_repo[user]:
            if not repo in r_ps:
                continue
            r_p = r_ps[repo]
            if 'topics' in r_p:
                topics.update(r_p['topics'])
            if 'languages' in r_p:
                languages.update(r_p['languages'])
            sizes.append(r_p['size'])
            forks.append(r_p['forks'])
            watchers.append(r_p['watchers'])
            subscribers.append(r_p['subscribers'])
        repo_size = sum(sizes)/len(sizes)
        repo_forks = sum(forks)/len(forks)
        repo_watchers = sum(watchers)/len(watchers)
        repo_subscribers = sum(subscribers)/len(subscribers)
        user_profiles.insert_one({
            'user':user,
            'topics':list(topics),
            'languages':list(languages),
            'repo_size':repo_size,
            'repo_forks':repo_forks,
            'repo_watchers':repo_watchers,
            'repo_subscribers':repo_subscribers,
        })
        
        uf.write("%s\t%s\n"%(user,json.dumps({
            'topics':list(topics),
            'languages':list(languages),
            'repo_size':repo_size,
            'repo_forks':repo_forks,
            'repo_watchers':repo_watchers,
            'repo_subscribers':repo_subscribers,
            })))

        cnt += 1