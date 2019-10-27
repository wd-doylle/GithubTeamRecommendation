import json

user_repo = {}
with open('team_tags.txt') as tmj:
    for tml in tmj.readlines():
        tml.split('\t')
        tm,dur,topics,lang,contr,center,aspl,ac,cen,sizes,repo_contributors,lang_diff,topic_diff,size_diff,wtch_diff,fork_diff,sbscrb_diff,feature_diff = tml.split('\t')
        for user in json.loads(tm):
            user_repo[user] = set()
with open('contributors.json') as tmj:
    for tml in tmj.readlines():
        line = json.loads(tml)
        for user in line['contributors']:
            if not user['login'] in user_repo:
                continue
            user_repo[user['login']].add(line['repo'])

repo_profiles = {}
with open('repo_profiles.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        profiles = json.loads(line[1])
        repo_profiles[repo] = profiles

user_profiles = []
for user in user_repo:
    topics = set()
    languages = set()
    sizes = []
    forks = []
    watchers = []
    subscribers = []
    for repo in user_repo[user]:
        if not repo in repo_profiles:
            continue
        if 'topics' in repo_profiles[repo]:
            topics.update(repo_profiles[repo]['topics'])
        if 'languages' in repo_profiles[repo]:
            languages.update(repo_profiles[repo]['languages'])
        sizes.append(repo_profiles[repo]['size'])
        repo_size = sum(sizes)/len(sizes)
        forks.append(repo_profiles[repo]['forks'])
        repo_forks = sum(forks)/len(forks)
        watchers.append(repo_profiles[repo]['watchers'])
        repo_watchers = sum(watchers)/len(watchers)
        subscribers.append(repo_profiles[repo]['subscribers'])
        repo_subscribers = sum(subscribers)/len(subscribers)
        user_profiles.append([tm,{
            'topics':list(topics),
            'languages':list(languages),
            'repo_size':repo_size,
            'repo_forks':repo_forks,
            'repo_watchers':repo_watchers,
            'repo_subscribers':repo_subscribers,
        }])

with open('user_profiles.json','w') as of:
    for p in user_profiles:
        of.write('%s\t%s\n'%(p[0],json.dumps(p[1])))