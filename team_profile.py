import json


repo_profiles = {}
with open('repo_profiles.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        profiles = json.loads(line[1])
        repo_profiles[repo] = profiles
team_profiles = []
repo_profiles_new = {}
repo_teams = {}
with open('team_tags.txt') as tmj:
    for tml in tmj.readlines():
        tml.split('\t')
        tm,dur,topics,lang,contr,center,aspl,ac,cen,sizes,repo_contributors,lang_diff,topic_diff,size_diff,wtch_diff,fork_diff,sbscrb_diff,feature_diff = tml.split('\t')
        # dur = int(dur)
        # aspl = json.loads(aspl)['all']
        # ac = json.loads(ac)['all']
        # cen = json.loads(cen)['all']
        # size = len(tm)
        repos = list(json.loads(sizes).keys())
        target = repos[-1]
        repos = repos[:-1]
        topics = set()
        languages = set()
        for repo in repos:
            if not repo in repo_teams:
                repo_teams[repo] = []
            repo_teams[repo].append(tm)
            if 'topics' in repo_profiles[repo]:
                topics = topics.union(repo_profiles[repo]['topics'])
            if 'languages' in repo_profiles[repo]:
                languages = languages.union(repo_profiles[repo]['languages'])
        repo_size = [repo_profiles[repo]['size'] for repo in repos]
        repo_size = sum(repo_size)/len(repo_size)
        repo_forks = [repo_profiles[repo]['forks'] for repo in repos]
        repo_forks = sum(repo_forks)/len(repo_forks)
        repo_watchers = [repo_profiles[repo]['watchers'] for repo in repos]
        repo_watchers = sum(repo_watchers)/len(repo_watchers)
        repo_subscribers = [repo_profiles[repo]['subscribers'] for repo in repos]
        repo_subscribers = sum(repo_subscribers)/len(repo_subscribers)
        team_profiles.append([tm,{
            'topics':list(topics),
            'languages':list(languages),
            'repo_size':repo_size,
            'repo_forks':repo_forks,
            'repo_watchers':repo_watchers,
            'repo_subscribers':repo_subscribers,
            'target':target
        }])
        if not target in repo_profiles_new:
            repo_profiles_new[target] = repo_profiles[target]


with open('team_profiles.json','w') as of:
    for p in team_profiles:
        of.write('%s\t%s\n'%(p[0],json.dumps(p[1])))

for repo in repo_profiles_new:
    repo_profiles_new[repo]['teams'] = []
    if repo in repo_teams:
        repo_profiles_new[repo]['teams'].extend(repo_teams[repo])

with open('repo_profiles.json','w') as of:
    for repo in repo_profiles_new:
        of.write('%s\t%s\n'%(repo,json.dumps(repo_profiles[repo])))