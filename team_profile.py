import json


repo_features = {}
with open('repo_features.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        features = json.loads(line[1])
        repo_features[repo] = features
team_profiles = []
with open('team_tags.txt') as tmj:
    for tml in tmj.readlines():
        tml.split('\t')
        tm,dur,topics,lang,contr,center,aspl,ac,cen,sizes,repo_contributors,lang_diff,topic_diff,size_diff,wtch_diff,fork_diff,sbscrb_diff,feature_diff = tml.split('\t')
        tm = json.loads(tm)
        # dur = int(dur)
        topics = json.loads(topics)
        lang = json.loads(lang)
        # aspl = json.loads(aspl)['all']
        # ac = json.loads(ac)['all']
        # cen = json.loads(cen)['all']
        # size = len(tm)
        repos = list(json.loads(sizes).keys())
        repo_size = [repo_features[repo]['size'] for repo in repos]
        repo_size = sum(repo_size)/len(repo_size)
        repo_forks = [repo_features[repo]['forks'] for repo in repos]
        repo_forks = sum(repo_forks)/len(repo_forks)
        repo_watchers = [repo_features[repo]['watchers'] for repo in repos]
        repo_watchers = sum(repo_watchers)/len(repo_watchers)
        repo_subscribers = [repo_features[repo]['subscribers_count'] for repo in repos]
        repo_subscribers = sum(repo_subscribers)/len(repo_subscribers)
        team_profiles.append([tm,{
            'topics':topics,
            'languages':lang,
            'repo_size':repo_size,
            'repo_forks':repo_forks,
            'repo_watchers':repo_watchers,
            'repo_subscribers':repo_subscribers,
        }])


with open('team_profiles.dat','w') as of:
    for p in team_profiles:
        of.write('%s\t%s\n'%(json.dumps(p[0]),json.dumps(p[1])))