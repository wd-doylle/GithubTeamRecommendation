import json


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

with open('repo_profiles.json','w') as of:
    for repo in repo_profiles:
        of.write('%s\t%s\n'%(repo,json.dumps(repo_profiles[repo])))