import json


repo_profiles = {}
with open('repo_features.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        feature = json.loads(line[1])
        repo_profiles[repo] = feature
with open('repo_language.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        if not repo in repo_profiles:
            continue
        language = list(json.loads(line[1]).keys())
        repo_profiles[repo]['language'] = language
with open('repo_topics.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        if not repo in repo_profiles:
            continue
        topics = json.loads(line[1])
        repo_profiles[repo]['topics'] = topics

with open('repo_profiles.dat','w') as of:
    for repo in repo_profiles:
        of.write('%s\t%s\n'%(repo,repo_profiles[repo]))