import json

repo_score = {}
repo_freq = {}
with open("../user_interests_train.json") as sj:
    for l in sj.readlines():
        line = json.loads(l)
        for repo in line['interests']:
            if not repo in repo_score:
                repo_score[repo] = 0
                repo_freq[repo] = 0
            repo_score[repo] += line['interests'][repo]
            repo_freq[repo] += 1

for rec in sorted(repo_score,key=lambda x:repo_score[x],reverse=True)[:50]:
    print(rec,repo_score[rec])

print("...")

for rec in sorted(repo_freq,key=lambda x:repo_freq[x],reverse=True)[:50]:
    print(rec,repo_freq[rec])