import json

repos = set()
with open('team_tags.txt') as tmj:
    for tml in tmj.readlines():
        tml.split('\t')
        tm,dur,topics,lang,contr,center,aspl,ac,cen,sizes,repo_contributors,lang_diff,topic_diff,size_diff,wtch_diff,fork_diff,sbscrb_diff,feature_diff = tml.split('\t')
        repos.update(json.loads(sizes).keys())

with open('repos.txt','w') as rt:
    for repo in repos:
        rt.write(repo+'\n')