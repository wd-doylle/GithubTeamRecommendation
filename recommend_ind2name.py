import json
import sys
from pymongo import MongoClient

client = MongoClient()
db = client['gtr']

team_score_file = sys.argv[1]
output_file = sys.argv[2]

repo_core_targets = db['repo_core_targets']
repos = []
for row in repo_core_targets.find():
    repos.append(row['repo'])

team_profiles = db['team_profiles']
teams = []
for row in team_profiles.find():
    teams.append(row['team'])

cnt = 0
with open(output_file,'w') as of:
    with open(team_score_file) as uf:
        for l in uf.readlines():
            line = l.strip().split('\t')
            repo = repos[int(line[0])]
            recs = [json.loads(r) for r in line[1:]]
            of.write(repo)
            for tm,score in recs:
                of.write("\t%s"%(json.dumps((teams[tm],score))))
            of.write('\n')
            print(cnt)
            cnt += 1