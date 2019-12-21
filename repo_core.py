import json
from pymongo import MongoClient


client = MongoClient()
db = client['gtr']
repo_teams = db['repo_teams']
repos = set(repo_teams.distinct('repo'))


def sort_to_k(ary,k,key=lambda x:x,reversed=False):
    k = min(k,len(ary))
    for i in range(k):
        for j in range(len(ary)-1-i):
            if not reversed:
                if key(ary[len(ary)-1-j]) < key(ary[len(ary)-2-j]):
                    ary[len(ary)-1-j],ary[len(ary)-2-j] = ary[len(ary)-2-j],ary[len(ary)-1-j]
            else:
                if key(ary[len(ary)-1-j]) > key(ary[len(ary)-2-j]):
                    ary[len(ary)-1-j],ary[len(ary)-2-j] = ary[len(ary)-2-j],ary[len(ary)-1-j]
    return ary


db.drop_collection('repo_core_targets')
r_ct = db['repo_core_targets']
cnt = 0
with open("contributors.json") as rj:
    for l in rj.readlines():
        j = json.loads(l)
        repo = j['repo']
        if not repo in repos:
            continue
        print(cnt)
        user_contri = {}
        for c in j['contributors']:
            user_contri[c['login']] = c['contributions']
        kk = len(user_contri)//10
        cntr = sort_to_k(list(user_contri),key = lambda x: user_contri[x], reversed=True,k=kk)
        repo_core = set(cntr[:kk])
        repo_target = []
        repo_core_team = []
        for tm in repo_teams.find_one({'repo':repo})['teams']:
            is_target = False
            for mem in json.loads(tm):
                if not mem in repo_core:
                    is_target = True
                    break
            if is_target:
                repo_target.append(tm)
            else:
                repo_core_team.append(tm)
        if repo_target:
            r_ct.insert_one({
                'repo':repo,
                'core':cntr[:kk],
                'core_team':repo_core_team,
                'targets':repo_target
            })
        
        cnt += 1