import json
from pymongo import MongoClient


client = MongoClient()
db = client['gtr']
r_t = db['repo_teams']
repo_teams = {}
for row in r_t.find():
    repo_teams[row['repo']] = row['teams']

users = set()
with open("network.json") as nj:
    j = json.load(nj)
    nodes = j['nodes']
    for n in nodes:
        users.add(n)

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


split_ratio = 0.2
db.drop_collection('repo_core_targets')
r_ct = db['repo_core_targets']
cnt = 0
with open("contributors.json") as rj:
    for l in rj.readlines():
        j = json.loads(l)
        repo = j['repo']
        if not repo in repo_teams:
            continue
        print(cnt)
        user_contri = {}
        for c in j['contributors']:
            if not c['login'] in users:
                continue
            user_contri[c['login']] = c['contributions']
        if not user_contri:
            continue
        kk = len(user_contri) - int(len(user_contri)*split_ratio)
        cntr = sort_to_k(list(user_contri),key = lambda x: user_contri[x], reversed=True,k=kk)
        repo_core = set(cntr[:kk])
        repo_target = []
        repo_core_team = []
        for tm in repo_teams[repo]:
            is_target = False
            for mem in json.loads(tm):
                if not mem in repo_core:
                    is_target = True
                    break
            if is_target:
                repo_target.append(tm)
            else:
                repo_core_team.append(tm)
        r_ct.insert_one({
            'repo':repo,
            'core_users':{c:user_contri[c] for c in cntr[:kk]},
            'core_teams':repo_core_team,
            'target_teams':repo_target,
            'target_users':{c:user_contri[c] for c in cntr[kk:]}
        })
        
        cnt += 1