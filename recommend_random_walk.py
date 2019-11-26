import json

rec_sim = {}
with open('recommend_random_walk_sim.json') as rj:
    for l in rj.readlines():
        line = l.strip().split('\t')
        repo = line[0]
        recs = [json.loads(l) for l in line[1:]]
        rec_sim[repo] = recs

rec_ref = {}
with open('recommend_random_walk_ref.json') as rj:
    for l in rj.readlines():
        line = l.strip().split('\t')
        repo = line[0]
        recs = [json.loads(l) for l in line[1:]]
        rec_ref[repo] = recs
        
rec_social = {}
with open('recommend_random_walk_social.json') as rj:
    for l in rj.readlines():
        line = l.strip().split('\t')
        repo = line[0]
        recs = [json.loads(l) for l in line[1:]]
        rec_social[repo] = recs

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

k = 50
cnt = 0
with open('recommend_random_walk.json','w') as rj:
    for repo in rec_sim:
        print(cnt)
        team_point = {}
        for rec in rec_sim[repo]:
            if not rec[0] in team_point:
                team_point[rec[0]] = 0
            team_point[rec[0]] += rec[1]
        for rec in rec_ref[repo]:
            if not rec[0] in team_point:
                team_point[rec[0]] = 0
            team_point[rec[0]] += rec[1]
        for rec in rec_social[repo]:
            if not rec[0] in team_point:
                team_point[rec[0]] = 0
            team_point[rec[0]] += rec[1]
        teams = sort_to_k(list(team_point.keys()),k,key=lambda x:team_point[x],reversed=True)
        rj.write(repo)
        for i in range(k):
            rj.write("\t%s"%(json.dumps((teams[i],team_point[teams[i]]))))
        rj.write('\n')
        cnt += 1