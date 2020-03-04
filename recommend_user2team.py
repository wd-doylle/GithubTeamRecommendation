import json
import sys

user_score_file = sys.argv[1]
output_file = sys.argv[2]


users = []
with open("user_profiles.json") as uj:
    for l in uj.readlines():
        line = json.loads(l)
        users.append(line['user'])

repos = []
with open("repo_profiles.json") as uj:
    for l in uj.readlines():
        line = json.loads(l)
        repos.append(line['repo'])

user_weights = {}
with open('team_profiles.json') as rj:
    for l in rj.readlines():
        doc = json.loads(l)
        tm = doc['team']
        for user in json.loads(tm):
            if not user in user_weights:
                user_weights[user] = {}
            contris = sum(doc['member_contributions'].values())
            degrees = sum(doc['member_degrees'].values())
            user_weights[user][tm] = {
                'contri':doc['member_contributions'][user]/contris,
                'degree':doc['member_degrees'][user]/degrees
            }

repo_recs = {}
repo_recs_contri = {}
repo_recs_degree = {}
with open(user_score_file) as uf:
    for l in uf.readlines():
        line = l.strip().split('\t')
        user = users[int(line[0])]
        # user = int(line[0])
        # uu = users[user]
        if not user in user_weights:
            continue
        recs = [json.loads(s) for s in line[1:]]
        for repo,interest in recs:
            repo = repos[repo]
            if not repo in repo_recs:
                repo_recs[repo] = {}
                repo_recs_contri[repo] = {}
                repo_recs_degree[repo] = {}
            for team in user_weights[user]:
                if not team in repo_recs[repo]:
                    repo_recs[repo][team] = []
                    repo_recs_contri[repo][team] = 0
                    repo_recs_degree[repo][team] = 0
                repo_recs[repo][team].append(interest)
                repo_recs_contri[repo][team] += user_weights[user][team]['contri']*interest
                repo_recs_degree[repo][team] += user_weights[user][team]['degree']*interest


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


k = 30

f_min = open(output_file+"_min.json",'w')
f_max = open(output_file+"_max.json",'w')
f_mean = open(output_file+"_mean.json",'w')
f_contri = open(output_file+"_contri.json",'w')
f_degree = open(output_file+"_degree.json",'w')
cnt = 0
for repo in repo_recs:
    print(cnt)
    team_score_min = {}
    team_score_mean = {}
    team_score_max = {}
    for tm in repo_recs[repo]:
        team_score_min[tm] = min(repo_recs[repo][tm])
        team_score_max[tm] = max(repo_recs[repo][tm])
        team_score_mean[tm] = sum(repo_recs[repo][tm])/len(repo_recs[repo][tm])
    tms_min = sort_to_k(list(team_score_min),k,key=lambda i:team_score_min[i])
    tms_mean = sort_to_k(list(team_score_mean),k,key=lambda i:team_score_mean[i])
    tms_max = sort_to_k(list(team_score_min),k,key=lambda i:team_score_max[i])
    tms_contri = sort_to_k(list(team_score_min),k,key=lambda i:repo_recs_contri[repo][i])
    tms_degree = sort_to_k(list(team_score_min),k,key=lambda i:repo_recs_degree[repo][i])


    f_min.write(repo)
    for tm in tms_min[:k]:
        f_min.write("\t%s"%(json.dumps((tm,float(team_score_min[tm])))))
    f_min.write('\n')

    f_max.write(repo)
    for tm in tms_max[:k]:
        f_max.write("\t%s"%(json.dumps((tm,float(team_score_max[tm])))))
    f_max.write('\n')
    
    f_mean.write(repo)
    for tm in tms_mean[:k]:
        f_mean.write("\t%s"%(json.dumps((tm,float(team_score_mean[tm])))))
    f_mean.write('\n')
    
    f_contri.write(repo)
    for tm in tms_contri[:k]:
        f_contri.write("\t%s"%(json.dumps((tm,float(repo_recs_contri[repo][tm])))))
    f_contri.write('\n')

    f_degree.write(repo)
    for tm in tms_degree[:k]:
        f_degree.write("\t%s"%(json.dumps((tm,float(repo_recs_degree[repo][tm])))))
    f_degree.write('\n')
    
    cnt += 1