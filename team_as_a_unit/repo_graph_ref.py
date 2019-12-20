import json
import os
import re

repos = []
repo_inds = {}
with open("repo_profiles_new.json") as rj:
    for l in rj.readlines():
        line = l.strip().split('\t')
        repos.append(line[0])
        repo_inds[line[0]] = len(repos)-1

year = 0
month = 0
day = 0

issuecomment_dir = r"D:\\GithubGroupDetection\\issuecomment\\"

repo_graph = {}
years = os.listdir(issuecomment_dir)
months = os.listdir(issuecomment_dir+years[year])
days = os.listdir(issuecomment_dir+years[year]+'/'+months[month])
repo_detials = []
prog = re.compile(r"([0-9A-Za-z\-]+/[0-9A-Za-z\-]+)(@a-f0-9]{7}|#[0-9]+)")
for y in range(year,len(years)):
    months = os.listdir(issuecomment_dir+years[y])
    for m in range(month,len(months)):
        days = os.listdir(issuecomment_dir+years[y]+'/'+months[m])
        for d in range(day,len(days)):
            with open(issuecomment_dir+years[y]+'/'+months[m]+'/'+days[d]) as ic:
                for line in ic.readlines():
                    j = json.loads(line)
                    repo = j['repo_name']
                    if not 'body' in j or not j['body']:
                        continue
                    match = prog.findall(j['body'])
                    if not match:
                        continue
                    for g in match:
                        if g[0] in repo_inds:
                            if not repo in repo_graph:
                                repo_graph[repo] = {}
                            if not repo_inds[g[0]] in repo_graph[repo]:
                                repo_graph[repo][repo_inds[g[0]]] = 0
                            repo_graph[repo][repo_inds[g[0]]] += 1
                            # print(repo,g[0])
            print(y,m,d)



with open('repo_graph_ref.json','w') as rp:
    for repo in repos:
        g = repo_graph[repo] if repo in repo_graph else {}
        rp.write("%s\t%s\n"%(repo,json.dumps(g)))