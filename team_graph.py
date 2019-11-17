import json
import os
import re

teams = []
team_inds = {}
with open('team_profiles.json') as tj:
    for l in tj.readlines():
        line = l.strip().split('\t')
        teams.append(line[0])
        team_inds[line[0]] = len(teams)-1

repo_team = {}
with open("repo_profiles_new.json") as rj:
    for l in rj.readlines():
        line = l.strip().split('\t')
        tms = json.loads(line[1])['teams']
        repo_team[line[0]] = tms

team_graph = {}
for repo in repo_team:
    for i,team in enumerate(repo_team[repo]):
        for j in range(i+1,len(repo_team[repo])):
            if not team in team_graph:
                team_graph[team] = {}
            if not team_inds[repo_team[repo][j]] in team_graph[team]:
                team_graph[team][team_inds[repo_team[repo][j]]] = 0
            team_graph[team][team_inds[repo_team[repo][j]]] += 1


with open('team_graph.json','w') as rp:
    for team in teams:
        g = team_graph[team] if team in team_graph else {}
        rp.write("%s\t%s\n"%(team,json.dumps(g)))