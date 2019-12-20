import json
import os
import re

teams = []
team_inds = {}
user_team = {}
with open('team_profiles.json') as tj:
    for l in tj.readlines():
        line = l.strip().split('\t')
        teams.append(line[0])
        for user in json.loads(line[0]):
            if not user in user_team:
                user_team[user] = []
            user_team[user].append(len(teams)-1)
        team_inds[line[0]] = len(teams)-1

repo_team = {}
team_graph = {}
cnt = 0
with open("contributors.json") as rj:
    for l in rj.readlines():
        print(cnt)
        contributors = [c['login'] for c in json.loads(l)['contributors']]
        intersect = list(set(contributors).intersection(user_team))
        for i,u in enumerate(intersect):
            for team in user_team[u]:
                if not team in team_graph:
                    team_graph[team] = {}
                for j in range(i+1,len(intersect)):
                    for tm in user_team[intersect[j]]:
                        if tm == team:
                            continue
                        if not tm in team_graph[team]:
                            team_graph[team][tm] = 0
                        team_graph[team][tm] += 1
        cnt += 1


with open('team_graph.json','w') as rp:
    for i,team in enumerate(teams):
        g = team_graph[i] if i in team_graph else {}
        rp.write("%s\t%s\n"%(team,json.dumps(g)))