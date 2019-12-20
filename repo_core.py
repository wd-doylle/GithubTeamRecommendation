import json

repos = []
with open('repo_profiles_new.json') as rj:
    for rl in rj.readlines():
        line = rl.strip().split('\t')
        repos.append(line[0])

teams = []
user_team_ind = {}
repo_team_ind = {}
cnt = 0
with open('team_tags.txt') as tmj:
    for tml in tmj.readlines():
        tm,dur,topics,lang,contr,center,aspl,ac,cen,sizes,repo_contributors,lang_diff,topic_diff,size_diff,wtch_diff,fork_diff,sbscrb_diff,feature_diff = tml.split('\t')
        teams.append(tm)
        rps = list(json.loads(sizes).keys())
        for repo in rps:
            if not repo in repo_team_ind:
                repo_team_ind[repo] = []
            repo_team_ind[repo].append(cnt)
        for user in json.loads(tm):
            if not user in user_team_ind:
                user_team_ind[user] = []
            user_team_ind[user].append(cnt)
        cnt += 1



repo_core = {}
repo_core_team_ind = {}
with open('repo_target_core.json','w') as tj:
    with open("contributors.json") as rj:
        for l in rj.readlines():
            j = json.loads(l)
            repo = j['repo']
            if not repo in repos:
                continue
            contribution = {}
            for c in j['contributors']:
                contribution[c['login']] = c['contributions']
            t_c = {}
            # repo_team_mem = set()
            for tm in repo_team_ind[repo]:
                t_c[tm] = 0
                team = json.loads(teams[tm])
                # repo_team_mem.update(team)
                for member in team:
                    if member in contribution:
                        t_c[tm] += contribution[member]
            if len(repo_team_ind[repo])>1:
                core_team_ind = max(t_c,key=lambda x: t_c[x])
                repo_core[repo] = set(json.loads(teams[core_team_ind]))
            else:
                core_team_ind = -1
                repo_core[repo] = set()
            repo_core_team_ind[repo] = core_team_ind
            tj.write(repo)
            for tm in repo_team_ind[repo]:
                if tm == repo_core_team_ind[repo]:
                    continue
                tj.write('\t%s'%(teams[tm]))
            tj.write('\n')
            kk = len(contribution)//10
            cntr = sort_to_k(list(contribution),key = lambda x: contribution[x], reversed=True,k=kk)
            for i in range(kk):
                # if cntr[i] in repo_team_mem:
                #     continue
                repo_core[repo].add(cntr[i])