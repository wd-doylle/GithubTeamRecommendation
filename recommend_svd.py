import json
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import train_test_split
from queue import PriorityQueue

ratings = []
repos = []
teams = set()
repo_teams = {}
with open('repo_profiles_new.json') as rj:
    for rl in rj.readlines():
        line = rl.split('\t')
        repo = line[0]
        profile = json.loads(line[1])
        repos.append(repo)
        teams.update(profile['teams'])
        repo_teams[repo] = set(profile['teams'])
        for team in profile['teams']:
            ratings.append([repo,team,1])


reader = Reader(rating_scale=(0,1))
dataset = Dataset.load_from_df(pd.DataFrame(ratings),reader)
trainset, testset = train_test_split(dataset, test_size=.25)
algo = SVD()
full_data = dataset.build_full_trainset()
algo.fit(full_data)

with open('recommend_svd.json','w') as oj:
    cnt = 0
    for repo in repos:
        cnt += 1
        # if cnt > 500:
        #     break
        print(cnt)
        rec = []
        queue = PriorityQueue()
        for team in teams:
            if team in repo_teams[repo]:
                continue
            p = algo.predict(repo,team)
            queue.put_nowait((p[3],team))
            if queue.qsize() > 50:
                queue.get_nowait()
        while queue.qsize()>0:
            rec.append(queue.get_nowait())
        oj.write(repo+'\t')
        for tm in rec:
            oj.write(json.dumps(tm)+'\t')
        oj.write('\n')