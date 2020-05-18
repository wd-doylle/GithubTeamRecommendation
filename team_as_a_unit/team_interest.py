import json
import networkx as nx
from pymongo import MongoClient

client = MongoClient()
db = client['gtr']

team_profiles = db['team_profiles']
team_users = {}
for doc in team_profiles.find():
        tm = doc['team']
        team_users[tm] = []
        for user in json.loads(tm):
            team_users[tm].append(user)

user_interest = {}
user_profiles = db['user_profiles']
for doc in user_profiles.find():
    user = doc.pop('user')
    user_interest[user] = doc

with open('team_interest.json','w') as oj:
    for i,tm in enumerate(team_users):
        print(i)
        team_interest = {'team':tm}
        repo_size = []
        repo_forks = []
        repo_subscribers = []
        repo_watchers = []
        languages = set()
        topics = set()
        for user in team_users[tm]:
            if not user in user_interest:
                continue
            repo_size.append(user_interest[user]['repo_size'])
            repo_forks.append(user_interest[user]['repo_forks'])
            repo_subscribers.append(user_interest[user]['repo_subscribers'])
            repo_watchers.append(user_interest[user]['repo_watchers'])
            languages.update(user_interest[user]['languages'])
            topics.update(user_interest[user]['topics'])
        if not repo_size:
            continue
        team_interest['repo_size'] = sum(repo_size)/len(repo_size)
        team_interest['repo_forks'] = sum(repo_forks)/len(repo_forks)
        team_interest['repo_subscribers'] = sum(repo_subscribers)/len(repo_subscribers)
        team_interest['repo_watchers'] = sum(repo_watchers)/len(repo_watchers)
        team_interest['languages'] = list(languages)
        team_interest['topics'] = list(topics)
        oj.write(json.dumps(team_interest)+'\n')
