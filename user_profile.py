import json
from pymongo import MongoClient

client = MongoClient()
db = client['gtr']
team_profiles = db['team_profiles']

repo_core_targets = db['repo_core_targets']
user_contri_repo = {}
# user_contri_repo_train = {}
for row in repo_core_targets.find():
    repo = row['repo']
    for user in row['core_users']:
        if not user in user_contri_repo:
            user_contri_repo[user] = {}
        # if not user in user_contri_repo_train:
        #     user_contri_repo_train[user] = {}
        user_contri_repo[user][repo] = row['core_users'][user]
        # user_contri_repo_train[user][repo] = row['core_users'][user]
    for user in row['target_users']:
        if not user in user_contri_repo:
            user_contri_repo[user] = {}
        user_contri_repo[user][repo] = row['target_users'][user]

r_ps = db['repo_profiles']
repo_profiles = {}
for doc in r_ps.find():
    repo_profiles[doc['repo']] = doc


def user_feature(user_repos,repo_profiles):
    topics = set()
    languages = set()
    sizes = []
    forks = []
    watchers = []
    subscribers = []
    for repo in user_repos:
        r_p = repo_profiles[repo]
        if 'topics' in r_p:
            topics.update(r_p['topics'])
        if 'languages' in r_p:
            languages.update(r_p['languages'])
        sizes.append(r_p['size'])
        forks.append(r_p['forks'])
        watchers.append(r_p['watchers'])
        subscribers.append(r_p['subscribers'])
    repo_size = sum(sizes)/len(sizes)
    repo_forks = sum(forks)/len(forks)
    repo_watchers = sum(watchers)/len(watchers)
    repo_subscribers = sum(subscribers)/len(subscribers)
    profile = {
        'topics':list(topics),
        'languages':list(languages),
        'repo_size':repo_size,
        'repo_forks':repo_forks,
        'repo_watchers':repo_watchers,
        'repo_subscribers':repo_subscribers
    }
    return profile


db.drop_collection('user_profiles')
user_profiles = db['user_profiles']
cnt = 0
for user in user_contri_repo:
    print(cnt)
    profile = user_feature(user_contri_repo[user],repo_profiles)
    profile['user'] = user
    user_profiles.insert_one(profile)

    cnt += 1

# db.drop_collection('user_profiles_train')
# user_profiles = db['user_profiles_train']
# cnt = 0
# for user in user_contri_repo_train:
#     print(cnt)
#     profile = user_feature(user_contri_repo_train[user],repo_profiles)
#     profile['user'] = user
#     user_profiles.insert_one(profile)

#     cnt += 1