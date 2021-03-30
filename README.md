# GithubTeamRecommendation
## Prerequisites
- Python >= 3.5.1
- Mongodb Server >= 3.4.24
- CUDA >= 9.2 for training on gpu
- Jupyter Notebook for visualizing results
- Python packages: pymongo, pytorch, pandas
## How to build dataset
```sh
# Build dataset in local mongodb
$ python3 repo_profile.py
$ python3 repo_core.py
$ python3 team_profile.py
$ python3 user_profile.py
$ python3 build_dataset.py

# If you want to export json files
$ mongoexport --collection=repo_profiles --db=gtr --out=repo_profiles.json
$ mongoexport --collection=team_profiles --db=gtr --out=team_profiles.json
$ mongoexport --collection=user_profiles --db=gtr --out=user_profiles.json
```
## How to run experiments
### GAT
```sh
$ python3 GAT/main.py --fast-mode --epochs=5

# To try alternative architectures
$ python3 GAT/main.py --fast-mode --alt=1 --epochs=5
$ python3 GAT/main.py --fast-mode --alt=2 --epochs=5
```
### SoAGREE
```sh
$ $ python3 soagree/main_soagree.py --epochs=5
```
### Euclidean
```sh
$ python3 recommend_euclidean.py
```
### Social
```sh
$ python3 recommend_social_core.py
```
## How to visualize results
```sh
$ ./ind2name.sh
$ jupyter notebook evaluate.ipynb
```
