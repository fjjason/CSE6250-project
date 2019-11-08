# CSE6250-project

This is the repository for 2019 Fall CSE6250 project.

# Usage

Generate Plots & CSV of Channels and Labels

> `python program/eda.py`

Transform Data

> `python program/data_prep_v0.py`

Train Model

> `python program/model_v0.py`

# Description of Parameters & Labels

...

# Conda

## tldr;

> `conda env create -f environment.yml`

> `conda activate cse6250-project-sleep`

## setup steps:

environment originally created via:

> `conda create --name cse6250-project-sleep`

activate the new environment:

> `conda activate cse6250-project-sleep`

copied a base environment.yml and then installed packages individually:

> `conda install <package>`

now, environment can be installed from `environment.yml` via

> `conda env create -f environment.yml`
