# CSE6250-project

This is the repository for 2019 Fall CSE6250 project.

# Usage

Generate Plots & CSV of Channels and Labels

`python program/eda.py`

Transform Data

`python program/data_prep_v1.py`

Train Model

`python program/model_v1.py`

# Description of Parameters & Labels

16 features are computed. Each feature represents the bandpower of bands present in the patient's electrical channels. Each feature is computed for each band over a 1 hour interval over each channel.

Labels represent whether the channel sample indicates poor sleep. Label of 1 if the patient had a mild difficulty of falling asleep, otherwise the label is 0. These labels correspond to a set of 16 bandpower features mentioned.

# Conda Environment

```
conda env create -f environment.yml
conda activate cse6250-project-sleep
```
