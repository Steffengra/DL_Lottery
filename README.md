
The code presented was used in the following publication
[(preprint here)](https://arxiv.org/abs/2304.10176).

[1] S. Gracla, E. Beck, C. Bockelmann and A. Dekorsy,
"Robust Deep Reinforcement Learning Scheduling via Weight Anchoring",
in *IEEE Communications Letters*,
2022, doi: [10.1109/LCOMM.2022.3214574](https://doi.org/10.1109/LCOMM.2022.3214574).

Email: {**gracla**, beck, bockelmann, dekorsy}@ant.uni-bremen.de

The project structure is as follows:

```
/project/
├─ DL_Lottery_imports/                      | python modules
├─ .gitignore                               | .gitignore
├─ config.py                                | contains configurable parameters
├─ requirements.txt                         | project dependencies
├─ runner.py                                | orchestrates training & testing
├─ test_anchoring_critical_allocation.py    | wrapper for training & testing different configurations
├─ test_anchoring_random_training.py        | wrapper for random baseline
```