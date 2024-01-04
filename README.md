# TNNLS2023 - MG-SIN: Multi-Graph Sparse Interaction Network for Multi-Task Stance Detection
This project involves the code and supplementary materials of paper "MG-SIN: Multi-Graph Sparse Interaction Network for Multi-Task Stance Detection".

# Dependencies
* pytorch == 1.9.0
* numpy == 1.20.1
* scikit-learn == 0.24.1
* transformer == 4.12.3
* scipy == 1.6.2
* spacy == 3.2.0
* digitalepidemiologylab/covid-twitter-bert-v2-mnli


# Run
Running MG-SIN is followed as:

    python -W ignore::RuntimeWarning run.py

If you want to switch datasets, modify the index of "target_list" in run.py

# Cite
This is the code of the TNNLS 2023 paper "MG-SIN: Multi-Graph Sparse Interaction Network for Multi-Task Stance Detection". **Please cite our paper if you use the code**:
```
@ARTICLE{Chai2023MGSIN,
  author={Chai, Heyan and Cui, Jinhao and Tang, Siyu and Ding, Ye and Liu, Xinwang and Fang, Binxing and Liao, Qing},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={MG-SIN: Multigraph Sparse Interaction Network for Multitask Stance Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2023.3328659}}
```
