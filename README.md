# MG-SIN
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
