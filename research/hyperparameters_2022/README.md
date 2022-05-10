# Hyperparameter Tuning with Renyi Differential Privacy

### Nicolas Papernot and Thomas Steinke

This repository contains the code used to reproduce some of the experiments in
our
[ICLR 2022 paper on hyperparameter tuning with differential privacy](https://openreview.net/forum?id=-70L8lpp9DF).

You can reproduce Figure 7 in the paper by running `figure7.py`. It loads by
default values used to plot the figure contained in the paper, and we also
included a dictionary `lr_acc.json` containing the accuracy of a large number of
ML models trained with different learning rates. If you'd like to try our
approach to fine-tune your own parameters, you will have to modify the code that
interacts with this dictionary (`lr_acc` in the code from `figure7.py`).

## Citing this work

If you use this repository for academic research, you are highly encouraged
(though not required) to cite our paper:

```
@inproceedings{
papernot2022hyperparameter,
title={Hyperparameter Tuning with Renyi Differential Privacy},
author={Nicolas Papernot and Thomas Steinke},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=-70L8lpp9DF}
}
```
