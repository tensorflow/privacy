## Membership Inference Attacks From First Principles

This directory contains code to reproduce our paper:

**"Membership Inference Attacks From First Principles"** <br>
https://arxiv.org/abs/2112.03570 <br>
by Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramèr.

### INSTALLING

You will need to install fairly standard dependencies

`pip install scipy, sklearn, numpy, matplotlib`

and also some machine learning framework to train models. We train our models
with JAX + ObJAX so you will need to follow build instructions for that
https://github.com/google/objax
https://objax.readthedocs.io/en/latest/installation_setup.html

### RUNNING THE CODE

#### 1. Train the models

The first step in our attack is to train shadow models. As a baseline that
should give most of the gains in our attack, you should start by training 16
shadow models with the command

> bash scripts/train_demo.sh

or if you have multiple GPUs on your machine and want to train these models in
parallel, then modify and run

> bash scripts/train_demo_multigpu.sh

This will train several CIFAR-10 wide ResNet models to ~91% accuracy each, and
will output a bunch of files under the directory exp/cifar10 with structure:

```
exp/cifar10/
- experiment_N_of_16
-- hparams.json
-- keep.npy
-- ckpt/
--- 0000000100.npz
-- tb/
```

#### 2. Perform inference

Once the models are trained, now it's necessary to perform inference and save
the output features for each training example for each model in the dataset.

> python3 inference.py --logdir=exp/cifar10/

This will add to the experiment directory a new set of files

```
exp/cifar10/
- experiment_N_of_16
-- logits/
--- 0000000100.npy
```

where this new file has shape (50000, 10) and stores the model's output features
for each example.

#### 3. Compute membership inference scores

Finally we take the output features and generate our logit-scaled membership
inference scores for each example for each model.

> python3 score.py exp/cifar10/

And this in turn generates a new directory

```
exp/cifar10/
- experiment_N_of_16
-- scores/
--- 0000000100.npy
```

with shape (50000,) storing just our scores.

### PLOTTING THE RESULTS

Finally we can generate pretty pictures, and run the plotting code

> python3 plot.py

which should give (something like) the following output

![Log-log ROC Curve for all attacks](fprtpr.png "Log-log ROC Curve")

```
Attack Ours (online)
   AUC 0.6676, Accuracy 0.6077, TPR@0.1%FPR of 0.0169
Attack Ours (online, fixed variance)
   AUC 0.6856, Accuracy 0.6137, TPR@0.1%FPR of 0.0593
Attack Ours (offline)
   AUC 0.5488, Accuracy 0.5500, TPR@0.1%FPR of 0.0130
Attack Ours (offline, fixed variance)
   AUC 0.5549, Accuracy 0.5537, TPR@0.1%FPR of 0.0299
Attack Global threshold
   AUC 0.5921, Accuracy 0.6044, TPR@0.1%FPR of 0.0009
```

where the global threshold attack is the baseline, and our online,
online-with-fixed-variance, offline, and offline-with-fixed-variance attack
variants are the four other curves. Note that because we only train a few
models, the fixed variance variants perform best.

### Citation

You can cite this paper with

```
@article{carlini2021membership,
  title={Membership Inference Attacks From First Principles},
  author={Carlini, Nicholas and Chien, Steve and Nasr, Milad and Song, Shuang and Terzis, Andreas and Tramer, Florian},
  journal={arXiv preprint arXiv:2112.03570},
  year={2021}
}
```
