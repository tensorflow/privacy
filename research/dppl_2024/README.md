# Beyond the Mean: Differentially Private Prototypes for Private Transfer Learning
This folder contains the code for

**Beyond the Mean: Differentially Private Prototypes for Private Transfer Learning**  
by Dariush Wahdany, Matthew Jagielski, Adam Dziedzic, Franziska Boenisch  
https://arxiv.org/abs/2406.08039

Abstract:
Machine learning (ML) models have been shown to leak private information from their training datasets. Differential Privacy (DP), typically implemented through the differential private stochastic gradient descent algorithm (DP-SGD), has become the standard solution to bound leakage from the models. Despite recent improvments, DP-SGD-based approaches for private learning still usually struggle in the high privacy ($\varepsilon<0.1$) and low data regimes, and when the private training datasets are imbalanced. To overcome these limitations, we propose Differentially Private Prototype Learning (DPPL) as a new paradigm for private transfer learning. DPPL leverages publicly pre-trained encoders to extract features from private data and generates DP prototypes that represent each private class in the embedding space and can be publicly released for inference. Since our DP prototypes can be obtained from only a few private training data points and without iterative noise addition, they offer high-utility predictions and strong privacy guarantees even under the notion of pure DP. We additionally show that privacy-utility trade-offs can be further improved when leveraging the public data beyond pre-training of the encoder: we are able to privately sample our DP prototypes from the publicly available data points used to train the encoder. Our experimental evaluation with four state-of-the-art encoders, four vision datasets, and under different data and unbalancedness regimes demonstrate DPPL's high performance under strong privacy guarantees in challenging private learning setups.



## Table of Contents

- [Installation](#installation)
- [Description](#description)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Conda
```bash
conda env create -f env.yaml
```

### Pip
```bash
pip install -r requirements.txt
```

## Description

### Imbalanced Datasets
We construct the imbalanced datasets in `lib.utils.give_imbalanced_set`. The function places an upper bound on the number of samples per class according to the minimum number of samples per class. So for an imbalance ratio of $1$, the dataset is actually balanced. `lib.utils.decay` implements the decay function $f(c)=N\exp{-\lambda c}$. The class indices are shuffled depending on the seed, therefore whether classes are part of the majority or minority classes is random.

### DPPL-Mean
The implementation of **DPPL-Mean** can be found in `dppl_mean.py`. We first load the private dataset, average-pool its features and obtain imbalanced datasets as described above.
The private mean estimation occurs using the Jax-reimplementation of [*CoinPress*](https://proceedings.neurips.cc/paper_files/paper/2020/hash/a684eceee76fc522773286a895bc8436-Abstract.html)  in `lib.utils.coinpress`.

### DPPL-Public
The implementation of **DPPL-Public** can be found in `dppl_public.py`. We first load the private dataset and obtain imbalanced datasets as described above. The scores are computed using `lib.utils.pairwise_distance`, a function returning cosine distances $\in [0,2]$. `lib.utils.scores_single` implements the score calculation for a single public sample, by substracting the distance to each private sample from $2$, clipping the result to $[d_{\min},d_{\max}]$ and normalizing it to $[0,1]$, before summing over all the private samples. In our implementation the sensitivity is therefore always $1$, but the mechanism is identical to one where the scores are not normalized to $[0,1]$ and the sensitivity is reduced instead.

Finally, given the scores `lib.public.exponential` implements the exponential mechanism. Depending on whether the utility function is monotonic or not, we multiply the sensitivity by $2$ to achieve $\epsilon$-DP. For numerical reasons, the substract from all exponents the maximum exponent. Since this is the constant factor $\exp(-c)$ for all samples, the proportionality of the probalities and therefore the mechanism doesn't change, since the exponential mechanism is invariant to scaling of the utility function.

### DPPL-Public Top-K
The implementation of **DPPL-Public Top-K** can be found in `dppl_public_topk.py`. We first load the private dataset and obtain imbalanced datasets as described above. The scores are computed as in [DPPL-Public](#dppl-public). Our unordered top-K selection is implemented using the efficient sampling algorithm from [Duff](http://arxiv.org/abs/2010.04235) (Prop. 5). `lib.public.give_topk_proto_idx` returns the indices of the prototypes w.r.t. to the order of C, i.e. if it returns $0$ it means the best utility, $1$ the second best and so on. To do so, the utility is sampled with `lib.public.exponential_parallel` using the exponential mechanism in parallel for all classes. The remainder of `lib.public.give_topk_proto_idx` is just to uniformly sample the remaining $K-1$ prototypes, s.t. their utility is higher than the sampled one.

### Hyperparameters
We provide the hyperparameters for the models and datasets we used in `hparams_mean.md`, `hparams_public.md` and `hparams_public_topk.md`.

## Usage

Before running any of the experiments, set the path to your embeddings in `config/common.yaml`. Further options are
- Epsilon
- Imbalance Ratio
- Seed

We provide the required embeddings as a [huggingface dataset](https://huggingface.co/datasets/lsc64/DPPL-embeddings).

### DPPL-Mean
(Optional): In `config/mean.yaml`, change `pool` to any desired integer value. It configures the optional average pooling before the mean estimation and can improve utility especially at strict privacy budgets.

```bash
python dppl_mean.py
```
### DPPL-Public
(Optional): In `config/public.yaml`, change `max_score` and `min_score` to any desired values in [0,2], s.t. min_score < max_score. It defines the clipping of the scores and can improve utility especially at strict privacy budgets.

**Required**: In `config/public.yaml`, change `dataset.public_data` to the path to your public dataset embeddings.


```bash
python dppl_mean.py
```

### DPPL-Public Top-K
(Optional): In `config/public_topk.yaml`, change `max_score` and `min_score` to any desired values in [0,2], s.t. min_score < max_score. It defines the clipping of the scores and can improve utility especially at strict privacy budgets. Also, change `k` to any integer value. It defines how many prototypes are selected per class and can improve utility especially at lower privacy regimes.

**Required**: In `config/public_topk.yaml`, change `dataset.public_data` to the path to your public dataset embeddings.


```bash
python dppl_public_topk.py
```

