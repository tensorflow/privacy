# Measure Privacy

Differential privacy is a framework for measuring the privacy guarantees
provided by an algorithm and can be expressed using the values ε (epsilon) and δ
(delta). Of the two, ε is more important and more sensitive to the choice of
hyperparameters. Roughly speaking, they mean the following:

*   ε gives a ceiling on how much the probability of a particular output can
    increase by including (or removing) a single training example. You usually
    want it to be a small constant (less than 10, or for more stringent privacy
    guarantees, less than 1). However, this is only an upper bound, and a large
    value of epsilon may still mean good practical privacy.
*   δ bounds the probability of an arbitrary change in model behavior. You can
    usually set this to a very small number (1e-7 or so) without compromising
    utility. A rule of thumb is to set it to be less than the inverse of the
    training data size.

The relationship between training hyperparameters and the resulting privacy in
terms of (ε, δ) is complicated and tricky to state explicitly. Our current
recommended approach is at the bottom of the [Get Started page](get_started.md),
which involves finding the maximum noise multiplier one can use while still
having reasonable utility, and then scaling the noise multiplier and number of
microbatches. TensorFlow Privacy provides a tool, `compute_dp_sgd_privacy` to
compute (ε, δ) based on the noise multiplier σ, the number of training steps
taken, and the fraction of input data consumed at each step. The amount of
privacy increases with the noise multiplier σ and decreases the more times the
data is used on training. Generally, in order to achieve an epsilon of at most
10.0, we need to set the noise multiplier to around 0.3 to 0.5, depending on the
dataset size and number of epochs. See the
[classification privacy tutorial](../tutorials/classification_privacy.ipynb) to
see the approach.

For more detail, see
[the original DP-SGD paper](https://arxiv.org/pdf/1607.00133.pdf).

You can use `compute_dp_sgd_privacy` to find out the epsilon given a fixed delta
value for your model [../tutorials/classification_privacy.ipynb]:

*   `q` : the sampling ratio - the probability of an individual training point
    being included in a mini batch (`batch_size/number_of_examples`).
*   `noise_multiplier` : A float that governs the amount of noise added during
    training. Generally, more noise results in better privacy and lower utility.
*   `steps` : The number of global steps taken.

A detailed writeup of the theory behind the computation of epsilon and delta is
available at
[Differential Privacy of the Sampled Gaussian Mechanism](https://arxiv.org/abs/1908.10530).
