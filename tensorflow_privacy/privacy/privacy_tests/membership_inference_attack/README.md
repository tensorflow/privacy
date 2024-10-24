# Membership inference attack

A good privacy-preserving model learns from the training data, but doesn't
memorize it. This library provides empirical tests for measuring potential
memorization.

Technically, the tests build classifiers that infer whether a particular sample
was present in the training set. The more accurate such classifier is, the more
memorization is present and thus the less privacy-preserving the model is.

The privacy vulnerability (or memorization potential) is measured via the area
under the ROC-curve (`auc`) or via max{|fpr - tpr|} (`advantage`) of the attack
classifier. These measures are very closely related. We can also obtain a lower
bound for the differential privacy epsilon.

The tests provided by the library are "black box". That is, only the outputs of
the model are used (e.g., losses, logits, predictions). Neither model internals
(weights) nor input samples are required.

## How to use

### Installation notes

To use the latest version of the MIA library, please install TF Privacy with
"pip install -U git+https://github.com/tensorflow/privacy". See
https://github.com/tensorflow/privacy/issues/151 for more details.

### Basic usage

The simplest possible usage is

```python
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData

# Suppose we have evaluated the model on training and test examples to get the
# per-example losses:
# loss_train  shape: (n_train, )
# loss_test  shape: (n_test, )

attacks_result = mia.run_attacks(
    AttackInputData(loss_train=loss_train, loss_test=loss_test)
)
```

This example calls `run_attacks` with the default options to run a host of
(fairly simple) attacks behind the scenes (depending on which data is fed in),
and computes the most important measures.

> NOTE: The train and test sets are balanced internally, i.e., an equal number
> of in-training and out-of-training examples is chosen for the attacks
> (whichever has fewer examples). These are subsampled uniformly at random
> without replacement from the larger of the two.

Then, we can view the attack results by:

```python
print(attacks_result.summary())
# Example output:
# Best-performing attacks over all slices
#   LOGISTIC_REGRESSION (with 7041 training and 3156 test examples) achieved an AUC of 0.72 on slice CORRECTLY_CLASSIFIED=False
#   LOGISTIC_REGRESSION (with 7041 training and 3156 test examples) achieved an advantage of 0.34 on slice CORRECTLY_CLASSIFIED=False
#   LOGISTIC_REGRESSION (with 5000 training and 1000 test examples) achieved a positive predictive value of 1.00 on slice CLASS=0
#   THRESHOLD_ATTACK (with 50000 training and 10000 test examples) achieved top-5 epsilon lower bounds of 4.6254, 4.6121, 4.5986, 4.5850, 4.5711 on slice Entire dataset
```

### Other codelabs

Please head over to the
[codelabs](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/codelabs)
section for an overview of the library in action.

### Advanced usage

#### Specifying attacks to run

Sometimes, we have more information about the data, such as the logits and the
labels, and we may want to have finer-grained control of the attack, such as
using more complicated classifiers instead of the simple threshold attack, and
looks at the attack results by examples' class. In thoses cases, we can provide
more information to `run_attacks`.

```python
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
```

First, similar as before, we specify the input for the attack as an
`AttackInputData` object:

```python
# Suppose we have the labels as integers starting from 0
# labels_train  shape: (n_train, )
# labels_test  shape: (n_test, )

# Evaluate your model on training and test examples to get
# logits_train  shape: (n_train, n_classes)
# logits_test  shape: (n_test, n_classes)
# loss_train  shape: (n_train, )
# loss_test  shape: (n_test, )

attack_input = AttackInputData(
    logits_train=logits_train,
    logits_test=logits_test,
    loss_train=loss_train,
    loss_test=loss_test,
    labels_train=labels_train,
    labels_test=labels_test,
)
```

Instead of `logits`, you can also specify `probs_train` and `probs_test` as the
predicted probability vectors of each example.

Then, we specify some details of the attack. The first part includes the
specifications of the slicing of the data. For example, we may want to evaluate
the result on the whole dataset, or by class, percentiles, or the correctness of
the model's classification. These can be specified by a `SlicingSpec` object.

```python
slicing_spec = SlicingSpec(
    entire_dataset=True,
    by_class=True,
    by_percentiles=False,
    by_classification_correctness=True,
)
```

The second part specifies the classifiers for the attacker to use. Currently,
our API supports five classifiers, including `AttackType.THRESHOLD_ATTACK` for
simple threshold attack, `AttackType.LOGISTIC_REGRESSION`,
`AttackType.MULTI_LAYERED_PERCEPTRON`, `AttackType.RANDOM_FOREST`, and
`AttackType.K_NEAREST_NEIGHBORS` which use the corresponding machine learning
models. For some model, different classifiers can yield pretty different
results. We can put multiple classifiers in a list:

```python
attack_types = [
    AttackType.THRESHOLD_ATTACK,
    AttackType.LOGISTIC_REGRESSION,
]
```

Now, we can call the `run_attacks` methods with all specifications:

```python
attacks_result = mia.run_attacks(
    attack_input=attack_input,
    slicing_spec=slicing_spec,
    attack_types=attack_types,
)
```

This returns an object of type `AttackResults`. We can, for example, use the
following code to see the attack results specified per-slice, as we have request
attacks by class and by model's classification correctness.

```python
print(attacks_result.summary(by_slices = True))
# Example output:
# Best-performing attacks over all slices
#   LOGISTIC_REGRESSION (with 7041 training and 3156 test examples) achieved an AUC of 0.72 on slice CORRECTLY_CLASSIFIED=False
#   LOGISTIC_REGRESSION (with 7041 training and 3156 test examples) achieved an advantage of 0.34 on slice CORRECTLY_CLASSIFIED=False
#   LOGISTIC_REGRESSION (with 5000 training and 1000 test examples) achieved a positive predictive value of 1.00 on slice CLASS=0
#   THRESHOLD_ATTACK (with 50000 training and 10000 test examples) achieved top-5 epsilon lower bounds of 4.6254, 4.6121, 4.5986, 4.5850, 4.5711 on slice Entire dataset

# Best-performing attacks over slice: "Entire dataset"
#   LOGISTIC_REGRESSION (with 50000 training and 10000 test examples) achieved an AUC of 0.58
#   LOGISTIC_REGRESSION (with 50000 training and 10000 test examples) achieved an advantage of 0.17
#   THRESHOLD_ATTACK (with 50000 training and 10000 test examples) achieved a positive predictive value of 0.86
#   THRESHOLD_ATTACK (with 50000 training and 10000 test examples) achieved top-5 epsilon lower bounds of 4.6254, 4.6121, 4.5986, 4.5850, 4.5711

# Best-performing attacks over slice: "CLASS=0"
#   LOGISTIC_REGRESSION (with 5000 training and 1000 test examples) achieved an AUC of 0.63
#   LOGISTIC_REGRESSION (with 5000 training and 1000 test examples) achieved an advantage of 0.19
#   LOGISTIC_REGRESSION (with 5000 training and 1000 test examples) achieved a positive predictive value of 1.00
#   THRESHOLD_ATTACK (with 5000 training and 1000 test examples) achieved top-5 epsilon lower bounds of 4.1920, 4.1645, 4.1364, 4.1074, 4.0775

# ...

# Best-performing attacks over slice: "CORRECTLY_CLASSIFIED=True"
#   LOGISTIC_REGRESSION (with 42959 training and 6844 test examples) achieved an AUC of 0.51
#   LOGISTIC_REGRESSION (with 42959 training and 6844 test examples) achieved an advantage of 0.05
#   LOGISTIC_REGRESSION (with 42959 training and 6844 test examples) achieved a positive predictive value of 0.94
#   THRESHOLD_ATTACK (with 42959 training and 6844 test examples) achieved top-5 epsilon lower bounds of 0.9495, 0.6358, 0.5630, 0.4536, 0.4341

# Best-performing attacks over slice: "CORRECTLY_CLASSIFIED=False"
#   LOGISTIC_REGRESSION (with 7041 training and 3156 test examples) achieved an AUC of 0.72
#   LOGISTIC_REGRESSION (with 7041 training and 3156 test examples) achieved an advantage of 0.34
#   LOGISTIC_REGRESSION (with 7041 training and 3156 test examples) achieved a positive predictive value of 0.97
#   LOGISTIC_REGRESSION (with 7041 training and 3156 test examples) achieved top-5 epsilon lower bounds of 3.8844, 3.8678, 3.8510, 3.8339, 3.8165
```

#### Viewing and plotting the attack results

We have seen an example of using `summary()` to view the attack results as text.
We also provide some other ways for inspecting the attack results.

To get the attack that achieves the maximum attacker advantage, AUC, or epsilon
lower bound, we can do

```python
max_auc_attacker = attacks_result.get_result_with_max_auc()
max_advantage_attacker = attacks_result.get_result_with_max_attacker_advantage()
max_epsilon_attacker = attacks_result.get_result_with_max_epsilon()
```

Then, for individual attack, such as `max_auc_attacker`, we can check its type,
attacker advantage, AUC, and epsilon lower bound by

```python
print(
    "Attack type with max AUC: %s, AUC of %.2f, Attacker advantage of %.2f, Epsilon lower bound of %s"
    % (
        max_auc_attacker.attack_type,
        max_auc_attacker.roc_curve.get_auc(),
        max_auc_attacker.roc_curve.get_attacker_advantage(),
        max_auc_attacker.get_epsilon_lower_bound()
    )
)
# Example output:
# Attack type with max AUC: LOGISTIC_REGRESSION, AUC of 0.72, Attacker advantage of 0.34, Epsilon lower bound of [3.88435257 3.86781797 3.85100545 3.83390548 3.81650809]
```

We can also plot its ROC curve by

```python
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting

figure = plotting.plot_roc_curve(max_auc_attacker.roc_curve)
```

which would give a figure like the one below
![roc_fig](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/codelab_roc_fig.png?raw=true)

Additionally, we provide functionality to convert the attack results into Pandas
dataframe:

```python
import pandas as pd

pd.set_option("display.max_rows", 8, "display.max_columns", None)
print(attacks_result.calculate_pd_dataframe())
# Example output:
#            slice feature slice value  train size  test size          attack type  Attacker advantage  Positive predictive value       AUC  Epsilon lower bound_1  Epsilon lower bound_2  Epsilon lower bound_3  Epsilon lower bound_4  Epsilon lower bound_5
# 0         Entire dataset                   50000      10000     THRESHOLD_ATTACK            0.172520                   0.862614  0.581630               4.625393               4.612104               4.598635               4.584982               4.571140
# 1         Entire dataset                   50000      10000  LOGISTIC_REGRESSION            0.173060                   0.862081  0.583981               4.531399               4.513775               4.511974               4.498905               4.492165
# 2                  class           0        5000       1000     THRESHOLD_ATTACK            0.162000                   0.877551  0.580728               4.191954               4.164547               4.136368               4.107372               4.077511
# 3                  class           0        5000       1000  LOGISTIC_REGRESSION            0.193800                   1.000000  0.627758               3.289194               3.220285               3.146292               3.118849               3.066407
# ...
# 22  correctly_classified        True       42959       6844     THRESHOLD_ATTACK            0.043953                   0.862643  0.474713               0.949550               0.635773               0.563032               0.453640               0.434125
# 23  correctly_classified        True       42959       6844  LOGISTIC_REGRESSION            0.048963                   0.943218  0.505334               0.597257               0.596095               0.594016               0.592702               0.590765
# 24  correctly_classified       False        7041       3156     THRESHOLD_ATTACK            0.326865                   0.941176  0.707597               3.818741               3.805451               3.791982               3.778329               3.764488
# 25  correctly_classified       False        7041       3156  LOGISTIC_REGRESSION            0.336655                   0.972222  0.717386               3.884353               3.867818               3.851005               3.833905               3.816508
```

#### Advanced Membership Inference Attacks

Threshold MIA uses the intuition that training samples usually have lower loss
than test samples, and it thus predict samples with loss lower than a threshold
as in-training / member. However, some data samples might be intrinsically
harder than others. For example, a hard sample might have pretty high loss even
when included in the training set, and an easy sample might get low loss even
when it's not. So using the same threshold for all samples might be suboptimal.

People have considered customizing the membership prediction criteria for
different examples by looking at how they behave when included in or excluded
from the training set. To do that, we can train a few shadow models with
training sets being different subsets of all samples. Then for each sample, we
will know what its loss looks like when it's a member or non-member. Now, we can
compare its loss from the target model to those from the shadow models. For
example, if the average loss is `x` when the sample is a member, and `y` when
it's not, we might adjust the target loss by subtracting `(x+y)/2`. We can
expect the adjusted losses of different samples to be more of the same scale
compared to the original target losses. This gives us potentially better
estimations for membership.

In `advanced_mia.py`, we provide the method described above, and another method
that uses a more advanced way, i.e. distribution fitting to estimate membership.
`advanced_mia_example.py` shows an example for doing the advanced membership
inference on a CIFAR-10 task.

### External guides / press mentions

*   [Introductory blog post](https://franziska-boenisch.de/posts/2021/01/membership-inference/)
    to the theory and the library by Franziska Boenisch from the Fraunhofer
    AISEC institute.
*   [Google AI Blog Post](https://ai.googleblog.com/2021/01/google-research-looking-back-at-2020.html#ResponsibleAI)
*   [TensorFlow Blog Post](https://blog.tensorflow.org/2020/06/introducing-new-privacy-testing-library.html)
*   [VentureBeat article](https://venturebeat.com/2020/06/24/google-releases-experimental-tensorflow-module-that-tests-the-privacy-of-ai-models/)
*   [Tech Xplore article](https://techxplore.com/news/2020-06-google-tensorflow-privacy-module.html)

## Contact / Feedback

Fill out this
[Google form](https://docs.google.com/forms/d/1DPwr3_OfMcqAOA6sdelTVjIZhKxMZkXvs94z16UCDa4/edit)
or reach out to us at tf-privacy@google.com and let us know how you’re using
this module. We’re keen on hearing your stories, feedback, and suggestions!

## Contributing

If you wish to add novel attacks to the attack library, please check our
[guidelines](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/CONTRIBUTING.md).

## Copyright

Copyright 2021 - Google LLC
