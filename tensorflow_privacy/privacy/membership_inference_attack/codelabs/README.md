# Membership inference attack examples

## Introductory codelab

The easiest way to get started is to go through [the introductory codelab](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/codelabs/codelab.ipynb).
This trains a simple image classification model and tests it against a series
of membership inference attacks.

For a more detailed overview of the library, please check the sections below.

## End to end example
As an alternative to the introductory codelab, we also have a standalone
[example.py](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/codelabs/example.py).

## Sequence to sequence models

If you're interested in sequence to sequence model attacks, please see the
[seq2seq colab](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/codelabs/third_party/seq2seq_membership_inference/seq2seq_membership_inference_codelab.ipynb).

## Membership probability score

If you're interested in the membership probability score (also called privacy
risk score) developed by Song and Mittal, please see their
[membership probability codelab](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/codelabs/membership_probability_codelab.ipynb).

The accompanying paper is on [arXiv](https://arxiv.org/abs/2003.10595).
## Specifying attacks to run

Sometimes, we have more information about the data, such as the logits and the
labels,
and we may want to have finer-grained control of the attack, such as using more
complicated classifiers instead of the simple threshold attack, and looks at the
attack results by examples' class.
In thoses cases, we can provide more information to `run_attacks`.

```python
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
```

First, similar as before, we specify the input for the attack as an
`AttackInputData` object:

```python
# Evaluate your model on training and test examples to get
# logits_train  shape: (n_train, n_classes)
# logits_test  shape: (n_test, n_classes)
# loss_train  shape: (n_train, )
# loss_test  shape: (n_test, )

attack_input = AttackInputData(
    logits_train = logits_train,
    logits_test = logits_test,
    loss_train = loss_train,
    loss_test = loss_test,
    labels_train = labels_train,
    labels_test = labels_test)
```

Instead of `logits`, you can also specify
`probs_train` and `probs_test` as the predicted probabilty vectors of each
example.

Then, we specify some details of the attack.
The first part includes the specifications of the slicing of the data. For
example, we may want to evaluate the result on the whole dataset, or by class,
percentiles, or the correctness of the model's classification.
These can be specified by a `SlicingSpec` object.

```python
slicing_spec = SlicingSpec(
    entire_dataset = True,
    by_class = True,
    by_percentiles = False,
    by_classification_correctness = True)
```

The second part specifies the classifiers for the attacker to use.
Currently, our API supports five classifiers, including
`AttackType.THRESHOLD_ATTACK` for simple threshold attack,
`AttackType.LOGISTIC_REGRESSION`,
`AttackType.MULTI_LAYERED_PERCEPTRON`,
`AttackType.RANDOM_FOREST`, and
`AttackType.K_NEAREST_NEIGHBORS`
which use the corresponding machine learning models.
For some model, different classifiers can yield pertty different results.
We can put multiple classifers in a list:

```python
attack_types = [
    AttackType.THRESHOLD_ATTACK,
    AttackType.LOGISTIC_REGRESSION
]
```

Now, we can call the `run_attacks` methods with all specifications:

```python
attacks_result = mia.run_attacks(attack_input=attack_input,
                                 slicing_spec=slicing_spec,
                                 attack_types=attack_types)
```

This returns an object of type `AttackResults`. We can, for example, use the
following code to see the attack results specificed per-slice, as we have
request attacks by class and by model's classification correctness.

```python
print(attacks_result.summary(by_slices = True))
# Example output:
# ->  Best-performing attacks over all slices
#       THRESHOLD_ATTACK achieved an AUC of 0.75 on slice CORRECTLY_CLASSIFIED=False
#       THRESHOLD_ATTACK achieved an advantage of 0.38 on slice CORRECTLY_CLASSIFIED=False
#
#     Best-performing attacks over slice: "Entire dataset"
#       LOGISTIC_REGRESSION achieved an AUC of 0.61
#       THRESHOLD_ATTACK achieved an advantage of 0.22
#
#     Best-performing attacks over slice: "CLASS=0"
#       LOGISTIC_REGRESSION achieved an AUC of 0.62
#       LOGISTIC_REGRESSION achieved an advantage of 0.24
#
#     Best-performing attacks over slice: "CLASS=1"
#       LOGISTIC_REGRESSION achieved an AUC of 0.61
#       LOGISTIC_REGRESSION achieved an advantage of 0.19
#
#     ...
#
#     Best-performing attacks over slice: "CORRECTLY_CLASSIFIED=True"
#       LOGISTIC_REGRESSION achieved an AUC of 0.53
#       THRESHOLD_ATTACK achieved an advantage of 0.05
#
#     Best-performing attacks over slice: "CORRECTLY_CLASSIFIED=False"
#       THRESHOLD_ATTACK achieved an AUC of 0.75
#       THRESHOLD_ATTACK achieved an advantage of 0.38
```


## Viewing and plotting the attack results

We have seen an example of using `summary()` to view the attack results as text.
We also provide some other ways for inspecting the attack results.

To get the attack that achieves the maximum attacker advantage or AUC, we can do

```python
max_auc_attacker = attacks_result.get_result_with_max_auc()
max_advantage_attacker = attacks_result.get_result_with_max_attacker_advantage()
```
Then, for individual attack, such as `max_auc_attacker`, we can check its type,
attacker advantage and AUC by

```python
print("Attack type with max AUC: %s, AUC of %.2f, Attacker advantage of %.2f" %
      (max_auc_attacker.attack_type,
       max_auc_attacker.roc_curve.get_auc(),
       max_auc_attacker.roc_curve.get_attacker_advantage()))
# Example output:
# -> Attack type with max AUC: THRESHOLD_ATTACK, AUC of 0.75, Attacker advantage of 0.38
```
We can also plot its ROC curve by

```python
import tensorflow_privacy.privacy.membership_inference_attack.plotting as plotting

figure = plotting.plot_roc_curve(max_auc_attacker.roc_curve)
```
which would give a figure like the one below
![roc_fig](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/codelabs/codelab_roc_fig.png?raw=true)

Additionally, we provide functionality to convert the attack results into Pandas
data frame:

```python
import pandas as pd

pd.set_option("display.max_rows", 8, "display.max_columns", None)
print(attacks_result.calculate_pd_dataframe())
# Example output:
#           slice feature slice value attack type  Attacker advantage       AUC
# 0        entire_dataset               threshold            0.216440  0.600630
# 1        entire_dataset                      lr            0.212073  0.612989
# 2                 class           0   threshold            0.226000  0.611669
# 3                 class           0          lr            0.239452  0.624076
# ..                  ...         ...         ...                 ...       ...
# 22  correctly_classfied        True   threshold            0.054907  0.471290
# 23  correctly_classfied        True          lr            0.046986  0.525194
# 24  correctly_classfied       False   threshold            0.379465  0.748138
# 25  correctly_classfied       False          lr            0.370713  0.737148
```

## Copyright

Copyright 2020 - Google LLC
