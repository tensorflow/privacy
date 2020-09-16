# Membership inference attack

A good privacy-preserving model learns from the training data, but
doesn't memorize it. This library provides empirical tests for measuring
potential memorization.

Technically, the tests build classifiers that infer whether a particular sample
was present in the training set. The more accurate such classifier is, the more
memorization is present and thus the less privacy-preserving the model is.

The privacy vulnerability (or memorization potential) is measured
via the area under the ROC-curve (`auc`) or via max{|fpr - tpr|} (`advantage`)
of the attack classifier. These measures are very closely related.

The tests provided by the library are "black box". That is, only the outputs of
the model are used (e.g., losses, logits, predictions). Neither model internals
(weights) nor input samples are required.

## How to use


### API revamp note
We're **revamping our attacks API to make it more structured, modular and
extensible**. The docs below refers to the legacy experimental API and will be
updated soon. Stay tuned!

For a quick preview, you can take a look at `data_structures.py` and `membership_inference_attack_new.py`.

For now, here's a reference to the legacy API.


### Codelab

The easiest way to get started is to go through [the introductory codelab](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/codelab.ipynb).
This trains a simple image classification model and tests it against a series
of membership inference attacks.

For a more detailed overview of the library, please check the sections below.

### Basic usage

On the highest level, there is the `run_all_attacks_and_create_summary`
function, which chooses sane default options to run a host of (fairly simple)
attacks behind the scenes (depending on which data is fed in), computes the most
important measures and returns a summary of the results as a string of english
language (as well as optionally a python dictionary containing all results with
descriptive keys).

> NOTE: The train and test sets are balanced internally, i.e., an equal number
> of in-training and out-of-training examples is chosen for the attacks
> (whichever has fewer examples). These are subsampled uniformly at random
> without replacement from the larger of the two.

The simplest possible usage is

```python
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia

# Evaluate your model on training and test examples to get
# loss_train  shape: (n_train, )
# loss_test  shape: (n_test, )

summary, results = mia.run_all_attacks_and_create_summary(loss_train, loss_test, return_dict=True)
print(results)
# -> {'auc': 0.7044,
#     'best_attacker_auc': 'all_thresh_loss_auc',
#     'advantage': 0.3116,
#     'best_attacker_auc': 'all_thresh_loss_advantage'}
```

> NOTE: The keyword argument `return_dict` specified whether in addition to the
> `summary` the function also returns a python dictionary with the results.

If the model is a classifier, the logits or output probabilities (i.e., the
softmax of logits) can also be provided to perform stronger attacks.

> NOTE: The `logits_train` and `logits_test` arguments can also be filled with
> output probabilities per class ("posteriors").

```python
# logits_train  shape: (n_train, n_classes)
# logits_test  shape: (n_test, n_classes)

summary, results = mia.run_all_attacks_and_create_summary(loss_train, loss_test, logits_train,
                                      logits_test, return_dict=True)
print(results)
# -> {'auc': 0.5382,
#     'best_attacker_auc': 'all_lr_logits_loss_test_auc',
#     'advantage': 0.0572,
#     'best_attacker_auc': 'all_mlp_logits_loss_test_advantage'}
```

The `summary` will be a string in natural language describing the results in
more detail, e.g.,

```
========== AUC ==========
The best attack (all_lr_logits_loss_test_auc) achieved an auc of 0.5382.

========== ADVANTAGE ==========
The best attack (all_mlp_logits_loss_test_advantage) achieved an advantage of 0.0572.
```

Similarly, we can run attacks on the logits alone, without access to losses:

```python
summary, results = mia.run_all_attacks_and_create_summary(logits_train=logits_train,
                                      logits_test=logits_test,
                                      return_dict=True)
print(results)
# -> {'auc': 0.9278,
#     'best_attacker_auc': 'all_rf_logits_test_auc',
#     'advantage': 0.6991,
#     'best_attacker_auc': 'all_rf_logits_test_advantage'}
```

### Advanced usage

Finally, if we also have access to the true labels of the training and test
inputs, we can run the attacks for each class separately. If labels *and* logits
are provided, attacks only for misclassified (typically uncertain) examples are
also performed.

```python
summary, results = mia.run_all_attacks_and_create_summary(loss_train, loss_test, logits_train,
                                      logits_test, labels_train, labels_test,
                                      return_dict=True)
```

Here, we now also get as output the class with the maximal vulnerability
according to our metrics (`max_vuln_class_auc`, `max_vuln_class_advantage`)
together with the corresponding values (`class_<CLASS>_auc`,
`class_<CLASS>_advantage`). The same values exist in the `results` dictionary
for `min` instead of `max`, i.e., the least vulnerable classes. Moreover, the
gap between the maximum and minimum values (`max_class_gap_auc`,
`max_class_gap_advantage`) is also provided. Similarly, the vulnerability
metrics when the attacks are restricted to the misclassified examples
(`misclassified_auc`, `misclassified_advantage`) are also shown. Finally, the
results also contain the number of examples in each of these groups, i.e.,
within each of the reported classes as well as the number of misclassified
examples. The final `results` dictionary is of the form

```
{'auc': 0.9181,
 'best_attacker_auc': 'all_rf_logits_loss_test_auc',
 'advantage': 0.6915,
 'best_attacker_advantage': 'all_rf_logits_loss_test_advantage',
 'max_class_gap_auc': 0.254,
 'class_5_auc': 0.9512,
 'class_3_auc': 0.6972,
 'max_vuln_class_auc': 5,
 'min_vuln_class_auc': 3,
 'max_class_gap_advantage': 0.5073,
 'class_0_advantage': 0.8086,
 'class_3_advantage': 0.3013,
 'max_vuln_class_advantage': 0,
 'min_vuln_class_advantage': 3,
 'misclassified_n_examples': 4513.0,
 'class_0_n_examples': 899.0,
 'class_1_n_examples': 900.0,
 'class_2_n_examples': 931.0,
 'class_3_n_examples': 893.0,
 'class_4_n_examples': 960.0,
 'class_5_n_examples': 884.0}
```

### Setting the precision of the reported results

Finally, `run_all_attacks_and_create_summary` takes one extra keyword argument
`decimals`, expecting a positive integer. This sets the precision of all result
values as the number of decimals to report. It defaults to 4.

## Run all attacks and get all outputs

With the `run_all_attacks` function, one can run all implemented attacks on all
possible subsets of the data (all examples, split by class, split by confidence
deciles, misclassified only). This function returns a relatively large
dictionary with all attack results. This is the most detailed information one
could get about these types of membership inference attacks (besides plots for
each attack, see next section.) This is useful if you know exactly what you're
looking for.

> NOTE: The `run_all_attacks` function takes as an additional argument which
> trained attackers to run. In the `run_all_attacks_and_create_summary`, only
> logistic regression (`lr`) is trained as a binary classifier to distinguish
> in-training form out-of-training examples. In addition, with the
> `attack_classifiers` argument, one can add multi-layered perceptrons (`mlp`),
> random forests (`rf`), and k-nearest-neighbors (`knn`) or any subset thereof
> for the attack models. Note that these classifiers may not converge.

```python
mia.run_all_attacks(loss_train, loss_test, logits_train, logits_test,
                labels_train, labels_test,
                attack_classifiers=('lr', 'mlp', 'rf', 'knn'))
```

Again, `run_all_attacks` can be called on all combinations of losses, logits,
probabilities, and labels as long as at least either losses or logits
(probabilities) are provided.

## Fine grained control over individual attacks and plots

The `run_attack` function exposes the underlying workhorse of the
`run_all_attacks` and `run_all_attacks_and_create_summary` functionality. It
allows for fine grained control of which attacks to run individually.

As another key feature, this function also exposes options to store receiver
operator curve plots for the different attacks as well as histograms of losses
or the maximum logits/probabilities. Finally, we can also store all results
(including the values to reproduce the plots) to colossus.

All options are explained in detail in the doc string of the `run_attack`
function.

For example, to run a simple threshold attack on the losses only and store plots
and result data to colossus, run

```python
data_path = '/Users/user/Desktop/test/'  # set to None to not store data
figure_path = '/Users/user/Desktop/test/' # set to None to not store figures

mia.attack(loss_train=loss_train,
           loss_test=loss_test,
           metric='auc',
           output_directory=data_path,
           figure_directory=figure_path)
```

Among other things, the `run_attack` functionality allows to control:

*   which metrics to output (`metric` argument, using `auc` or `advantage` or
    both)
*   which classifiers (logistic regression, multi-layered perceptrons, random
    forests) to train as attackers beyond the simple threshold attacks
    (`attack_classifiers`)
*   to only attack a specific (set of) classes (`by_class`)
*   to only attack specific percentiles of the data (`by_percentile`).
    Percentiles here are computed by looking at the largest logit or probability
    for each example, i.e., how confident the model is in its prediction.
*   to only attack the misclassified examples (`only_misclassified`)
*   not to balance examples between the in-training and out-of-training examples
    using `balance`. By default an equal number of examples from train and test
    are selected for the attacks (whichever is smaller).
*   the test set size for trained attacks (`test_size`). When a classifier is
    trained to distinguish between train and test examples, a train-test split
    for that classifier itself is required.
*   for the train-test split as well as for the class balancing randomness is
    used with a seed specified by `random_state`.

## Contact / Feedback

Fill out this
[Google form](https://docs.google.com/forms/d/1DPwr3_OfMcqAOA6sdelTVjIZhKxMZkXvs94z16UCDa4/edit)
or reach out to us at tf-privacy@google.com and let us know how you’re using
this module. We’re keen on hearing your stories, feedback, and suggestions!

## Contributing

If you wish to add novel attacks to the attack library, please check our
[guidelines](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/CONTRIBUTING.md).

## Copyright

Copyright 2020 - Google LLC
