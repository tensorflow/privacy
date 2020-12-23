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

### Basic usage

The simplest possible usage is

```python
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData

# Suppose we have the labels as integers starting from 0
# labels_train  shape: (n_train, )
# labels_test  shape: (n_test, )

# Evaluate your model on training and test examples to get
# loss_train  shape: (n_train, )
# loss_test  shape: (n_test, )

attacks_result = mia.run_attacks(
    AttackInputData(
        loss_train = loss_train,
        loss_test = loss_test,
        labels_train = labels_train,
        labels_test = labels_test))
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
# -> Best-performing attacks over all slices
#      THRESHOLD_ATTACK (with 50000 training and 10000 test examples) achieved an AUC of 0.59 on slice Entire dataset
#      THRESHOLD_ATTACK (with 50000 training and 10000 test examples) achieved an advantage of 0.20 on slice Entire dataset
```

### Advanced usage / Other codelabs

Please head over to the [codelabs](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/membership_inference_attack/codelabs)
section for an overview of the library in action.


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
