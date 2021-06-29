# Secret Sharer Attack

A good privacy-preserving model learns from the training data, but
doesn't memorize it.
This folder contains codes for conducting the Secret Sharer attack from [this paper](https://arxiv.org/abs/1802.08232).
It is a method to test if a machine learning model memorizes its training data.

The high level idea is to insert some random sequences as “secrets” into the
training data, and then measure if the model has memorized those secrets.
If there is significant memorization, it means that there can be potential
privacy risk.

## How to Use

### Overview of the files

-   `generate_secrets.py` contains the code for generating secrets.
-   `exposures.py` contains code for evaluating exposures.
-   `secret_sharer_example.ipynb` is an example (character-level LSTM) for using
    the above code to conduct secret sharer attack.


### Contact / Feedback

Fill out this
[Google form](https://docs.google.com/forms/d/1DPwr3_OfMcqAOA6sdelTVjIZhKxMZkXvs94z16UCDa4/edit)
or reach out to us at tf-privacy@google.com and let us know how you’re using
this module. We’re keen on hearing your stories, feedback, and suggestions!

## Contributing

If you wish to add novel attacks to the attack library, please check our
[guidelines](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/CONTRIBUTING.md).

## Copyright

Copyright 2021 - Google LLC
