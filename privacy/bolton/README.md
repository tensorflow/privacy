# Bolton Subpackage

This package contains source code for the Bolton method. This method is a subset
of methods used in the ensuring privacy in machine learning that leverages
additional assumptions to provide a new way of approaching the privacy
guarantees.

## Bolton Description

This method uses 4 key steps to achieve privacy guarantees:
  1. Adds noise to weights after training (output perturbation).
  2. Projects weights to R after each batch
  3. Limits learning rate
  4. Use a strongly convex loss function (see compile)

For more details on the strong convexity requirements, see:
Bolt-on Differential Privacy for Scalable Stochastic Gradient
Descent-based Analytics by Xi Wu et al.

## Why Bolton?

The major difference for the Bolton method is that it injects noise post model
convergence, rather than noising gradients or weights during training. This
approach requires some additional constraints listed in the Description.
Should the use-case and model satisfy these constraints, this is another
approach that can be trained to maximize utility while maintaining the privacy.
The paper describes in detail the advantages and disadvantages of this approach
and its results compared to some other methods, namely noising at each iteration
and no noising.

## Tutorials

This package has a tutorial that can be found in the root tutorials directory,
under `bolton_tutorial.py`.

## Contribution

This package was initially contributed by Georgian Partners with the hope of
growing the tensorflow/privacy library. There are several rich use cases for
delta-epsilon privacy in machine learning, some of which can be explored here:
https://medium.com/apache-mxnet/epsilon-differential-privacy-for-machine-learning-using-mxnet-a4270fe3865e
https://arxiv.org/pdf/1811.04911.pdf

## Contacts

In addition to the maintainers of tensorflow/privacy listed in the root
README.md, please feel free to contact members of Georgian Partners. In
particular,

* Georgian Partners(@georgianpartners)
* Ji Chao Zhang(@Jichaogp)
* Christopher Choquette(@cchoquette)

## Copyright

Copyright 2019 - Google LLC
