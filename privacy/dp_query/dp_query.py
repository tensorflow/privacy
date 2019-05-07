# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An interface for differentially private query mechanisms.

The DPQuery class abstracts the differential privacy mechanism needed by DP-SGD.

The nomenclature is not specific to machine learning, but rather comes from
the differential privacy literature. Therefore, instead of talking about
examples, minibatches, and gradients, the code talks about records, samples and
queries. For more detail, please see the paper here:
https://arxiv.org/pdf/1812.06210.pdf

A common usage paradigm for this class is centralized DP-SGD training on a
fixed set of training examples, which we call "standard DP-SGD training."
In such training, SGD applies as usual by computing gradient updates from a set
of training examples that form a minibatch. However, each minibatch is broken
up into disjoint "microbatches."  The gradient of each microbatch is computed
and clipped to a maximum norm, with the "records" for all such clipped gradients
forming a "sample" that constitutes the entire minibatch. Subsequently, that
sample can be "queried" to get an averaged, noised gradient update that can be
applied to model parameters.

In order to prevent inaccurate accounting of privacy parameters, the only
means of inspecting the gradients and updates of SGD training is via the use
of the below interfaces, and through the accumulation and querying of a
"sample state" abstraction. Thus, accessing data is indirect on purpose.

The DPQuery class also allows the use of a global state that may change between
samples. In the common situation where the privacy mechanism remains unchanged
throughout the entire training process, the global state is usually None.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class DPQuery(object):
  """Interface for differentially private query mechanisms."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def initial_global_state(self):
    """Returns the initial global state for the DPQuery."""
    pass

  @abc.abstractmethod
  def derive_sample_params(self, global_state):
    """Given the global state, derives parameters to use for the next sample.

    Args:
      global_state: The current global state.

    Returns:
      Parameters to use to process records in the next sample.
    """
    pass

  @abc.abstractmethod
  def initial_sample_state(self, global_state, tensors):
    """Returns an initial state to use for the next sample.

    Args:
      global_state: The current global state.
      tensors: A structure of tensors used as a template to create the initial
        sample state.

    Returns: An initial sample state.
    """
    pass

  @abc.abstractmethod
  def accumulate_record(self, params, sample_state, record):
    """Accumulates a single record into the sample state.

    Args:
      params: The parameters for the sample. In standard DP-SGD training,
        the clipping norm for the sample's microbatch gradients (i.e.,
        a maximum norm magnitude to which each gradient is clipped)
      sample_state: The current sample state. In standard DP-SGD training,
        the accumulated sum of previous clipped microbatch gradients.
      record: The record to accumulate. In standard DP-SGD training,
        the gradient computed for the examples in one microbatch, which
        may be the gradient for just one example (for size 1 microbatches).

    Returns:
      The updated sample state. In standard DP-SGD training, the set of
      previous mcrobatch gradients with the addition of the record argument.
    """
    pass

  @abc.abstractmethod
  def get_noised_result(self, sample_state, global_state):
    """Gets query result after all records of sample have been accumulated.

    Args:
      sample_state: The sample state after all records have been accumulated.
        In standard DP-SGD training, the accumulated sum of clipped microbatch
        gradients (in the special case of microbatches of size 1, the clipped
        per-example gradients).
      global_state: The global state, storing long-term privacy bookkeeping.

    Returns:
      A tuple (result, new_global_state) where "result" is the result of the
      query and "new_global_state" is the updated global state. In standard
      DP-SGD training, the result is a gradient update comprising a noised
      average of the clipped gradients in the sample state---with the noise and
      averaging performed in a manner that guarantees differential privacy.
    """
    pass
