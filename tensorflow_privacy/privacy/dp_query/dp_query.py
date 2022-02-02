# Copyright 2020, The TensorFlow Authors.
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

import abc
import collections

import tensorflow as tf


class DPQuery(metaclass=abc.ABCMeta):
  """Interface for differentially private query mechanisms.

  Differential privacy is achieved by processing records to bound sensitivity,
  accumulating the processed records (usually by summing them) and then
  adding noise to the aggregated result. The process can be repeated to compose
  applications of the same mechanism, possibly with different parameters.

  The DPQuery interface specifies a functional approach to this process. A
  global state maintains state that persists across applications of the
  mechanism. For each application, the following steps are performed:

  1. Use the global state to derive parameters to use for the next sample of
     records.
  2. Initialize a sample state that will accumulate processed records.
  3. For each record:
     a. Process the record.
     b. Accumulate the record into the sample state.
  4. Get the result of the mechanism, possibly updating the global state to use
     in the next application.
  5. Derive metrics from the global state.

  Here is an example using the GaussianSumQuery. Assume there is some function
  records_for_round(round) that returns an iterable of records to use on some
  round.

  ```
  dp_query = tensorflow_privacy.GaussianSumQuery(
      l2_norm_clip=1.0, stddev=1.0)
  global_state = dp_query.initial_global_state()

  for round in range(num_rounds):
    sample_params = dp_query.derive_sample_params(global_state)
    sample_state = dp_query.initial_sample_state()
    for record in records_for_round(round):
      sample_state = dp_query.accumulate_record(
          sample_params, sample_state, record)

    result, global_state = dp_query.get_noised_result(
        sample_state, global_state)
    metrics = dp_query.derive_metrics(global_state)

    # Do something with result and metrics...
  ```
  """

  def initial_global_state(self):
    """Returns the initial global state for the DPQuery.

    The global state contains any state information that changes across
    repeated applications of the mechanism. The default implementation returns
    just an empty tuple for implementing classes that do not have any persistent
    state.

    Returns:
      The global state.
    """
    return ()

  def derive_sample_params(self, global_state):
    """Given the global state, derives parameters to use for the next sample.

    For example, if the mechanism needs to clip records to bound the norm,
    the clipping norm should be part of the sample params. In a distributed
    context, this is the part of the state that would be sent to the workers
    so they can process records.

    Args:
      global_state: The current global state.

    Returns:
      Parameters to use to process records in the next sample.
    """
    del global_state  # unused.
    return ()

  @abc.abstractmethod
  def initial_sample_state(self, template=None):
    """Returns an initial state to use for the next sample.

    For typical `DPQuery` classes that are aggregated by summation, this should
    return a nested structure of zero tensors of the appropriate shapes, to
    which processed records will be aggregated.

    Args:
      template: A nested structure of tensors, TensorSpecs, or numpy arrays used
        as a template to create the initial sample state. It is assumed that the
        leaves of the structure are python scalars or some type that has
        properties `shape` and `dtype`.
    Returns: An initial sample state.
    """
    pass

  def preprocess_record(self, params, record):
    """Preprocesses a single record.

    This preprocessing is applied to one client's record, e.g. selecting vectors
    and clipping them to a fixed L2 norm. This method can be executed in a
    separate TF session, or even on a different machine, so it should not depend
    on any TF inputs other than those provided as input arguments. In
    particular, implementations should avoid accessing any TF tensors or
    variables that are stored in self.

    Args:
      params: The parameters for the sample. In standard DP-SGD training, the
        clipping norm for the sample's microbatch gradients (i.e., a maximum
        norm magnitude to which each gradient is clipped)
      record: The record to be processed. In standard DP-SGD training, the
        gradient computed for the examples in one microbatch, which may be the
        gradient for just one example (for size 1 microbatches).

    Returns:
      A structure of tensors to be aggregated.
    """
    del params  # unused.
    return record

  @abc.abstractmethod
  def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
    """Accumulates a single preprocessed record into the sample state.

    This method is intended to only do simple aggregation, typically just a sum.
    In the future, we might remove this method and replace it with a way to
    declaratively specify the type of aggregation required.

    Args:
      sample_state: The current sample state. In standard DP-SGD training, the
        accumulated sum of previous clipped microbatch gradients.
      preprocessed_record: The preprocessed record to accumulate.

    Returns:
      The updated sample state.
    """
    pass

  def accumulate_record(self, params, sample_state, record):
    """Accumulates a single record into the sample state.

    This is a helper method that simply delegates to `preprocess_record` and
    `accumulate_preprocessed_record` for the common case when both of those
    functions run on a single device. Typically this will be a simple sum.

    Args:
      params: The parameters for the sample. In standard DP-SGD training, the
        clipping norm for the sample's microbatch gradients (i.e., a maximum
        norm magnitude to which each gradient is clipped)
      sample_state: The current sample state. In standard DP-SGD training, the
        accumulated sum of previous clipped microbatch gradients.
      record: The record to accumulate. In standard DP-SGD training, the
        gradient computed for the examples in one microbatch, which may be the
        gradient for just one example (for size 1 microbatches).

    Returns:
      The updated sample state. In standard DP-SGD training, the set of
      previous microbatch gradients with the addition of the record argument.
    """
    preprocessed_record = self.preprocess_record(params, record)
    return self.accumulate_preprocessed_record(sample_state,
                                               preprocessed_record)

  @abc.abstractmethod
  def merge_sample_states(self, sample_state_1, sample_state_2):
    """Merges two sample states into a single state.

    This can be useful if aggregation is performed hierarchically, where
    multiple sample states are used to accumulate records and then
    hierarchically merged into the final accumulated state. Typically this will
    be a simple sum.

    Args:
      sample_state_1: The first sample state to merge.
      sample_state_2: The second sample state to merge.

    Returns:
      The merged sample state.
    """
    pass

  @abc.abstractmethod
  def get_noised_result(self, sample_state, global_state):
    """Gets the query result after all records of sample have been accumulated.

    The global state can also be updated for use in the next application of the
    DP mechanism.

    Args:
      sample_state: The sample state after all records have been accumulated. In
        standard DP-SGD training, the accumulated sum of clipped microbatch
        gradients (in the special case of microbatches of size 1, the clipped
        per-example gradients).
      global_state: The global state, storing long-term privacy bookkeeping.

    Returns:
      A tuple `(result, new_global_state, event)` where:
        * `result` is the result of the query,
        * `new_global_state` is the updated global state, and
        * `event` is the `DpEvent` that occurred.
      In standard DP-SGD training, the result is a gradient update comprising a
      noised average of the clipped gradients in the sample state---with the
      noise and averaging performed in a manner that guarantees differential
      privacy.
    """
    pass

  def derive_metrics(self, global_state):
    """Derives metric information from the current global state.

    Any metrics returned should be derived only from privatized quantities.

    Args:
      global_state: The global state from which to derive metrics.

    Returns:
      A `collections.OrderedDict` mapping string metric names to tensor values.
    """
    del global_state
    return collections.OrderedDict()


def _zeros_like(arg):
  """A `zeros_like` function that also works for `tf.TensorSpec`s."""
  try:
    arg = tf.convert_to_tensor(value=arg)
  except TypeError:
    pass
  return tf.zeros(arg.shape, arg.dtype)


def _safe_add(x, y):
  """Adds x and y but if y is None, simply returns x."""
  return x if y is None else tf.add(x, y)


class SumAggregationDPQuery(DPQuery):
  """Base class for DPQueries that aggregate via sum."""

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return tf.nest.map_structure(_zeros_like, template)

  def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
    """Implements `tensorflow_privacy.DPQuery.accumulate_preprocessed_record`."""
    return tf.nest.map_structure(_safe_add, sample_state, preprocessed_record)

  def merge_sample_states(self, sample_state_1, sample_state_2):
    """Implements `tensorflow_privacy.DPQuery.merge_sample_states`."""
    return tf.nest.map_structure(tf.add, sample_state_1, sample_state_2)
