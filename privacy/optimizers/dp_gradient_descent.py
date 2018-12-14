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

"""DPGradientDescentOptimizer for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import privacy.optimizers.gaussian_average_query as ph


class DPGradientDescentOptimizer(tf.train.GradientDescentOptimizer):
  """Optimizer that implements the DP gradient descent algorithm.

  """

  def __init__(self,
               learning_rate,
               use_locking=False,
               l2_norm_clip=1e9,
               noise_multiplier=0.0,
               nb_microbatches=1,
               name='DPGradientDescent'):
    """Construct a new DP gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate to
        use.
      use_locking: If True use locks for update operations.
      l2_norm_clip: Clipping parameter for DP-SGD.
      noise_multiplier: Noise multiplier for DP-SGD.
      nb_microbatches: Number of microbatches in which to split the input.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "DPGradientDescent".  @compatibility(eager) When
        eager execution is enabled, `learning_rate` can be a callable that takes
        no arguments and returns the actual value to use. This can be useful for
        changing these values across different invocations of optimizer
        functions. @end_compatibility
    """
    super(DPGradientDescentOptimizer, self).__init__(learning_rate, use_locking,
                                                     name)
    stddev = l2_norm_clip * noise_multiplier
    self._nb_microbatches = nb_microbatches
    self._privacy_helper = ph.GaussianAverageQuery(l2_norm_clip, stddev,
                                                   nb_microbatches)
    self._ph_global_state = self._privacy_helper.initial_global_state()

  def compute_gradients(self,
                        loss,
                        var_list,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):

    # Note: it would be closer to the correct i.i.d. sampling of records if
    # we sampled each microbatch from the appropriate binomial distribution,
    # although that still wouldn't be quite correct because it would be sampling
    # from the dataset without replacement.
    microbatches_losses = tf.reshape(loss, [self._nb_microbatches, -1])
    sample_params = (
        self._privacy_helper.derive_sample_params(self._ph_global_state))

    def process_microbatch(i, sample_state):
      """Process one microbatch (record) with privacy helper."""
      grads, _ = zip(*super(DPGradientDescentOptimizer, self).compute_gradients(
          tf.gather(microbatches_losses, [i]), var_list, gate_gradients,
          aggregation_method, colocate_gradients_with_ops, grad_loss))
      grads_list = list(grads)
      sample_state = self._privacy_helper.accumulate_record(
          sample_params, sample_state, grads_list)
      return [tf.add(i, 1), sample_state]

    i = tf.constant(0)

    if var_list is None:
      var_list = (
          tf.trainable_variables() +
          tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    sample_state = self._privacy_helper.initial_sample_state(
        self._ph_global_state, var_list)

    # Use of while_loop here requires that sample_state be a nested structure of
    # tensors. In general, we would prefer to allow it to be an arbitrary
    # opaque type.
    _, final_state = tf.while_loop(
        lambda i, _: tf.less(i, self._nb_microbatches), process_microbatch,
        [i, sample_state])
    final_grads, self._ph_global_state = (
        self._privacy_helper.get_noised_average(final_state,
                                                self._ph_global_state))

    return zip(final_grads, var_list)
