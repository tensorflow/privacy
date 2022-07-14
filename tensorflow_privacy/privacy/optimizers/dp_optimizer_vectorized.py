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
"""Vectorized differentially private optimizers for TensorFlow."""

from absl import logging
import tensorflow as tf

AdagradOptimizer = tf.compat.v1.train.AdagradOptimizer
AdamOptimizer = tf.compat.v1.train.AdamOptimizer
GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name


def make_vectorized_optimizer_class(cls):
  """Given a subclass of `tf.compat.v1.train.Optimizer`, returns a vectorized DP-SGD subclass of it.

  Args:
    cls: Class from which to derive a DP subclass. Should be a subclass of
      `tf.compat.v1.train.Optimizer`.

  Returns:
    A DP-SGD subclass of `cls`.
  """
  child_code = cls.compute_gradients.__code__
  if child_code is not parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):  # pylint: disable=empty-docstring
    __doc__ = ("""Vectorized DP subclass of `{base_class}` using Gaussian
       averaging.

       You can use this as a differentially private replacement for
       `{base_class}`. This optimizer implements DP-SGD using
       the standard Gaussian mechanism. It differs from `{dp_class}` in that
       it attempts to vectorize the gradient computation and clipping of
       microbatches.

       When instantiating this optimizer, you need to supply several
       DP-related arguments followed by the standard arguments for
       `{short_base_class}`.

       Examples:

       ```python
       # Create optimizer.
       opt = {dp_vectorized_class}(l2_norm_clip=1.0, noise_multiplier=0.5, num_microbatches=1,
                <standard arguments>)
       ```

       When using the optimizer, be sure to pass in the loss as a
       rank-one tensor with one entry for each example.

       ```python
       # Compute loss as a tensor. Do not call tf.reduce_mean as you
       # would with a standard optimizer.
       loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
           labels=labels, logits=logits)

       train_op = opt.minimize(loss, global_step=global_step)
       ```
       """).format(
           base_class='tf.compat.v1.train.' + cls.__name__,
           dp_class='DP' +
           cls.__name__.replace('Optimizer', 'GaussianOptimizer'),
           short_base_class=cls.__name__,
           dp_vectorized_class='VectorizedDP' + cls.__name__)

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
        noise_multiplier: Ratio of the standard deviation to the clipping norm.
        num_microbatches: Number of microbatches into which each minibatch is
          split. If `None`, will default to the size of the minibatch, and
          per-example gradients will be computed.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      super().__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._num_microbatches = num_microbatches
      self._was_compute_gradients_called = False

    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None,
                          gradient_tape=None):
      """DP-SGD version of base class method."""
      self._was_compute_gradients_called = True
      if callable(loss):
        # TF is running in Eager mode
        raise NotImplementedError('Vectorized optimizer unavailable for TF2.')
      else:
        # TF is running in graph mode, check we did not receive a gradient tape.
        if gradient_tape:
          raise ValueError('When in graph mode, a tape should not be passed.')

        batch_size = tf.shape(input=loss)[0]
        if self._num_microbatches is None:
          self._num_microbatches = batch_size

        # Note: it would be closer to the correct i.i.d. sampling of records if
        # we sampled each microbatch from the appropriate binomial distribution,
        # although that still wouldn't be quite correct because it would be
        # sampling from the dataset without replacement.
        microbatch_losses = tf.reshape(loss, [self._num_microbatches, -1])

        if var_list is None:
          var_list = (
              tf.compat.v1.trainable_variables() + tf.compat.v1.get_collection(
                  tf.compat.v1.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        def process_microbatch(microbatch_loss):
          """Compute clipped grads for one microbatch."""
          microbatch_loss = tf.reduce_mean(input_tensor=microbatch_loss)
          grads, _ = zip(*super(DPOptimizerClass, self).compute_gradients(
              microbatch_loss, var_list, gate_gradients, aggregation_method,
              colocate_gradients_with_ops, grad_loss))
          grads_list = [
              g if g is not None else tf.zeros_like(v)
              for (g, v) in zip(list(grads), var_list)
          ]
          # Clip gradients to have L2 norm of l2_norm_clip.
          # Here, we use TF primitives rather than the built-in
          # tf.clip_by_global_norm() so that operations can be vectorized
          # across microbatches.
          grads_flat = tf.nest.flatten(grads_list)
          squared_l2_norms = [
              tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
          ]
          global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
          div = tf.maximum(global_norm / self._l2_norm_clip, 1.)
          clipped_flat = [g / div for g in grads_flat]
          clipped_grads = tf.nest.pack_sequence_as(grads_list, clipped_flat)
          return clipped_grads

        clipped_grads = tf.vectorized_map(process_microbatch, microbatch_losses)

        def reduce_noise_normalize_batch(stacked_grads):
          summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
          noise_stddev = self._l2_norm_clip * self._noise_multiplier
          noise = tf.random.normal(
              tf.shape(input=summed_grads), stddev=noise_stddev)
          noised_grads = summed_grads + noise
          return noised_grads / tf.cast(self._num_microbatches, tf.float32)

        final_grads = tf.nest.map_structure(reduce_noise_normalize_batch,
                                            clipped_grads)

        return list(zip(final_grads, var_list))

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
      # pylint: disable=g-doc-args, g-doc-return-or-yield
      """DP-SGD version of base class method."""
      assert self._was_compute_gradients_called, (
          'compute_gradients() on the differentially private optimizer was not'
          ' called. Which means that the training is not differentially '
          'private. It happens for example in Keras training in TensorFlow '
          '2.0+.')
      return super(DPOptimizerClass, self).apply_gradients(
          grads_and_vars=grads_and_vars, global_step=global_step, name=name)

  return DPOptimizerClass


VectorizedDPAdagradOptimizer = make_vectorized_optimizer_class(AdagradOptimizer)
VectorizedDPAdamOptimizer = make_vectorized_optimizer_class(AdamOptimizer)
VectorizedDPSGDOptimizer = make_vectorized_optimizer_class(
    GradientDescentOptimizer)

VectorizedDPAdagrad = VectorizedDPAdagradOptimizer
VectorizedDPAdam = VectorizedDPAdamOptimizer
VectorizedDPSGD = VectorizedDPSGDOptimizer
