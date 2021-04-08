# Copyright 2021, The TensorFlow Authors.
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
"""Keras Model for vectorized dpsgd with XLA acceleration."""

import tensorflow as tf


def make_dp_model_class(cls):
  """Given a subclass of `tf.keras.Model`, returns a DP-SGD version of it."""

  class DPModelClass(cls):
    """A DP version of `cls`, which should be a subclass of `tf.keras.Model`."""

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        use_xla=True,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initializes the DPModelClass.

        Args:
            l2_norm_clip: Clipping norm (max L2 norm of per microbatch
              gradients).
            noise_multiplier: Ratio of the standard deviation to the clipping
              norm.
            use_xla: If `True`, compiles train_step to XLA.
      """
      super(DPModelClass, self).__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier

      if use_xla:
        self.train_step = tf.function(
            self.train_step, experimental_compile=True)

    def _process_per_example_grads(self, grads):
      grads_flat = tf.nest.flatten(grads)
      squared_l2_norms = [
          tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
      ]
      global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
      div = tf.maximum(global_norm / self._l2_norm_clip, 1.)
      clipped_flat = [g / div for g in grads_flat]
      return tf.nest.pack_sequence_as(grads, clipped_flat)

    def _reduce_per_example_grads(self, stacked_grads):
      summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
      noise_stddev = self._l2_norm_clip * self._noise_multiplier
      noise = tf.random.normal(
          tf.shape(input=summed_grads), stddev=noise_stddev)
      noised_grads = summed_grads + noise
      return noised_grads / tf.cast(stacked_grads.shape[0], noised_grads.dtype)

    def _compute_per_example_grads(self, data):
      x, y = data
      with tf.GradientTape() as tape:
        # We need to add the extra dimension to x and y because model
        # expects batched input.
        y_pred = self(x[None], training=True)
        loss = self.compiled_loss(
            y[None], y_pred, regularization_losses=self.losses)

      grads_list = tape.gradient(loss, self.trainable_variables)
      clipped_grads = self._process_per_example_grads(grads_list)
      return tf.squeeze(y_pred, axis=0), loss, clipped_grads

    def train_step(self, data):
      _, y = data
      y_pred, _, per_eg_grads = tf.vectorized_map(
          self._compute_per_example_grads, data)
      grads = tf.nest.map_structure(self._reduce_per_example_grads,
                                    per_eg_grads)
      self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
      self.compiled_metrics.update_state(y, y_pred)
      return {m.name: m.result() for m in self.metrics}

  DPModelClass.__doc__ = ('DP subclass of `tf.keras.{}`.').format(cls.__name__)

  return DPModelClass


DPModel = make_dp_model_class(tf.keras.Model)
DPSequential = make_dp_model_class(tf.keras.Sequential)
