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

  class DPModelClass(cls):  # pylint: disable=empty-docstring
    __doc__ = ("""DP subclass of `{base_model}`.

       This can be used as a differentially private replacement for
       {base_model}. This class implements DP-SGD using the standard
       Gaussian mechanism.

       When instantiating this class, you need to supply several
       DP-related arguments followed by the standard arguments for
       `{short_base_model}`.

       Examples:

       ```python
       # Create Model instance.
       model = {dp_model_class}(l2_norm_clip=1.0, noise_multiplier=0.5, use_xla=True,
                <standard arguments>)
       ```

       You should use your {dp_model_class} instance with a standard instance
       of `tf.keras.Optimizer` as the optimizer, and a standard reduced loss.
       You do not need to use a differentially private optimizer.

       ```python
       # Use a standard (non-DP) optimizer.
       optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

       # Use a standard reduced loss.
       loss = tf.keras.losses.MeanSquaredError()

       model.compile(optimizer=optimizer, loss=loss)
       model.fit(train_data, train_labels, epochs=1, batch_size=32)
       ```

       """).format(
           base_model='tf.keras.' + cls.__name__,
           short_base_model=cls.__name__,
           dp_model_class='DP' + cls.__name__)

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        use_xla=True,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initializes the DPModelClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch
          gradients).
        noise_multiplier: Ratio of the standard deviation to the clipping
          norm.
        num_microbatches: Number of microbatches.
        use_xla: If `True`, compiles train_step to XLA.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__`
          method.
      """
      super().__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier

      # Given that `num_microbatches` was added as an argument after the fact,
      # this check helps detect unintended calls to the earlier API.
      # In particular, boolean values supplied to `use_xla` in the earlier API
      # will raise an error.
      if isinstance(num_microbatches, bool):
        raise ValueError('Boolean value supplied for `num_microbatches`. '
                         'Did you intend it for `use_xla`?')

      self._num_microbatches = num_microbatches

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
        y_pred = self(x, training=True)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

      grads_list = tape.gradient(loss, self.trainable_variables)
      clipped_grads = self._process_per_example_grads(grads_list)
      return y_pred, loss, clipped_grads

    def train_step(self, data):
      """DP-SGD version of base class method."""
      _, y = data
      batch_size = y.shape[0]

      if self._num_microbatches is None:
        self._num_microbatches = batch_size
      if batch_size % self._num_microbatches != 0:
        raise ValueError('Number of_microbatches must divide batch size.')

      def reshape_fn(x):
        new_shape = (self._num_microbatches,
                     batch_size // self._num_microbatches) + x.shape[1:]
        return tf.reshape(x, new_shape)

      data = tf.nest.map_structure(reshape_fn, data)

      y_pred, _, per_eg_grads = tf.vectorized_map(
          self._compute_per_example_grads, data)

      y_pred = tf.reshape(y_pred, (batch_size) + y_pred.shape[2:])

      grads = tf.nest.map_structure(self._reduce_per_example_grads,
                                    per_eg_grads)
      self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
      self.compiled_metrics.update_state(y, y_pred)
      return {m.name: m.result() for m in self.metrics}

  return DPModelClass


DPModel = make_dp_model_class(tf.keras.Model)
DPSequential = make_dp_model_class(tf.keras.Sequential)
