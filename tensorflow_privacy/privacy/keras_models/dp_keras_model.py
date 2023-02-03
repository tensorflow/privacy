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

from absl import logging
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import clip_grads
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils


def make_dp_model_class(cls):
  """Given a subclass of `tf.keras.Model`, returns a DP-SGD version of it."""

  class DPModelClass(cls):  # pylint: disable=missing-class-docstring
    __doc__ = (
        """DP subclass of `{base_model}`.

       This can be used as a differentially private replacement for
       {base_model}. This class implements DP-SGD using the standard
       Gaussian mechanism.

       This class also utilizes a faster gradient clipping algorithm if the
       following two conditions hold:
        (i)  the trainable layers of the model are keys in the `dict` input
             `layer_registry`,
        (ii) the loss `tf.Tensor` for a given batch of examples is either a
             scalar or a 2D `tf.Tensor` that has only one column
             `(i.e., tf.shape(loss)[1] == 1)` and whose i-th row corresponds to
             the loss of the i-th example.
       This clipping algorithm specifically computes clipped gradients at the
       per-example level using the layer registry functions in `layer_registry`
       (see clip_grads.py for more information about the algorithm). In this
       setting, microbatching is not used (it is equivalent to
       `num_microbatches == batch_size`), and the input `num_microbatches`
       is ignored.

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

       """
    ).format(
        base_model='tf.keras.' + cls.__name__,
        short_base_model=cls.__name__,
        dp_model_class='DP' + cls.__name__,
    )

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        use_xla=True,
        layer_registry=None,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs,
    ):
      """Initializes the DPModelClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
        noise_multiplier: Ratio of the standard deviation to the clipping norm.
        num_microbatches: Number of microbatches.
        use_xla: If `True`, compiles train_step to XLA.
        layer_registry: A `dict` of layers that support "fast" gradient norm
          computations. The key is the class of the layer and the value is a
          function that returns a `tuple` `(output, sqr_grad_norms, vars)`,
          where `output` is the pre-activator tensor, `sqr_grad_norms` is
          related to the squared norms of a layer's pre-activation tensor, and
          `vars` are relevant trainable weights (see
          `layer_registry_factories.py` for examples).
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      super().__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._layer_registry = layer_registry

      # Given that `num_microbatches` was added as an argument after the fact,
      # this check helps detect unintended calls to the earlier API.
      # In particular, boolean values supplied to `use_xla` in the earlier API
      # will raise an error.
      if isinstance(num_microbatches, bool):
        raise ValueError('Boolean value supplied for `num_microbatches`. '
                         'Did you intend it for `use_xla`?')

      # If all the trainable layers are in the input layer registry, we
      # don't need to use microbatching and can instead use the "fast"
      # chain rule trick for computing per-example gradients (peg).
      if (
          layer_registry is not None
          and gradient_clipping_utils._all_trainable_layers_are_registered(
              self, layer_registry
          )
          and gradient_clipping_utils._has_internal_compute_graph(self)
      ):
        if num_microbatches is not None:
          raise ValueError(
              'Cannot initialize a model where num_microbatches '
              'is not `None` and all trainable layers are '
              'registered in layer_registry.'
          )
        self._num_microbatches = None
        self._enable_fast_peg_computation = True
      else:
        self._num_microbatches = num_microbatches
        self._enable_fast_peg_computation = False

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
      """DP-SGD version of base class method.

      Uses the "fast" gradient clipping algorithm to generate per-example
      clipped gradients if (i) all the trainable layers of the model are
      registered in the layer_registry input of the model constructor and
      (ii) if the model contains an internal compute graph (e.g., this
      condition is satisfied if the model subclasses the keras.Sequential or
      keras.engine.functional.Functional class).

      If (i) and (ii) above do not hold, then clips and aggregates
      gradients at the microbatch level.

      Args:
        data: see the base class.

      Returns:
        See the base class.
      """
      if self._enable_fast_peg_computation:
        logging.info(
            'Computing gradients using the fast per-example gradient '
            'norm algorithm.'
        )
        # Computes the per-example gradient norms using a "fast" clipping
        # trick, and uses these norms to clip the per-example gradients.
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_pred, clipped_grads = clip_grads.compute_pred_and_clipped_gradients(
            self, x, y, self._l2_norm_clip, self._layer_registry
        )
        grads = gradient_clipping_utils.add_aggregate_noise(
            self, x, clipped_grads, self._l2_norm_clip, self._noise_multiplier
        )
      else:
        logging.info('Computing gradients using microbatching.')
        # Computes per-example clipped gradients directly. This is called
        # if at least one of the layers cannot use the "fast" gradient clipping
        # algorithm.
        # TODO(wkong): check if the following is valid with sample weights.
        _, y = data
        batch_size = y.shape[0]

        if self._num_microbatches is None:
          self._num_microbatches = batch_size
        if batch_size % self._num_microbatches != 0:
          raise ValueError('Number of_microbatches must divide batch size.')

        def reshape_fn(x):
          new_shape = (
              self._num_microbatches,
              batch_size // self._num_microbatches,
          ) + x.shape[1:]
          return tf.reshape(x, new_shape)

        data = tf.nest.map_structure(reshape_fn, data)

        y_pred, _, per_eg_grads = tf.vectorized_map(
            self._compute_per_example_grads, data
        )

        y_pred = tf.reshape(y_pred, (batch_size) + y_pred.shape[2:])

        grads = tf.nest.map_structure(
            self._reduce_per_example_grads, per_eg_grads
        )

      # Forward the private gradients to the optimizer and return the results.
      self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
      self.compiled_metrics.update_state(y, y_pred)
      return {m.name: m.result() for m in self.metrics}

  return DPModelClass


DPModel = make_dp_model_class(tf.keras.Model)
DPSequential = make_dp_model_class(tf.keras.Sequential)
