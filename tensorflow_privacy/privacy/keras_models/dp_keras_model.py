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
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry as lr

_PRIVATIZED_LOSS_NAME = 'privatized_loss'


def make_dp_model_class(cls):
  """Given a subclass of `tf.keras.Model`, returns a DP-SGD version of it."""

  class DPModelClass(cls):  # pylint: disable=missing-class-docstring
    __doc__ = ("""DP subclass of `{base_model}`.

        This can be used as a differentially private replacement for
        {base_model}. This class implements DP-SGD using the standard
        Gaussian mechanism.

        This class also utilizes a faster gradient clipping algorithm if the
        following two conditions hold:
        (i)  the trainable layers of the model are keys in the input
             `layer_registry`,
        (ii) the loss `tf.Tensor` for a given batch of examples is either a
             scalar or a 2D `tf.Tensor` that has only one column
             `(i.e., tf.shape(loss)[1] == 1)` and whose i-th row corresponds to
             the loss of the i-th example.
        This clipping algorithm specifically computes clipped gradients at the
        per-example or per microbatch (when `num_microbatches` is not None)
        level using the layer registry functions in `layer_registry` (see
        clip_grads.py for more information about the algorithm).

        WARNING: with faster gradient clipping, and when num_microbatches is not
        None, the per microbatch loss is assumed to be computed as the mean
        of the loss over the microbatch, or effectively, by reshaping the loss
        from the shape [batch_size, ...] to the shape
        [num_microbatches, batch_size/num_microbatches, ...] and computing the
        mean of the loss over the microbatches. This would require that the loss
        function behaves accordingly. This is true for multiple common
        predefined keras loss functions (e.g. mean_squared_loss,
        binary_crossentropy) but may not hold for custom losses (and how such
        aggregation is done is not exposed by the loss function, unfortunately).
        It is the caller's responsibility to make sure that the loss function
        does behave this way.

        WARNING: This API does not have privacy guarantees for custom
        layer-level losses created by the `layer.add_loss()` API. It does,
        however, support layer regularization losses. All of these layer-level
        losses are found in `model.losses`.

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
        layer_registry: A `LayerRegistry` instance containing functions that
          help compute gradient norms quickly. See
          `tensorflow_privacy.privacy.fast_gradient_clipping.layer_registry` for
          more details.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      super().__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._layer_registry = layer_registry
      self._clipping_loss = None

      # Given that `num_microbatches` was added as an argument after the fact,
      # this check helps detect unintended calls to the earlier API.
      # In particular, boolean values supplied to `use_xla` in the earlier API
      # will raise an error.
      if isinstance(num_microbatches, bool):
        raise ValueError(
            'Boolean value supplied for `num_microbatches`. '
            'Did you intend it for `use_xla`?'
        )
      self._num_microbatches = num_microbatches

      # If all the trainable layers are in the input layer registry, we
      # don't need to use microbatching and can instead use the "fast"
      # chain rule trick for computing per-example gradients (peg).
      if (
          layer_registry is not None
          and gradient_clipping_utils.all_trainable_layers_are_registered(
              self, layer_registry
          )
          and gradient_clipping_utils.has_internal_compute_graph(self)
      ):
        self._enable_fast_peg_computation = True
      else:
        self._enable_fast_peg_computation = False

      if use_xla:
        self.train_step = tf.function(
            self.train_step, experimental_compile=True
        )

    def _process_per_example_grads(self, grads):
      grads_flat = tf.nest.flatten(grads)
      squared_l2_norms = [
          tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
      ]
      global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
      div = tf.maximum(global_norm / self._l2_norm_clip, 1.0)
      clipped_flat = [g / div for g in grads_flat]
      return tf.nest.pack_sequence_as(grads, clipped_flat)

    def _reduce_per_example_grads(self, stacked_grads):
      summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
      if self._noise_multiplier > 0:
        noise_stddev = self._l2_norm_clip * self._noise_multiplier
        noise = tf.random.normal(
            tf.shape(input=summed_grads), stddev=noise_stddev
        )
        summed_grads = summed_grads + noise
      return summed_grads / tf.cast(
          tf.shape(stacked_grads)[0], summed_grads.dtype
      )

    def _compute_per_example_grads(self, data):
      if self._clipping_loss is None:
        self._make_clipping_loss()
      microbatched_x, microbatched_y = data
      with tf.GradientTape() as tape:
        microbatched_y_pred = self(microbatched_x, training=True)
        # NOTE: `self._clipping_loss` does not include any regularization terms.
        microbatched_loss = self._clipping_loss(
            microbatched_y, microbatched_y_pred
        )
      grads_list = tape.gradient(microbatched_loss, self.trainable_variables)
      clipped_grads = self._process_per_example_grads(grads_list)
      return clipped_grads

    def _make_clipping_loss(self):
      """Creates a LossesContainer to be used for clipping.

      To compute the privatized loss, we wrap the model's compiled_loss inside a
      new LossesContainer. This lets us avoid calling model.compiled_loss, which
      appends the loss value to the returned metrics (we want to avoid this as
      the privatized loss does not reflect the true loss and can be misleading).
      """
      losses_container_cls = self.compiled_loss.__class__
      self._clipping_loss = losses_container_cls(
          self.compiled_loss._user_losses,  # pylint:disable=protected-access
          loss_weights=self.compiled_loss._user_loss_weights,  # pylint:disable=protected-access
          output_names=self.output_names,
          total_loss_mean=tf.keras.metrics.Mean(name=_PRIVATIZED_LOSS_NAME),
      )

    def train_step(self, data):
      """DP-SGD version of base class method.

      Uses the "fast" gradient clipping algorithm to generate per-example
      clipped gradients if (i) all the trainable layers of the model are
      registered in the layer_registry input of the model constructor and
      (ii) if the model contains an internal compute graph (e.g., this
      condition is satisfied if the model subclasses the keras.Sequential or
      keras.engine.functional.Functional class).

      If (i) and (ii) above do not hold, then this function clips and aggregates
      gradients at the microbatch level.

      Args:
        data: see the base class.

      Returns:
        See the base class.
      """
      if self._clipping_loss is None:
        self._make_clipping_loss()
      output_metrics = {}
      x, y, weights = tf.keras.utils.unpack_x_y_sample_weight(data)
      if weights is not None:
        raise NotImplementedError(
            'DPModel does not currently support weighted losses.'
        )
      batch_size = tf.shape(y)[0]
      eff_num_microbatches = self._num_microbatches or batch_size

      # Branch based on gradient clipping algorithm.
      if self._enable_fast_peg_computation:
        logging.info(
            'Computing gradients using the fast per-example gradient '
            'norm algorithm.'
        )
        # Computes the per-example gradient norms using a "fast" clipping
        # trick, and uses these norms to clip the per-example gradients.
        # NOTE: Reshaping of the input according to the effective number of
        # microbatches is done here.
        clipped_grads, y_pred, clipping_loss = (
            clip_grads.compute_clipped_gradients_and_outputs(
                self,
                x,
                y,
                self._l2_norm_clip,
                self._layer_registry,
                self._num_microbatches,
                self._clipping_loss,
            )
        )
        output_metrics[_PRIVATIZED_LOSS_NAME] = clipping_loss
        if self._noise_multiplier > 0:
          grads = gradient_clipping_utils.add_aggregate_noise(
              self,
              clipped_grads,
              eff_num_microbatches,
              self._l2_norm_clip,
              self._noise_multiplier,
          )
        else:
          grads = clipped_grads
      else:
        logging.info('Computing gradients using original clipping algorithm.')
        # Computes per-example clipped gradients directly. This is called
        # if at least one of the layers cannot use the "fast" gradient clipping
        # algorithm.
        reshape_fn = lambda z: lr.add_microbatch_axis(z, eff_num_microbatches)
        microbatched_data = tf.nest.map_structure(reshape_fn, data)
        clipped_grads = tf.vectorized_map(
            self._compute_per_example_grads,
            microbatched_data,
        )
        y_pred = self(x, training=True)
        grads = tf.nest.map_structure(
            self._reduce_per_example_grads, clipped_grads
        )

      # Add the values and gradients contributed by regularization losses.
      if self.losses:
        logging.warning(
            'Losses in `model.losses` must be input (batch) independent in '
            'order to obtain the desired differential privacy guarantees.'
        )
        with tf.GradientTape() as tape:
          summed_regularization_loss = tf.add_n(self.losses)
        regularization_grads = tape.gradient(
            summed_regularization_loss,
            self.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        grads = [a + b for (a, b) in zip(grads, regularization_grads)]
        if self._enable_fast_peg_computation:
          output_metrics[_PRIVATIZED_LOSS_NAME] += summed_regularization_loss

      # Log the true loss, including regularization losses.
      self.compiled_loss(y, y_pred, regularization_losses=self.losses)

      # Forward the private gradients to the optimizer and return the results.
      self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
      self.compiled_metrics.update_state(y, y_pred)
      for m in self.metrics:
        output_metrics[m.name] = m.result()

      return output_metrics

  return DPModelClass


DPModel = make_dp_model_class(tf.keras.Model)
DPSequential = make_dp_model_class(tf.keras.Sequential)
