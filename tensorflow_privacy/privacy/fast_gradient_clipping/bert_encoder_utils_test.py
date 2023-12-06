# Copyright 2023, The TensorFlow Authors.
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
"""Tests of `bert_encoder_utils.py`."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
from tensorflow_privacy.privacy.fast_gradient_clipping import bert_encoder_utils


def compute_bert_sample_inputs(
    batch_size, sequence_length, vocab_size, num_types
):
  """Returns a set of BERT encoder inputs."""
  word_id_sample = np.random.randint(
      vocab_size, size=(batch_size, sequence_length)
  )
  mask_sample = np.random.randint(2, size=(batch_size, sequence_length))
  type_id_sample = np.random.randint(
      num_types,
      size=(batch_size, sequence_length),
  )
  return [word_id_sample, mask_sample, type_id_sample]


def get_small_bert_encoder_and_sample_inputs(dict_outputs=False):
  """Returns a small BERT encoder for testing."""
  hidden_size = 2
  vocab_size = 3
  num_types = 4
  max_sequence_length = 5
  inner_dense_units = 6
  output_range = 1
  num_heads = 2
  num_transformer_layers = 3
  seed = 777

  bert_encoder = tfm.nlp.networks.BertEncoder(
      vocab_size=vocab_size,
      hidden_size=hidden_size,
      num_attention_heads=num_heads,
      num_layers=num_transformer_layers,
      max_sequence_length=max_sequence_length,
      inner_dim=inner_dense_units,
      type_vocab_size=num_types,
      output_range=output_range,
      initializer=tf.keras.initializers.GlorotUniform(seed),
      dict_outputs=dict_outputs,
  )

  batch_size = 3
  bert_sample_inputs = compute_bert_sample_inputs(
      batch_size,
      max_sequence_length,
      vocab_size,
      num_types,
  )

  return bert_encoder, bert_sample_inputs


def get_shared_trainable_variables(model1, model2):
  """Returns the shared trainable variables (by name) between models."""
  common_names = {v.name for v in model1.trainable_variables} & {
      v.name for v in model2.trainable_variables
  }
  tvars1 = [v for v in model1.trainable_variables if v.name in common_names]
  tvars2 = [v for v in model2.trainable_variables if v.name in common_names]
  return tvars1, tvars2


def custom_reduced_loss(y_batch, y_pred):
  del y_batch
  # Create a loss multiplier to avoid small gradients.
  large_value_multiplier = 1e10
  sqr_outputs = []
  for t in y_pred:
    reduction_axes = tf.range(1, len(t.shape))
    sqr_outputs.append(tf.reduce_sum(tf.square(t), axis=reduction_axes))
  sqr_tsr = tf.stack(sqr_outputs, axis=1)
  return large_value_multiplier * tf.reduce_sum(sqr_tsr, axis=1)


class BertEncoderUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_outputs_are_equal(self):
    true_encoder, sample_inputs = get_small_bert_encoder_and_sample_inputs()
    unwrapped_encoder = bert_encoder_utils.get_unwrapped_bert_encoder(
        true_encoder
    )
    true_outputs = true_encoder(sample_inputs)
    computed_outputs = unwrapped_encoder(sample_inputs)
    self.assertAllClose(true_outputs, computed_outputs)

  def test_shared_trainable_variables_are_equal(self):
    true_encoder, sample_inputs = get_small_bert_encoder_and_sample_inputs()
    unwrapped_encoder = bert_encoder_utils.get_unwrapped_bert_encoder(
        true_encoder
    )
    # Initializes the trainable variable shapes.
    true_encoder(sample_inputs)
    unwrapped_encoder(sample_inputs)
    # The official BERT encoder may initialize trainable variables that are
    # not used in a model forward pass. Hence, they are invisible when we
    # try to unwrapping layers using our utility function.
    true_vars, computed_vars = get_shared_trainable_variables(
        true_encoder, unwrapped_encoder
    )
    self.assertAllClose(true_vars, computed_vars)

  def test_shared_gradients_are_equal(self):
    true_encoder, sample_inputs = get_small_bert_encoder_and_sample_inputs()
    unwrapped_encoder = bert_encoder_utils.get_unwrapped_bert_encoder(
        true_encoder
    )
    # Create a loss multiplier to avoid small gradients.
    dummy_labels = None
    with tf.GradientTape(persistent=True) as tape:
      true_outputs = true_encoder(sample_inputs)
      true_sqr_sum = tf.reduce_sum(
          custom_reduced_loss(dummy_labels, true_outputs)
      )
      computed_outputs = unwrapped_encoder(sample_inputs)
      computed_sqr_sum = tf.reduce_sum(
          custom_reduced_loss(dummy_labels, computed_outputs)
      )
    # The official BERT encoder may initialize trainable variables that are
    # not used in a model forward pass. Hence, they are invisible when we
    # try to unwrapping layers using our utility function.
    true_vars, computed_vars = get_shared_trainable_variables(
        true_encoder, unwrapped_encoder
    )
    true_grads = tape.gradient(true_sqr_sum, true_vars)
    computed_grads = tape.gradient(computed_sqr_sum, computed_vars)
    self.assertEqual(len(true_grads), len(computed_grads))
    for g1, g2 in zip(true_grads, computed_grads):
      self.assertEqual(type(g1), type(g2))
      if isinstance(g1, tf.IndexedSlices):
        self.assertAllClose(g1.values, g2.values)
        self.assertAllEqual(g2.indices, g2.indices)
      else:
        self.assertAllClose(g1, g2)


if __name__ == '__main__':
  tf.test.main()
