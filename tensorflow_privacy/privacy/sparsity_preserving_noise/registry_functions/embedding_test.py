# Copyright 2024, The TensorFlow Authors.
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
"""Tests for embedding."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_privacy.privacy.sparsity_preserving_noise.registry_functions import embedding


class EmbeddingLayerWithMultipleTrainableVariables(tf.keras.layers.Embedding):

  def build(self, input_shape):
    self.some_other_variable = self.add_weight(
        name="some_other_variable",
        shape=(10, 10),
        trainable=True,
    )
    super().build(input_shape)


class EmbeddingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="rank2_input",
          input_ids=tf.constant([[0], [0], [4], [2]]),
          num_microbatches=None,
          vocab_size=8,
          expected_contribution_counts=tf.SparseTensor(
              indices=[[0], [2], [4]],
              values=tf.constant([2.0, 1.0, 1.0], dtype=tf.float64),
              dense_shape=[8],
          ),
      ),
      dict(
          testcase_name="rank2_multi_input",
          input_ids=tf.constant([[0, 2], [0, 2], [4, 5], [2, 3]]),
          num_microbatches=None,
          vocab_size=8,
          expected_contribution_counts=tf.SparseTensor(
              indices=[[0], [2], [3], [4], [5]],
              values=tf.constant([1.0, 1.5, 0.5, 0.5, 0.5], dtype=tf.float64),
              dense_shape=[8],
          ),
      ),
      dict(
          testcase_name="rank3_input",
          input_ids=tf.constant(
              [[[0], [2]], [[0], [2]], [[4], [5]], [[2], [3]]]
          ),
          num_microbatches=None,
          vocab_size=8,
          expected_contribution_counts=tf.SparseTensor(
              indices=[[0], [2], [3], [4], [5]],
              values=tf.constant([1.0, 1.5, 0.5, 0.5, 0.5], dtype=tf.float64),
              dense_shape=[8],
          ),
      ),
      dict(
          testcase_name="ragged_input",
          input_ids=tf.ragged.constant([[0, 2], [2], [2, 3, 4], [4, 5]]),
          num_microbatches=None,
          vocab_size=8,
          expected_contribution_counts=tf.SparseTensor(
              indices=[[0], [2], [3], [4], [5]],
              values=tf.constant(
                  [0.5, 1.5 + 1.0 / 3, 1.0 / 3, 0.5 + 1.0 / 3, 0.5],
                  dtype=tf.float64,
              ),
              dense_shape=[8],
          ),
      ),
      dict(
          testcase_name="rank2_input_num_microbatches_2",
          input_ids=tf.constant([[0], [0], [4], [2]]),
          num_microbatches=2,
          vocab_size=8,
          expected_contribution_counts=tf.SparseTensor(
              indices=[[0], [2], [4]],
              values=tf.constant([1.0, 0.5, 0.5], dtype=tf.float64),
              dense_shape=[8],
          ),
      ),
      dict(
          testcase_name="ragged_input_num_microbatches_2",
          input_ids=tf.ragged.constant([[0, 2], [2], [2, 3, 4], [4, 5]]),
          num_microbatches=2,
          vocab_size=8,
          expected_contribution_counts=tf.SparseTensor(
              indices=[[0], [2], [3], [4], [5]],
              values=tf.constant(
                  [1.0 / 3, 2.0 / 3 + 1.0 / 5, 1.0 / 5, 2.0 / 5, 1.0 / 5],
                  dtype=tf.float64,
              ),
              dense_shape=[8],
          ),
      ),
  )
  def test_embedding_layer_contribution_histogram_fn(
      self,
      input_ids,
      expected_contribution_counts,
      vocab_size,
      num_microbatches,
  ):
    grad = None
    contribution_counts = embedding.embedding_layer_contribution_histogram_fn(
        grad, input_ids, vocab_size, num_microbatches
    )
    tf.debugging.assert_equal(
        tf.sparse.to_dense(contribution_counts),
        tf.sparse.to_dense(expected_contribution_counts),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="input_None",
          input_ids=None,
      ),
      dict(
          testcase_name="input_SparseTensor",
          input_ids=tf.SparseTensor(
              indices=[[0, 0]],
              values=[0],
              dense_shape=(3, 8),
          ),
      ),
      dict(
          testcase_name="input_list",
          input_ids=[[0], [0], [1], [2]],
      ),
      dict(
          testcase_name="num_microbatches_not_divisible",
          input_ids=tf.constant([[0], [0], [4], [2]]),
          num_microbatches=3,
      ),
  )
  def test_embedding_layer_contribution_histogram_fn_errors(
      self, input_ids, num_microbatches=None
  ):
    with self.assertRaises(
        (NotImplementedError, ValueError, tf.errors.InvalidArgumentError)
    ):
      embedding.embedding_layer_contribution_histogram_fn(
          None, input_ids, 8, num_microbatches
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="input_kwargs",
          error_message="Embedding layer calls should not receive kwargs.",
          input_kwargs={"foo": "bar"},
      ),
      dict(
          testcase_name="input_args_more_than_one",
          error_message="Only layer inputs of length 1 are permitted.",
          input_args=[tf.constant([0]), tf.constant([0])],
      ),
      dict(
          testcase_name="input_args_none",
          error_message="Only layer inputs of length 1 are permitted.",
      ),
      dict(
          testcase_name="input_sparse",
          error_message="Sparse input tensors are not supported.",
          input_args=[
              tf.SparseTensor(indices=[[0, 0]], values=[0], dense_shape=(3, 8))
          ],
      ),
      dict(
          testcase_name="layer_one_hot_matmul",
          error_message=(
              "The experimental embedding feature '_use_one_hot_matmul' is not"
              " supported."
          ),
          input_args=[tf.constant([0])],
          layer_kwargs={"use_one_hot_matmul": True},
      ),
      dict(
          testcase_name="layer_sparse",
          error_message="Sparse output tensors are not supported.",
          input_args=[tf.constant([0])],
          layer_kwargs={"sparse": True},
      ),
  )
  def test_embedding_layer_contribution_histogram_errors(
      self,
      error_message,
      input_args=None,
      input_kwargs=None,
      layer_kwargs=None,
  ):
    layer_kwargs = layer_kwargs or {}
    layer = tf.keras.layers.Embedding(input_dim=8, output_dim=4, **layer_kwargs)
    layer.build(input_shape=(None, 1))
    with self.assertRaisesRegex(
        (NotImplementedError, ValueError),
        error_message,
    ):
      embedding.embedding_layer_contribution_histogram(
          layer, input_args, input_kwargs, None
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_variables",
          build=False,  # unbuilt layer has no trainable variables
      ),
      dict(
          testcase_name="two_variables",
          build=True,
      ),
  )
  def test_embedding_layer_contribution_histogram_embedding_layer_invalid_trainable_variables(
      self, build
  ):
    input_args = [tf.constant([0])]
    input_kwargs = {}
    layer = EmbeddingLayerWithMultipleTrainableVariables(
        input_dim=8, output_dim=4
    )
    if build:
      layer.build(input_shape=(None, 1))
    with self.assertRaisesRegex(
        ValueError,
        "Embedding layer must have exactly one trainable variable.",
    ):
      embedding.embedding_layer_contribution_histogram(
          layer, input_args, input_kwargs
      )

  def test_embedding_layer_contribution_histogram_embedding(self):
    input_args = [tf.constant([[0], [0], [4], [2]])]
    input_kwargs = {}
    layer = tf.keras.layers.Embedding(input_dim=8, output_dim=4)
    layer.build(input_shape=(None, 1))

    contribution_counts_fn_dict = (
        embedding.embedding_layer_contribution_histogram(
            layer, input_args, input_kwargs
        )
    )
    self.assertEqual(list(contribution_counts_fn_dict.keys()), ["embeddings:0"])
    contribution_counts_fn = contribution_counts_fn_dict["embeddings:0"]
    dummy_gradient = None
    contribution_counts = contribution_counts_fn(dummy_gradient)
    expected_contribution_counts = tf.SparseTensor(
        indices=[[0], [2], [4]],
        values=tf.constant([2.0, 1.0, 1.0], dtype=tf.float64),
        dense_shape=[8],
    )
    tf.debugging.assert_equal(
        tf.sparse.to_dense(contribution_counts),
        tf.sparse.to_dense(expected_contribution_counts),
    )


if __name__ == "__main__":
  tf.test.main()
