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
              dense_shape=[
                  8,
              ],
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
              dense_shape=[
                  8,
              ],
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
              dense_shape=[
                  8,
              ],
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
              dense_shape=[
                  8,
              ],
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
              dense_shape=[
                  8,
              ],
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
              dense_shape=[
                  8,
              ],
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


if __name__ == "__main__":
  tf.test.main()
