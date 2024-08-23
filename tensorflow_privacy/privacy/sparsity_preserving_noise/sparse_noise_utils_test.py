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
"""Tests for sparse_noise_utils."""

from absl.testing import parameterized
import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils
from tensorflow_privacy.privacy.sparsity_preserving_noise import sparse_noise_utils


class SparseNoiseUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='one_sparse_layer',
          noise_multiplier=1.0,
          sparse_selection_ratio=0.8,
          sparse_selection_contribution_counts=[
              tf.SparseTensor(
                  indices=[[0]],
                  values=[1],
                  dense_shape=[3],
              )
          ],
      ),
      dict(
          testcase_name='multiple_sparse_layer',
          noise_multiplier=1.0,
          sparse_selection_ratio=0.1,
          sparse_selection_contribution_counts=[
              tf.SparseTensor(
                  indices=[[0]],
                  values=[1],
                  dense_shape=[3],
              ),
              tf.SparseTensor(
                  indices=[[0]],
                  values=[1],
                  dense_shape=[3],
              ),
              tf.SparseTensor(
                  indices=[[0]],
                  values=[1],
                  dense_shape=[3],
              ),
          ],
      ),
  )
  def test_split_noise_multiplier(
      self,
      noise_multiplier,
      sparse_selection_ratio,
      sparse_selection_contribution_counts,
  ):
    noise_multiplier_sparse, noise_multiplier_dense = (
        sparse_noise_utils.split_noise_multiplier(
            noise_multiplier,
            sparse_selection_ratio,
            sparse_selection_contribution_counts,
        )
    )
    num_sparse_layers = len(sparse_selection_contribution_counts)

    total_noise_multiplier_sparse = (
        noise_multiplier_sparse / num_sparse_layers**0.5
    )
    self.assertAlmostEqual(
        total_noise_multiplier_sparse,
        sparse_selection_ratio * noise_multiplier_dense,
    )
    total_noise_multiplier = (
        1.0
        / (
            1.0 / total_noise_multiplier_sparse**2
            + 1.0 / noise_multiplier_dense**2
        )
        ** 0.5
    )
    self.assertAlmostEqual(total_noise_multiplier, noise_multiplier)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_sparse_layers',
          noise_multiplier=1.0,
          sparse_selection_ratio=0.5,
          sparse_selection_contribution_counts=[],
          error_message='No sparse selections contribution counts found.',
      ),
      dict(
          testcase_name='sparse_layers_none',
          noise_multiplier=1.0,
          sparse_selection_ratio=0.5,
          sparse_selection_contribution_counts=[None],
          error_message='No sparse selections contribution counts found.',
      ),
      dict(
          testcase_name='zero_ratio',
          noise_multiplier=1.0,
          sparse_selection_ratio=0.0,
          sparse_selection_contribution_counts=[
              tf.SparseTensor(
                  indices=[[0]],
                  values=[1],
                  dense_shape=[3],
              )
          ],
          error_message='Sparse selection ratio must be between 0 and 1.',
      ),
      dict(
          testcase_name='one_ratio',
          noise_multiplier=1.0,
          sparse_selection_ratio=1.0,
          sparse_selection_contribution_counts=[
              tf.SparseTensor(
                  indices=[[0]],
                  values=[1],
                  dense_shape=[3],
              )
          ],
          error_message='Sparse selection ratio must be between 0 and 1.',
      ),
  )
  def test_split_noise_multiplier_errors(
      self,
      noise_multiplier,
      sparse_selection_ratio,
      sparse_selection_contribution_counts,
      error_message,
  ):
    with self.assertRaisesRegex(ValueError, error_message):
      sparse_noise_utils.split_noise_multiplier(
          noise_multiplier,
          sparse_selection_ratio,
          sparse_selection_contribution_counts,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='max_index_0',
          max_index=0,
      ),
      dict(
          testcase_name='max_index_10',
          max_index=10,
      ),
  )
  def test_sample_false_positive_indices_one_prob(self, max_index):
    sampled_indices = (
        sparse_noise_utils.sample_false_positive_indices(max_index, 1.0)
        .numpy()
        .tolist()
    )
    expected_indices = list(range(max_index + 1))
    self.assertEqual(sampled_indices, expected_indices)

  @parameterized.named_parameters(
      dict(
          testcase_name='max_index_0',
          max_index=0,
      ),
      dict(
          testcase_name='max_index_10',
          max_index=10,
      ),
  )
  def test_sample_false_positive_indices_zero_prob(self, max_index):
    sampled_indices = (
        sparse_noise_utils.sample_false_positive_indices(max_index, 0.0)
        .numpy()
        .tolist()
    )
    self.assertEmpty(sampled_indices)

  @parameterized.named_parameters(
      dict(
          testcase_name='max_index_10_prob_50',
          prob=0.5,
          max_index=10,
      ),
      dict(
          testcase_name='max_index_20_prob_25',
          prob=0.25,
          max_index=20,
      ),
      dict(
          testcase_name='max_index_20_prob_75',
          prob=0.75,
          max_index=20,
      ),
  )
  def test_sample_false_positive_indices_random(self, max_index, prob):
    sampled_indices = sparse_noise_utils.sample_false_positive_indices(
        max_index, prob
    )
    sampled_indices = sampled_indices.numpy()

    self.assertLessEqual(np.max(sampled_indices), max_index)
    self.assertGreaterEqual(np.min(sampled_indices), 0)

    self.assertGreater(
        stats.binomtest(k=len(sampled_indices), n=max_index, p=prob).pvalue,
        1e-10,
    )

    bins = np.arange(max_index + 1) + 1
    histogram, _ = np.histogram(sampled_indices, bins=bins)

    num_trials = 10000
    for _ in range(num_trials):
      sampled_indices = sparse_noise_utils.sample_false_positive_indices(
          max_index, prob
      ).numpy()
      histogram += np.histogram(sampled_indices, bins=bins)[0]

    min_pvalue = min(
        stats.binomtest(k=h.item(), n=num_trials, p=prob).pvalue
        for h in histogram
    )
    self.assertGreater(min_pvalue, 1e-10)

  def test_sample_true_positive_indices_empty(self):
    contribution_counts = tf.SparseTensor(
        indices=np.zeros((0, 1), dtype=np.int64),
        values=[],
        dense_shape=[8],
    )
    noise_multiplier = 10.0
    threshold = 2
    sampled_indices = sparse_noise_utils.sample_true_positive_indices(
        contribution_counts, noise_multiplier, threshold
    )
    sampled_indices = list(sampled_indices.numpy())
    expected_indices = []
    self.assertEqual(sampled_indices, expected_indices)

  def test_sample_true_positive_indices_without_noise(self):
    contribution_counts = tf.SparseTensor(
        indices=[[0], [3], [5], [7]],
        values=[3.0, 1.0, 1.0, 2.0],
        dense_shape=[8],
    )
    noise_multiplier = 0.0
    threshold = 2
    sampled_indices = sparse_noise_utils.sample_true_positive_indices(
        contribution_counts, noise_multiplier, threshold
    )
    sampled_indices = list(sampled_indices.numpy())
    expected_indices = [0, 7]
    self.assertEqual(sampled_indices, expected_indices)

  def test_sample_true_positive_indices_with_noise(self):
    contribution_counts = tf.SparseTensor(
        indices=[[0], [3], [5], [7]],
        values=[30.0, 1.0, 1.0, 20.0],
        dense_shape=[8],
    )
    noise_multiplier = 1.0
    threshold = 10
    sampled_indices = sparse_noise_utils.sample_true_positive_indices(
        contribution_counts, noise_multiplier, threshold
    )
    sampled_indices = list(sampled_indices.numpy())
    expected_indices = [0, 7]
    self.assertEqual(sampled_indices, expected_indices)

  def test_batch_size_heuristic(self):
    max_index = 100
    prob = 0.5
    batch_size = sparse_noise_utils._sample_sparse_indices_batch_size_heuristic(
        max_index, prob
    )
    self.assertGreater(batch_size, 0)
    self.assertLess(batch_size, max_index + 1)

  def test_sparse_private_partition_selection_without_noise(self):
    contribution_counts = tf.SparseTensor(
        indices=[[0], [3], [5]],
        values=[2.0, 1.0, 1.0],
        dense_shape=[8],
    )
    noise_multiplier = 0.0
    threshold = 2
    sampled_indices = (
        sparse_noise_utils.sparse_private_partition_selection(
            contribution_counts, noise_multiplier, threshold
        )
        .numpy()
        .tolist()
    )
    expected_indices = [0]
    self.assertEqual(sampled_indices, expected_indices)

  def test_sparse_private_partition_selection_with_noise(self):
    contribution_counts = tf.SparseTensor(
        indices=[[0], [3], [5]],
        values=[50.0, 1.0, 1.0],
        dense_shape=[1000],
    )
    noise_multiplier = 1.0
    threshold = 1
    sampled_indices = (
        sparse_noise_utils.sparse_private_partition_selection(
            contribution_counts, noise_multiplier, threshold
        )
        .numpy()
        .tolist()
    )
    expected_indices = [0]
    self.assertContainsSubset(expected_indices, sampled_indices)
    self.assertGreater(len(sampled_indices), 1)

  def test_remap_indices(self):
    expected_indices = [4, 9, 14]
    indices = tf.constant([1, 5, 10], tf.int64)
    skip_indices = tf.constant([0, 1, 2, 5], tf.int64)
    remapped_indices = sparse_noise_utils._remap_indices(indices, skip_indices)
    self.assertEqual(remapped_indices.numpy().tolist(), expected_indices)

  def test_remap_indices_no_skip(self):
    expected_indices = [1, 5, 10]
    indices = tf.constant([1, 5, 10], tf.int64)
    skip_indices = tf.constant([], tf.int64)
    remapped_indices = sparse_noise_utils._remap_indices(indices, skip_indices)
    self.assertEqual(remapped_indices.numpy().tolist(), expected_indices)

  def test_add_sparse_gradient_noise(self):
    grad = tf.IndexedSlices(
        values=tf.ones((1, 2)),
        indices=tf.constant([0]),
        dense_shape=tf.constant([2, 2]),
    )
    indices = tf.constant([1], dtype=tf.int64)
    noise_stddev = 1.0
    noised_grad = sparse_noise_utils.add_sparse_gradient_noise(
        grad, indices, noise_stddev
    )
    self.assertListEqual(
        noised_grad.indices.numpy().tolist(), indices.numpy().tolist()
    )
    one_index_values = noised_grad.values[0].numpy().tolist()
    self.assertNotEqual(one_index_values, [0.0, 0.0])

  def test_get_contribution_counts(self):
    trainable_vars = [
        tf.Variable(tf.ones((1, 2)), name='var1'),
        tf.Variable(tf.ones((1, 2)), name='var2'),
        tf.Variable(tf.ones((1, 2)), name='var3'),
    ]
    grads = [
        tf.IndexedSlices(
            values=tf.ones((1, 2)),
            indices=tf.constant([0]),
            dense_shape=tf.constant([2, 2]),
        ),
        tf.ones((1, 2)),
        tf.ones((1, 2)),
    ]
    varname_to_contribution_counts_fns = {
        'var1:0': lambda grad: 1.0,
        'var2:0': None,
    }
    contribution_counts = sparse_noise_utils.get_contribution_counts(
        trainable_vars, grads, varname_to_contribution_counts_fns
    )
    expected_contribution_counts = [1.0, None, None]
    self.assertEqual(contribution_counts, expected_contribution_counts)

  def test_add_sparse_noise_without_noise(self):
    grad = tf.IndexedSlices(
        values=tf.ones((3, 4)),
        indices=tf.constant([0, 3, 5]),
        dense_shape=tf.constant([8, 4]),
    )
    contribution_counts = tf.SparseTensor(
        indices=[[0], [3], [5]],
        values=[3.0, 1.0, 2.0],
        dense_shape=[8],
    )
    noised_grad = sparse_noise_utils.add_sparse_noise(
        grad,
        contribution_counts,
        noise_multiplier=0.0,
        noise_multiplier_sparse=0.0,
        l2_norm_clip=1.0,
        threshold=1,
    )
    self.assertEqual(
        noised_grad.indices.numpy().tolist(), grad.indices.numpy().tolist()
    )
    self.assertEqual(
        noised_grad.values.numpy().tolist(), grad.values.numpy().tolist()
    )

  def test_add_sparse_noise_with_noise(self):
    grad = tf.IndexedSlices(
        values=tf.ones((3, 4)),
        indices=tf.constant([0, 3, 5]),
        dense_shape=tf.constant([8, 4]),
    )
    contribution_counts = tf.SparseTensor(
        indices=[[0], [3], [5]],
        values=[10.0, 10.0, 20.0],
        dense_shape=[8],
    )
    noised_grad = sparse_noise_utils.add_sparse_noise(
        grad,
        contribution_counts,
        noise_multiplier=1.0,
        noise_multiplier_sparse=1.0,
        l2_norm_clip=1.0,
        threshold=5,
    )
    self.assertContainsSubset(
        grad.indices.numpy().tolist(),
        noised_grad.indices.numpy().tolist(),
    )
    noised_grad_dense = tf.scatter_nd(
        tf.reshape(noised_grad.indices, (-1, 1)),
        noised_grad.values,
        shape=(8, 4),
    ).numpy()
    noised_grad_valid_indices = noised_grad_dense[grad.indices.numpy()]
    self.assertTrue(
        np.all(np.not_equal(noised_grad_valid_indices, grad.values.numpy()))
    )

  def test_extract_varname_to_contribution_counts_fns(self):
    def fn1(_):
      return 1.0

    def fn2(_):
      return 2.0

    var1 = tf.Variable(tf.ones((1, 2)), name='var1')
    var2 = tf.Variable(tf.ones((1, 2)), name='var2')
    var3 = tf.Variable(tf.ones((1, 2)), name='var3')

    registry_fn_outputs_list = [
        gradient_clipping_utils.RegistryGeneratorFunctionOutput(
            layer_id='layer1',
            layer_vars=[var1],
            layer_sqr_norm_fn=None,
            layer_trainable_weights=[var1],
            varname_to_count_contribution_fn=None,
        ),
        gradient_clipping_utils.RegistryGeneratorFunctionOutput(
            layer_id='layer2',
            layer_vars=[var2],
            layer_sqr_norm_fn=None,
            layer_trainable_weights=[var2],
            varname_to_count_contribution_fn={
                'var2:0': [fn2],
            },
        ),
        gradient_clipping_utils.RegistryGeneratorFunctionOutput(
            layer_id='layer3',
            layer_vars=[var3],
            layer_sqr_norm_fn=None,
            layer_trainable_weights=[var3],
            varname_to_count_contribution_fn={
                'var3:0': [fn1],
            },
        ),
    ]
    expected_varname_to_contribution_counts_fns = {
        'var2:0': [fn2],
        'var3:0': [fn1],
    }
    varname_to_contribution_counts_fns = (
        sparse_noise_utils.extract_varname_to_contribution_counts_fns(
            registry_fn_outputs_list,
            trainable_vars=None,
        )
    )
    self.assertEqual(
        varname_to_contribution_counts_fns,
        expected_varname_to_contribution_counts_fns,
    )

  def test_extract_varname_to_contribution_counts_fns_duplicate_varnames(self):
    def fn1(_):
      return 1.0

    def fn2(_):
      return 2.0

    var1 = tf.Variable(tf.ones((1, 2)), name='var1')
    var2 = tf.Variable(tf.ones((1, 2)), name='var1')

    registry_fn_outputs_list = [
        gradient_clipping_utils.RegistryGeneratorFunctionOutput(
            layer_id='layer1',
            layer_vars=[var1],
            layer_sqr_norm_fn=None,
            layer_trainable_weights=[var1],
            varname_to_count_contribution_fn={
                'var1:0': [fn1],
            },
        ),
        gradient_clipping_utils.RegistryGeneratorFunctionOutput(
            layer_id='layer2',
            layer_vars=[var2],
            layer_sqr_norm_fn=None,
            layer_trainable_weights=[var2],
            varname_to_count_contribution_fn={
                'var1:0': [fn2],
            },
        ),
    ]

    with self.assertRaises(ValueError):
      sparse_noise_utils.extract_varname_to_contribution_counts_fns(
          registry_fn_outputs_list,
          trainable_vars=None,
      )


if __name__ == '__main__':
  tf.test.main()
