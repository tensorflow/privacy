# Copyright 2023, The TensorFlow Authors. All Rights Reserved.
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
"""Tests DPModel in distributed training."""
import contextlib
import multiprocessing
import os
import sys
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry
from tensorflow_privacy.privacy.keras_models import dp_keras_model


ds_combinations = tf.__internal__.distribute.combinations


STRATEGIES = [
    ds_combinations.one_device_strategy,
    ds_combinations.parameter_server_strategy_1worker_2ps_cpu,
]


def get_data(n, clip_norm):
  # Data is for hidden weights of w* = [3, 1] and bias of b* = 2.
  # For mean-squared loss, we have:
  # loss = (data * w + b - label)^2 = (data * (w - w*) + (b - b*))^2
  # Let T = (data * (w - w*) + (b - b*)), we have:
  #   grad_w = 2 * T * data
  #   grad_b = 2 * T
  # For w = [0, 0], b = 0:
  #   For data = [3, 4],  T = -15, grad_w = [-90, -120],  grad_b = -30
  #   For data = [1, -1], T = -4,  grad_w = [-8, 8],      grad_b = -8
  data = np.array([[3., 4.], [1., -1.]])
  labels = np.matmul(data, [[3], [1]]) + 2
  data, labels = np.tile(data, (n, 1)), np.tile(labels, (n, 1))

  def clip_grad(grad):
    norm = np.linalg.norm(grad)
    if norm <= clip_norm:
      return grad
    return grad / norm * clip_norm

  grad1 = clip_grad(np.array([-90., -120., -30.]))
  grad2 = clip_grad(np.array([-8., 8., -8.]))
  grad = np.mean(np.vstack([grad1, grad2]), axis=0)
  return data, labels, grad


class DPKerasModelDistributedTest(parameterized.TestCase, tf.test.TestCase):

  @ds_combinations.generate(
      tf.__internal__.test.combinations.combine(
          strategy=STRATEGIES,
          mode="eager",
          # Clip norm corresponding to no-clipping, clipping one gradient, and
          # clipping both gradients.
          clip_norm=[1e5, 200., 1.],
          model_or_sequential=["model", "sequential"],
          fast_clipping=[False, True],
      )
  )
  def test_training_works(
      self, strategy, clip_norm, model_or_sequential, fast_clipping
  ):
    if model_or_sequential == "sequential" and fast_clipping:
      self.skipTest("Fast clipping does not work for DPSequential.")

    if isinstance(strategy, tf.distribute.OneDeviceStrategy):
      strategy_scope = contextlib.nullcontext()
    else:
      strategy_scope = strategy.scope()

    n = 10
    x, y, expected_grad = get_data(n, clip_norm)

    def make_model():
      dense_layer = tf.keras.layers.Dense(
          1, kernel_initializer="zeros", bias_initializer="zeros"
      )
      dp_kwargs = dict(
          l2_norm_clip=clip_norm,
          noise_multiplier=0.0,
          num_microbatches=None,
          use_xla=False,
          layer_registry=layer_registry.make_default_layer_registry()
          if fast_clipping
          else None,
      )
      if model_or_sequential == "sequential":
        model = dp_keras_model.DPSequential(
            layers=[dense_layer],
            **dp_kwargs,
        )
      else:
        inputs = tf.keras.layers.Input((2,))
        outputs = dense_layer(inputs)
        model = dp_keras_model.DPModel(
            inputs=inputs, outputs=outputs, **dp_kwargs
        )
      return model

    with strategy_scope:
      model = make_model()
      self.assertEqual(model._enable_fast_peg_computation, fast_clipping)
      lr = 0.01
      opt = tf.keras.optimizers.SGD(learning_rate=lr)
      loss = tf.keras.losses.MeanSquaredError(
          reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
      )
      model.compile(loss=loss, optimizer=opt)
      history = model.fit(
          x=x,
          y=y,
          epochs=1,
          batch_size=x.shape[0],
          steps_per_epoch=1,
      )
      self.assertIn("loss", history.history)
      self.assertEqual(opt.iterations.numpy(), 1)
      model_weights = model.get_weights()

      expected_val = -expected_grad * lr
      self.assertAllClose(model_weights[0], expected_val[:2].reshape(-1, 1))
      self.assertAllClose(model_weights[1], [expected_val[2]])


def _set_spawn_exe_path():
  """Set the path to the executable for spawned processes.

  This utility searches for the binary the parent process is using, and sets
  the executable of multiprocessing's context accordingly.
  It is branched from tensorflow/python/distribute/multi_process_lib.py, the
  only change being that it additionally looks under "org_tensorflow_privacy".
  """
  if sys.argv[0].endswith(".py"):

    def guess_path(package_root):
      # If all we have is a python module path, we'll need to make a guess for
      # the actual executable path.
      if "bazel-out" in sys.argv[0] and package_root in sys.argv[0]:
        # Guess the binary path under bazel. For target
        # //tensorflow/python/distribute:input_lib_test_multiworker_gpu, the
        # argv[0] is in the form of
        # /.../tensorflow/python/distribute/input_lib_test.py
        # and the binary is
        # /.../tensorflow/python/distribute/input_lib_test_multiworker_gpu
        package_root_base = sys.argv[0][: sys.argv[0].rfind(package_root)]
        binary = os.environ["TEST_TARGET"][2:].replace(":", "/", 1)
        possible_path = os.path.join(package_root_base, package_root, binary)
        if os.access(possible_path, os.X_OK):
          return possible_path
      return None

    path = (
        guess_path("org_tensorflow")
        or guess_path("org_keras")
        or guess_path("org_tensorflow_privacy")
    )
    if path is not None:
      sys.argv[0] = path
      multiprocessing.get_context().set_executable(sys.argv[0])


if __name__ == "__main__":
  _set_spawn_exe_path()
  tf.__internal__.distribute.multi_process_runner.test_main()
