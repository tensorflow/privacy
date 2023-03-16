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
"""Tests DPSparseKerasSGDOptimizer in distributed training."""
import contextlib
import multiprocessing
import os
import sys
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras_sparse

ds_combinations = tf.__internal__.distribute.combinations


STRATEGIES = [
    ds_combinations.one_device_strategy,
    ds_combinations.parameter_server_strategy_1worker_2ps_cpu,
]


class DistributedTrainingTest(parameterized.TestCase, tf.test.TestCase):

  @ds_combinations.generate(
      tf.__internal__.test.combinations.combine(
          strategy=STRATEGIES, mode="eager"
      )
  )
  def test_training_works(self, strategy):
    if isinstance(strategy, tf.distribute.OneDeviceStrategy):
      strategy_scope = contextlib.nullcontext()
    else:
      strategy_scope = strategy.scope()

    def make_model():
      inputs = tf.keras.Input((1000,))
      dense = tf.keras.layers.Dense(
          units=1, use_bias=False, kernel_initializer=tf.initializers.ones()
      )
      outputs = dense(inputs)
      return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    x = tf.ones(shape=[5000, 1000])
    y = tf.zeros(shape=[5000])
    with strategy_scope:
      model = make_model()
      clip = 100.0
      noise_mult = 0.01
      acc_steps = 5
      batch_size = 10
      opt = dp_optimizer_keras_sparse.DPSparseKerasSGDOptimizer(
          l2_norm_clip=clip,
          noise_multiplier=noise_mult,
          gradient_accumulation_steps=acc_steps,
          learning_rate=0.001,
      )
      model.compile(
          loss=tf.keras.losses.MeanAbsoluteError(
              reduction=tf.keras.losses.Reduction.NONE
          ),
          optimizer=opt,
      )
      history = model.fit(
          x=x,
          y=y,
          epochs=2,
          steps_per_epoch=500,
          batch_size=batch_size,
      )
      self.assertIn("loss", history.history)
      # total steps: 1000 (2 epochs, 500 steps/epoch)
      # accumulation steps: 5
      # expected_iterations = total steps / accumulation steps
      expected_iterations = 1000 / acc_steps  # = 200
      # The loss is |w.x - y| (where w is the dense layer weights).
      # The gradient is sign(w.x - y)x. With the choice of x, y, the gradient
      # becomes x.
      # So each gradient update is w <- w - learning_rate*1 + noise
      expected_params = 1 - 0.001 * expected_iterations
      expected_noise = (
          0.001
          * clip
          * noise_mult
          * np.sqrt(expected_iterations)
          / (acc_steps * batch_size)
      )
      self.assertEqual(opt.iterations.numpy(), expected_iterations)
      self.assertAllClose(
          np.mean(model.trainable_variables[0].numpy()),
          expected_params,  # 0.8
          # stddev = expected_noise/√1000 (since we're averaging 1000 samples)
          # we set atol to 4 stddev
          atol=4 * expected_noise / np.sqrt(1000),  # 0.0358
      )
      self.assertAllClose(
          np.std(model.trainable_variables[0].numpy()),
          expected_noise,  # 0.2828
          atol=4 * expected_noise / np.sqrt(1000),  # 0.0358
      )


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
