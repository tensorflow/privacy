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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils


class ModelForwardPassTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      input_packing_type=[None, tuple, list, dict],
      output_packing_type=[None, tuple, list, dict],
  )
  def test_outputs_are_consistent(
      self, input_packing_type, output_packing_type
  ):
    num_dims = 3
    num_inputs = 1 if input_packing_type is None else 2
    num_outputs = 1 if output_packing_type is None else 2
    sample_inputs = [tf.keras.Input((num_dims,)) for i in range(num_inputs)]
    temp_sum = tf.stack(sample_inputs, axis=0)
    sample_outputs = [
        tf.multiply(temp_sum, float(i + 1.0)) for i in range(num_outputs)
    ]
    sample_x_batch = [
        tf.multiply(tf.range(num_dims, dtype=tf.float32), float(i + 1.0))
        for i in range(num_inputs)
    ]

    # Pack inputs.
    if input_packing_type is None:
      inputs = sample_inputs[0]
      x_batch = sample_x_batch[0]
    elif input_packing_type is not dict:
      inputs = input_packing_type(sample_inputs)
      x_batch = input_packing_type(sample_x_batch)
    else:
      inputs = {}
      x_batch = {}
      keys = [str(i) for i in range(len(sample_inputs))]
      for k, v1, v2 in zip(keys, sample_inputs, sample_x_batch):
        inputs[k] = v1
        x_batch[k] = v2

    # Pack outputs.
    if output_packing_type is None:
      outputs = sample_outputs[0]
    elif output_packing_type is not dict:
      outputs = output_packing_type(sample_outputs)
    else:
      outputs = {}
      keys = [str(i) for i in range(len(sample_outputs))]
      for k, v in zip(keys, sample_outputs):
        outputs[k] = v

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    computed_outputs, _ = gradient_clipping_utils.model_forward_pass(
        model,
        x_batch,
    )
    true_outputs = model(x_batch)
    self.assertAllClose(computed_outputs, true_outputs)


class DepthFirstBackwardPassTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      depth=[1, 2],
      input_packing_type=[None, tuple, list, dict],
      output_packing_type=[None, tuple, list, dict],
  )
  def test_layer_function(self, depth, input_packing_type, output_packing_type):
    num_dims = 3
    num_units = 5
    num_inputs = 1 if input_packing_type is None else 2
    num_outputs = 1 if output_packing_type is None else 2
    sample_inputs = [tf.keras.Input((num_dims,)) for i in range(num_inputs)]
    temp_sum = tf.stack(sample_inputs, axis=0)
    sample_sum = [
        tf.multiply(temp_sum, float(i + 1.0)) for i in range(num_outputs)
    ]
    sample_outputs = sample_sum
    for _ in range(depth):
      sample_outputs = [
          tf.keras.layers.Dense(num_units)(t) for t in sample_outputs
      ]

    # Pack inputs.
    if input_packing_type is None:
      inputs = sample_inputs[0]
    elif input_packing_type is not dict:
      inputs = input_packing_type(sample_inputs)
    else:
      inputs = {}
      keys = [str(i) for i in range(len(sample_inputs))]
      for k, v in zip(keys, sample_inputs):
        inputs[k] = v

    # Pack outputs.
    if output_packing_type is None:
      outputs = sample_outputs[0]
    elif output_packing_type is not dict:
      outputs = output_packing_type(sample_outputs)
    else:
      outputs = {}
      keys = [str(i) for i in range(len(sample_outputs))]
      for k, v in zip(keys, sample_outputs):
        outputs[k] = v

    # Append the trainable layers into a list.
    layer_list = []

    def layer_function(layer):
      if layer.trainable_variables:
        layer_list.append(layer)

    # Run the traversal and verify the outputs that are relevant to
    # the above layer function.
    gradient_clipping_utils.depth_first_backward_pass(outputs, layer_function)
    self.assertLen(layer_list, num_outputs * depth)
    for l in layer_list:
      self.assertIsInstance(l, tf.keras.layers.Dense)


if __name__ == '__main__':
  tf.test.main()
