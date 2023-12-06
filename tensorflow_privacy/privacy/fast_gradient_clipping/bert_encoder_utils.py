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
"""Utility functions for manipulating official Tensorflow BERT encoders."""

import tensorflow as tf
import tensorflow_models as tfm
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils


def dedup_bert_encoder(input_bert_encoder: tfm.nlp.networks.BertEncoder):
  """Deduplicates the layer names in a BERT encoder."""

  def _dedup(layer, attr_name, new_name):
    sublayer = getattr(layer, attr_name)
    if sublayer is None:
      return
    else:
      sublayer_config = sublayer.get_config()
      sublayer_config["name"] = new_name
      setattr(layer, attr_name, sublayer.from_config(sublayer_config))

  for layer in input_bert_encoder.layers:
    # NOTE: the ordering of the renames is important for the ordering of the
    # variables in the computed gradients. This is why we use three `for-loop`
    # instead of one.
    if isinstance(layer, tfm.nlp.layers.TransformerEncoderBlock):
      # pylint: disable=protected-access
      for attr_name in ["inner_dropout_layer", "attention_dropout"]:
        _dedup(layer, "_" + attr_name, layer.name + "/" + attr_name)
      # Some layers are nested within the main attention layer (if it exists).
      if layer._attention_layer is not None:
        prefix = layer.name + "/" + layer._attention_layer.name
        _dedup(layer, "_attention_layer", prefix + "/attention_layer")
        _dedup(
            layer._attention_layer,
            "_dropout_layer",
            prefix + "/attention_inner_dropout_layer",
        )
      for attr_name in ["attention_layer_norm", "intermediate_dense"]:
        _dedup(layer, "_" + attr_name, layer.name + "/" + attr_name)
      # This is one of the few times that we cannot build from a config, due
      # to the presence of lambda functions.
      if layer._intermediate_activation_layer is not None:
        policy = tf.keras.mixed_precision.global_policy()
        if policy.name == "mixed_bfloat16":
          policy = tf.float32
        layer._intermediate_activation_layer = tf.keras.layers.Activation(
            layer._inner_activation,
            dtype=policy,
            name=layer.name + "/intermediate_activation_layer",
        )
      for attr_name in ["output_dense", "output_dropout", "output_layer_norm"]:
        _dedup(layer, "_" + attr_name, layer.name + "/" + attr_name)
      # pylint: enable=protected-access


def get_unwrapped_bert_encoder(
    input_bert_encoder: tfm.nlp.networks.BertEncoder,
) -> tfm.nlp.networks.BertEncoder:
  """Creates a new BERT encoder whose layers are core Keras layers."""
  dedup_bert_encoder(input_bert_encoder)
  core_test_outputs = (
      gradient_clipping_utils.generate_model_outputs_using_core_keras_layers(
          input_bert_encoder,
          custom_layer_set={tfm.nlp.layers.TransformerEncoderBlock},
      )
  )
  return tf.keras.Model(
      inputs=input_bert_encoder.inputs,
      outputs=core_test_outputs,
  )
