# Lint as: python3
# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import tensorflow_privacy
"""Script to generate api_docs for TensorFlow Privacy."""

import os

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_privacy as tf_privacy

flags.DEFINE_string('output_dir', '/tmp/tf_privacy',
                    'Where to output the docs.')
flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy',
    'The url prefix for links to code.')
flags.DEFINE_string('site_path', 'responsible_ai/privacy/api_docs/python/',
                    'The location of the doc setin the site.')
flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files.')

FLAGS = flags.FLAGS

PROJECT_SHORT_NAME = 'tf_privacy'
PROJECT_FULL_NAME = 'TensorFlow Privacy'


def gen_api_docs():
  """Generates api docs for the tensorflow docs package."""
  output_dir = FLAGS.output_dir

  doc_generator = generate_lib.DocGenerator(
      root_title=PROJECT_FULL_NAME,
      py_modules=[(PROJECT_SHORT_NAME, tf_privacy)],
      base_dir=os.path.dirname(tf_privacy.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      site_path=FLAGS.site_path,
      search_hints=FLAGS.search_hints,
      private_map={},
      # This callback cleans up a lot of aliases caused by internal imports.
      callbacks=[public_api.explicit_package_contents_filter])

  doc_generator.build(output_dir)
  print('Output docs to: ', output_dir)


def main(_):
  gen_api_docs()


if __name__ == '__main__':
  app.run(main)
