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
"""TensorFlow Privacy/Privacy Tests library setup file for pip."""

import setuptools

with open('tensorflow_privacy/privacy/privacy_tests/version.py') as file:
  globals_dict = {}
  exec(file.read(), globals_dict)  # pylint: disable=exec-used
  VERSION = globals_dict['__version__']

README = (
    'A Python library that includes implementations of tests for empirical '
    'privacy.'
)

setuptools.setup(
    name='tensorflow_empirical_privacy',
    version=VERSION,
    description='Tests for empirical privacy.',
    long_description=README,
    long_description_content_type='text/plain',
    url='https://github.com/tensorflow/privacy',
    license='Apache-2.0',
    packages=setuptools.find_packages(include=['*privacy.privacy_tests*']),
    install_requires=[
        'absl-py>=1.0,==1.*',
        'immutabledict~=2.2',
        'matplotlib~=3.3',
        'numpy~=1.21',
        'pandas~=1.4',
        'scikit-learn>=1.0,==1.*',
        'scipy~=1.9',
        'statsmodels==0.14.0',
        'tensorflow~=2.4',
        'tensorflow-privacy>=0.8.12',
        'tf-models-official~=2.13',
    ],
    python_requires='>=3.9.0,<3.12',
)
