# Copyright 2018, The TensorFlow Authors.
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
"""TensorFlow Privacy library setup file for pip."""

from setuptools import find_packages
from setuptools import setup

with open('tensorflow_privacy/version.py') as file:
  globals_dict = {}
  exec(file.read(), globals_dict)  # pylint: disable=exec-used
  VERSION = globals_dict['__version__']

setup(
    name='tensorflow_privacy',
    version=VERSION,
    url='https://github.com/tensorflow/privacy',
    license='Apache-2.0',
    install_requires=[
        'absl-py>=1.0.0',
        'attrs>=21.2.0',
        'dm-tree>=0.1.1',
        'matplotlib>=3.3.4',
        'numpy>=1.21.5',
        'pandas>=1.1.4',
        'scipy>=1.5',
        'scikit-learn>=0.23',
        'tensorflow-datasets>=4.5.2',
        'tensorflow-estimator>=2.4',
        'tensorflow-probability>=0.15.0',
    ],
    extras_require={
        'tf': ['tensorflow>=2.4'],
        'tf_gpu': ['tensorflow-gpu>=2.4'],
    },
    packages=find_packages()
)
