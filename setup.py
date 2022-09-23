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
        'absl-py>=1.0,==1.*',
        'attrs~=21.4',
        'dm-tree==0.1.7',
        'dp-accounting==0.3.0',
        'matplotlib~=3.3',
        'numpy~=1.21',
        'pandas~=1.4',
        'scikit-learn>=1.0,==1.*',
        'scipy~=1.7',
        'tensorflow-datasets~=4.5',
        'tensorflow-estimator~=2.4',
        'tensorflow-probability==0.15.0',
        'tensorflow~=2.4',
    ],
    packages=find_packages())
