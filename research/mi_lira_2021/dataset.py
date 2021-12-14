# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Tuple, List

import numpy as np
import tensorflow as tf


def record_parse(serialized_example: str, image_shape: Tuple[int, int, int]):
    features = tf.io.parse_single_example(serialized_example,
                                          features={'image': tf.io.FixedLenFeature([], tf.string),
                                                    'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image']).set_shape(image_shape)
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    return dict(image=image, label=features['label'])


class DataSet:
    """Wrapper for tf.data.Dataset to permit extensions."""

    def __init__(self, data: tf.data.Dataset,
                 image_shape: Tuple[int, int, int],
                 augment_fn: Optional[Callable] = None,
                 parse_fn: Optional[Callable] = record_parse):
        self.data = data
        self.parse_fn = parse_fn
        self.augment_fn = augment_fn
        self.image_shape = image_shape

    @classmethod
    def from_arrays(cls, images: np.ndarray, labels: np.ndarray, augment_fn: Optional[Callable] = None):
        return cls(tf.data.Dataset.from_tensor_slices(dict(image=images, label=labels)), images.shape[1:],
                   augment_fn=augment_fn, parse_fn=None)

    @classmethod
    def from_files(cls, filenames: List[str],
                   image_shape: Tuple[int, int, int],
                   augment_fn: Optional[Callable],
                   parse_fn: Optional[Callable] = record_parse):
        filenames_in = filenames
        filenames = sorted(sum([tf.io.gfile.glob(x) for x in filenames], []))
        if not filenames:
            raise ValueError('Empty dataset, files not found:', filenames_in)
        return cls(tf.data.TFRecordDataset(filenames), image_shape, augment_fn=augment_fn, parse_fn=parse_fn)

    @classmethod
    def from_tfds(cls, dataset: tf.data.Dataset, image_shape: Tuple[int, int, int],
                  augment_fn: Optional[Callable] = None):
        return cls(dataset.map(lambda x: dict(image=tf.cast(x['image'], tf.float32) / 127.5 - 1, label=x['label'])),
                   image_shape, augment_fn=augment_fn, parse_fn=None)

    def __iter__(self):
        return iter(self.data)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]

        def call_and_update(*args, **kwargs):
            v = getattr(self.__dict__['data'], item)(*args, **kwargs)
            if isinstance(v, tf.data.Dataset):
                return self.__class__(v, self.image_shape, augment_fn=self.augment_fn, parse_fn=self.parse_fn)
            return v

        return call_and_update

    def augment(self, para_augment: int = 4):
        if self.augment_fn:
            return self.map(self.augment_fn, para_augment)
        return self

    def nchw(self):
        return self.map(lambda x: dict(image=tf.transpose(x['image'], [0, 3, 1, 2]), label=x['label']))

    def one_hot(self, nclass: int):
        return self.map(lambda x: dict(image=x['image'], label=tf.one_hot(x['label'], nclass)))

    def parse(self, para_parse: int = 2):
        if not self.parse_fn:
            return self
        if self.image_shape:
            return self.map(lambda x: self.parse_fn(x, self.image_shape), para_parse)
        return self.map(self.parse_fn, para_parse)
