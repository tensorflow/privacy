# Copyright 2021 Google LLC
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

# pylint: skip-file
# pyformat: disable

import os
import shutil
import json

import numpy as np
import tensorflow as tf  # For data augmentation.
import tensorflow_datasets as tfds
from absl import app, flags

from objax.util import EasyDict

# from mi_lira_2021
from dataset import DataSet
from train import augment, MemModule, network

FLAGS = flags.FLAGS


def get_data(seed):
    """
    This is the function to generate subsets of the data for training models.

    First, we get the training dataset either from the numpy cache
    or otherwise we load it from tensorflow datasets.

    Then, we compute the subset. This works in one of two ways.

    1. If we have a seed, then we just randomly choose examples based on
       a prng with that seed, keeping FLAGS.pkeep fraction of the data.

    2. Otherwise, if we have an experiment ID, then we do something fancier.
       If we run each experiment independently then even after a lot of trials
       there will still probably be some examples that were always included
       or always excluded. So instead, with experiment IDs, we guarantee that
       after FLAGS.num_experiments are done, each example is seen exactly half
       of the time in train, and half of the time not in train.

    Finally, we add some poisons. The same poisoned samples are added for
    each randomly generated training set.
    We first select FLAGS.num_poison_targets victim points that will be targeted
    by the poisoning attack. For each of these victim points, the attacker will
    insert FLAGS.poison_reps mislabeled replicas of the point into the training
    set.

    For CIFAR-10, we recommend that:

        `FLAGS.num_poison_targets * FLAGS.poison_reps < 5000`

    Otherwise, the poisons might introduce too much label noise and the model's
    accuracy (and the attack's success rate) will be degraded.
    """
    DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')

    if os.path.exists(os.path.join(FLAGS.logdir, "x_train.npy")):
        inputs = np.load(os.path.join(FLAGS.logdir, "x_train.npy"))
        labels = np.load(os.path.join(FLAGS.logdir, "y_train.npy"))
    else:
        print("First time, creating dataset")
        data = tfds.as_numpy(tfds.load(name=FLAGS.dataset, batch_size=-1, data_dir=DATA_DIR))
        inputs = data['train']['image']
        labels = data['train']['label']

        inputs = (inputs/127.5)-1
        np.save(os.path.join(FLAGS.logdir, "x_train.npy"), inputs)
        np.save(os.path.join(FLAGS.logdir, "y_train.npy"), labels)

    nclass = np.max(labels)+1

    np.random.seed(seed)
    if FLAGS.num_experiments is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(FLAGS.num_experiments, len(inputs)))
        order = keep.argsort(0)
        keep = order < int(FLAGS.pkeep * FLAGS.num_experiments)
        keep = np.array(keep[FLAGS.expid], dtype=bool)
    else:
        keep = np.random.uniform(0, 1, size=len(inputs)) <= FLAGS.pkeep

    xs = inputs[keep]
    ys = labels[keep]

    if FLAGS.num_poison_targets > 0:

        # select some points as targets
        np.random.seed(FLAGS.poison_pos_seed)
        poison_pos = np.random.choice(len(inputs), size=FLAGS.num_poison_targets, replace=False)

        # create mislabeled poisons for the targeted points and replicate each
        # poison `FLAGS.poison_reps` times
        y_noise = np.mod(labels[poison_pos] + np.random.randint(low=1, high=nclass, size=FLAGS.num_poison_targets), nclass)
        ypoison = np.repeat(y_noise, FLAGS.poison_reps)
        xpoison = np.repeat(inputs[poison_pos], FLAGS.poison_reps, axis=0)
        xs = np.concatenate((xs, xpoison), axis=0)
        ys = np.concatenate((ys, ypoison), axis=0)

        if not os.path.exists(os.path.join(FLAGS.logdir, "poison_pos.npy")):
            np.save(os.path.join(FLAGS.logdir, "poison_pos.npy"), poison_pos)

    if FLAGS.augment == 'weak':
        aug = lambda x: augment(x, 4)
    elif FLAGS.augment == 'mirror':
        aug = lambda x: augment(x, 0)
    elif FLAGS.augment == 'none':
        aug = lambda x: augment(x, 0, mirror=False)
    else:
        raise

    print(xs.shape, ys.shape)
    train = DataSet.from_arrays(xs, ys,
                                augment_fn=aug)
    test = DataSet.from_tfds(tfds.load(name=FLAGS.dataset, split='test', data_dir=DATA_DIR), xs.shape[1:])
    train = train.cache().shuffle(len(xs)).repeat().parse().augment().batch(FLAGS.batch)
    train = train.nchw().one_hot(nclass).prefetch(FLAGS.batch)
    test = test.cache().parse().batch(FLAGS.batch).nchw().prefetch(FLAGS.batch)

    return train, test, xs, ys, keep, nclass


def main(argv):
    del argv
    tf.config.experimental.set_visible_devices([], "GPU")

    seed = FLAGS.seed
    if seed is None:
        import time
        seed = np.random.randint(0, 1000000000)
        seed ^= int(time.time())

    args = EasyDict(arch=FLAGS.arch,
                    lr=FLAGS.lr,
                    batch=FLAGS.batch,
                    weight_decay=FLAGS.weight_decay,
                    augment=FLAGS.augment,
                    seed=seed)

    if FLAGS.expid is not None:
        logdir = "experiment-%d_%d" % (FLAGS.expid, FLAGS.num_experiments)
    else:
        logdir = "experiment-"+str(seed)
    logdir = os.path.join(FLAGS.logdir, logdir)

    if os.path.exists(os.path.join(logdir, "ckpt", "%010d.npz" % FLAGS.epochs)):
        print(f"run {FLAGS.expid} already completed.")
        return
    else:
        if os.path.exists(logdir):
            print(f"deleting run {FLAGS.expid} that did not complete.")
            shutil.rmtree(logdir)

    print(f"starting run {FLAGS.expid}.")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    train, test, xs, ys, keep, nclass = get_data(seed)

    # Define the network and train_it
    tm = MemModule(network(FLAGS.arch), nclass=nclass,
                   mnist=FLAGS.dataset == 'mnist',
                   epochs=FLAGS.epochs,
                   expid=FLAGS.expid,
                   num_experiments=FLAGS.num_experiments,
                   pkeep=FLAGS.pkeep,
                   save_steps=FLAGS.save_steps,
                   **args
    )

    r = {}
    r.update(tm.params)

    open(os.path.join(logdir, 'hparams.json'), "w").write(json.dumps(tm.params))
    np.save(os.path.join(logdir,'keep.npy'), keep)

    tm.train(FLAGS.epochs, len(xs), train, test, logdir,
             save_steps=FLAGS.save_steps)


if __name__ == '__main__':
    flags.DEFINE_string('arch', 'cnn32-3-mean', 'Model architecture.')
    flags.DEFINE_float('lr', 0.1, 'Learning rate.')
    flags.DEFINE_string('dataset', 'cifar10', 'Dataset.')
    flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay ratio.')
    flags.DEFINE_integer('batch', 256, 'Batch size')
    flags.DEFINE_integer('epochs', 100, 'Training duration in number of epochs.')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_integer('seed', None, 'Training seed.')
    flags.DEFINE_float('pkeep', .5, 'Probability to keep examples.')
    flags.DEFINE_integer('expid', None, 'Experiment ID')
    flags.DEFINE_integer('num_experiments', None, 'Number of experiments')
    flags.DEFINE_string('augment', 'weak', 'Strong or weak augmentation')
    flags.DEFINE_integer('eval_steps', 1, 'how often to get eval accuracy.')
    flags.DEFINE_integer('save_steps', 10, 'how often to get save model.')

    flags.DEFINE_integer('num_poison_targets', 250, 'Number of points to target '
                                                    'with the poisoning attack.')
    flags.DEFINE_integer('poison_reps', 8, 'Number of times to repeat each poison.')
    flags.DEFINE_integer('poison_pos_seed', 0, '')
    app.run(main)
