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

import functools
import os
import shutil
from typing import Callable
import json

import jax
import jax.numpy as jn
import numpy as np
import tensorflow as tf  # For data augmentation.
import tensorflow_datasets as tfds
from absl import app, flags

import objax
from objax.jaxboard import SummaryWriter, Summary
from objax.util import EasyDict
from objax.zoo import convnet, wide_resnet

from dataset import DataSet

FLAGS = flags.FLAGS

def augment(x, shift: int, mirror=True):
    """
    Augmentation function used in training the model.
    """
    y = x['image']
    if mirror:
        y = tf.image.random_flip_left_right(y)
    y = tf.pad(y, [[shift] * 2, [shift] * 2, [0] * 2], mode='REFLECT')
    y = tf.image.random_crop(y, tf.shape(x['image']))
    return dict(image=y, label=x['label'])


class TrainLoop(objax.Module):
    """
    Training loop for general machine learning models.
    Based on the training loop from the objax CIFAR10 example code.
    """
    predict: Callable
    train_op: Callable

    def __init__(self, nclass: int, **kwargs):
        self.nclass = nclass
        self.params = EasyDict(kwargs)

    def train_step(self, summary: Summary, data: dict, progress: np.ndarray):
        kv = self.train_op(progress, data['image'].numpy(), data['label'].numpy())
        for k, v in kv.items():
            if jn.isnan(v):
                raise ValueError('NaN, try reducing learning rate', k)
            if summary is not None:
                summary.scalar(k, float(v))

    def train(self, num_train_epochs: int, train_size: int, train: DataSet, test: DataSet, logdir: str, save_steps=100, patience=None):
        """
        Completely standard training. Nothing interesting to see here.
        """
        checkpoint = objax.io.Checkpoint(logdir, keep_ckpts=20, makedir=True)
        start_epoch, last_ckpt = checkpoint.restore(self.vars())
        train_iter = iter(train)
        progress = np.zeros(jax.local_device_count(), 'f')  # for multi-GPU

        best_acc = 0
        best_acc_epoch = -1

        with SummaryWriter(os.path.join(logdir, 'tb')) as tensorboard:
            for epoch in range(start_epoch, num_train_epochs):
                # Train
                summary = Summary()
                loop = range(0, train_size, self.params.batch)
                for step in loop:
                    progress[:] = (step + (epoch * train_size)) / (num_train_epochs * train_size)
                    self.train_step(summary, next(train_iter), progress)

                # Eval
                accuracy, total = 0, 0
                if epoch%FLAGS.eval_steps == 0 and test is not None:
                    for data in test:
                        total += data['image'].shape[0]
                        preds = np.argmax(self.predict(data['image'].numpy()), axis=1)
                        accuracy += (preds == data['label'].numpy()).sum()
                    accuracy /= total
                    summary.scalar('eval/accuracy', 100 * accuracy)
                    tensorboard.write(summary, step=(epoch + 1) * train_size)
                    print('Epoch %04d  Loss %.2f  Accuracy %.2f' % (epoch + 1, summary['losses/xe'](),
                                                                    summary['eval/accuracy']()))

                    if summary['eval/accuracy']() > best_acc:
                        best_acc = summary['eval/accuracy']()
                        best_acc_epoch = epoch
                    elif patience is not None and epoch > best_acc_epoch + patience:
                        print("early stopping!")
                        checkpoint.save(self.vars(), epoch + 1)
                        return

                else:
                    print('Epoch %04d  Loss %.2f  Accuracy --' % (epoch + 1, summary['losses/xe']()))

                if epoch%save_steps == save_steps-1:
                    checkpoint.save(self.vars(), epoch + 1)


# We inherit from the training loop and define predict and train_op.
class MemModule(TrainLoop):
    def __init__(self, model: Callable, nclass: int, mnist=False, **kwargs):
        """
        Completely standard training. Nothing interesting to see here.
        """
        super().__init__(nclass, **kwargs)
        self.model = model(1 if mnist else 3, nclass)
        self.opt = objax.optimizer.Momentum(self.model.vars())
        self.model_ema = objax.optimizer.ExponentialMovingAverageModule(self.model, momentum=0.999, debias=True)

        @objax.Function.with_vars(self.model.vars())
        def loss(x, label):
            logit = self.model(x, training=True)
            loss_wd = 0.5 * sum((v.value ** 2).sum() for k, v in self.model.vars().items() if k.endswith('.w'))
            loss_xe = objax.functional.loss.cross_entropy_logits(logit, label).mean()
            return loss_xe + loss_wd * self.params.weight_decay, {'losses/xe': loss_xe, 'losses/wd': loss_wd}

        gv = objax.GradValues(loss, self.model.vars())
        self.gv = gv

        @objax.Function.with_vars(self.vars())
        def train_op(progress, x, y):
            g, v = gv(x, y)
            lr = self.params.lr * jn.cos(progress * (7 * jn.pi) / (2 * 8))
            lr = lr * jn.clip(progress*100,0,1)
            self.opt(lr, g)
            self.model_ema.update_ema()
            return {'monitors/lr': lr, **v[1]}

        self.predict = objax.Jit(objax.nn.Sequential([objax.ForceArgs(self.model_ema, training=False)]))

        self.train_op = objax.Jit(train_op)


def network(arch: str):
    if arch == 'cnn32-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=32, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn32-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=32, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'cnn64-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn64-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'wrn28-1':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=1)
    elif arch == 'wrn28-2':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=2)
    elif arch == 'wrn28-10':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=10)
    raise ValueError('Architecture not recognized', arch)

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
        np.save(os.path.join(FLAGS.logdir, "x_train.npy"),inputs)
        np.save(os.path.join(FLAGS.logdir, "y_train.npy"),labels)

    nclass = np.max(labels)+1

    np.random.seed(seed)
    if FLAGS.num_experiments is not None:
        np.random.seed(0)
        keep = np.random.uniform(0,1,size=(FLAGS.num_experiments, FLAGS.dataset_size))
        order = keep.argsort(0)
        keep = order < int(FLAGS.pkeep * FLAGS.num_experiments)
        keep = np.array(keep[FLAGS.expid], dtype=bool)
    else:
        keep = np.random.uniform(0, 1, size=FLAGS.dataset_size) <= FLAGS.pkeep

    if FLAGS.only_subset is not None:
        keep[FLAGS.only_subset:] = 0

    xs = inputs[keep]
    ys = labels[keep]

    if FLAGS.augment == 'weak':
        aug = lambda x: augment(x, 4)
    elif FLAGS.augment == 'mirror':
        aug = lambda x: augment(x, 0)
    elif FLAGS.augment == 'none':
        aug = lambda x: augment(x, 0, mirror=False)
    else:
        raise

    train = DataSet.from_arrays(xs, ys,
                                augment_fn=aug)
    test = DataSet.from_tfds(tfds.load(name=FLAGS.dataset, split='test', data_dir=DATA_DIR), xs.shape[1:])
    train = train.cache().shuffle(8192).repeat().parse().augment().batch(FLAGS.batch)
    train = train.nchw().one_hot(nclass).prefetch(16)
    test = test.cache().parse().batch(FLAGS.batch).nchw().prefetch(16)

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


    if FLAGS.tunename:
        logdir = '_'.join(sorted('%s=%s' % k for k in args.items()))
    elif FLAGS.expid is not None:
        logdir = "experiment-%d_%d"%(FLAGS.expid,FLAGS.num_experiments)
    else:
        logdir = "experiment-"+str(seed)
    logdir = os.path.join(FLAGS.logdir, logdir)

    if os.path.exists(os.path.join(logdir, "ckpt", "%010d.npz"%FLAGS.epochs)):
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
                   only_subset=FLAGS.only_subset,
                   **args
    )

    r = {}
    r.update(tm.params)

    open(os.path.join(logdir,'hparams.json'),"w").write(json.dumps(tm.params))
    np.save(os.path.join(logdir,'keep.npy'), keep)

    tm.train(FLAGS.epochs, len(xs), train, test, logdir,
             save_steps=FLAGS.save_steps, patience=FLAGS.patience)



if __name__ == '__main__':
    flags.DEFINE_string('arch', 'cnn32-3-mean', 'Model architecture.')
    flags.DEFINE_float('lr', 0.1, 'Learning rate.')
    flags.DEFINE_string('dataset', 'cifar10', 'Dataset.')
    flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay ratio.')
    flags.DEFINE_integer('batch', 256, 'Batch size')
    flags.DEFINE_integer('epochs', 501, 'Training duration in number of epochs.')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_integer('seed', None, 'Training seed.')
    flags.DEFINE_float('pkeep', .5, 'Probability to keep examples.')
    flags.DEFINE_integer('expid', None, 'Experiment ID')
    flags.DEFINE_integer('num_experiments', None, 'Number of experiments')
    flags.DEFINE_string('augment', 'weak', 'Strong or weak augmentation')
    flags.DEFINE_integer('only_subset', None, 'Only train on a subset of images.')
    flags.DEFINE_integer('dataset_size', 50000, 'number of examples to keep.')
    flags.DEFINE_integer('eval_steps', 1, 'how often to get eval accuracy.')
    flags.DEFINE_integer('abort_after_epoch', None, 'stop trainin early at an epoch')
    flags.DEFINE_integer('save_steps', 10, 'how often to get save model.')
    flags.DEFINE_integer('patience', None, 'Early stopping after this many epochs without progress')
    flags.DEFINE_bool('tunename', False, 'Use tune name?')
    app.run(main)

