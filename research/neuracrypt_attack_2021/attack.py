# Copyright 2021 Google LLC
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
# limitations under the License.
# ==============================================================================

# This program solves the NeuraCrypt challenge to 100% accuracy.
# Given a set of encoded images and original versions of those,
# it shows how to match the original to the encoded.

import collections
import hashlib
import time
import multiprocessing as mp


import torch
import numpy as np
import torch.nn as nn
import scipy.stats
import matplotlib.pyplot as plt
from PIL import Image

import jax
import jax.numpy as jn
import objax
import scipy.optimize
import numpy as np
import multiprocessing as mp

# Objax neural network that's going to embed patches to a
# low dimensional space to guess if two patches correspond
# to the same orginal image.
class Model(objax.Module):
    def __init__(self):
        IN = 15
        H = 64
        self.encoder =objax.nn.Sequential([
            objax.nn.Linear(IN, H),
            objax.functional.leaky_relu,
            objax.nn.Linear(H, H),
            objax.functional.leaky_relu,
            objax.nn.Linear(H, 8)])
        self.decoder =objax.nn.Sequential([
            objax.nn.Linear(IN, H),
            objax.functional.leaky_relu,
            objax.nn.Linear(H, H),
            objax.functional.leaky_relu,
            objax.nn.Linear(H, 8)])
        self.scale = objax.nn.Linear(1, 1, use_bias=False)
    def encode(self, x):
        # Encode turns original images into feature space
        a = self.encoder(x)
        a = a/jn.sum(a**2,axis=-1,keepdims=True)**.5
        return a
    def decode(self, x):
        # And decode turns encoded images into feature space
        a = self.decoder(x)
        a = a/jn.sum(a**2,axis=-1,keepdims=True)**.5
        return a

# Proxy dataset for analysis
class ImageNet:
    num_chan = 3
    private_kernel_size = 16
    hidden_dim = 2048
    img_size = (256, 256)
    private_depth = 7
    def __init__(self, remove):
        self.remove_pixel_shuffle = remove

# Original dataset as used in the NeuraCrypt paper
class Xray:
    num_chan = 1
    private_kernel_size = 16
    hidden_dim = 2048
    img_size = (256, 256)
    private_depth = 4
    def __init__(self, remove):
        self.remove_pixel_shuffle = remove

## The following class is taken directly from the NeuraCrypt codebase.
## https://github.com/yala/NeuraCrypt
## which is originally licensed under the MIT License
class PrivateEncoder(nn.Module):
    def __init__(self, args, width_factor=1):
        super(PrivateEncoder, self).__init__()
        self.args = args
        input_dim = args.num_chan
        patch_size = args.private_kernel_size
        output_dim = args.hidden_dim
        num_patches =  (args.img_size[0] // patch_size) **2
        self.noise_size = 1

        args.input_dim = args.hidden_dim


        layers  = [
                    nn.Conv2d(input_dim, output_dim * width_factor, kernel_size=patch_size, dilation=1 ,stride=patch_size),
                    nn.ReLU()
                    ]
        for _ in range(self.args.private_depth):
            layers.extend( [
                nn.Conv2d(output_dim * width_factor, output_dim * width_factor , kernel_size=1, dilation=1, stride=1),
                nn.BatchNorm2d(output_dim * width_factor, track_running_stats=False),
                nn.ReLU()
            ])


        self.image_encoder = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, output_dim * width_factor))

        self.mixer = nn.Sequential( *[
            nn.ReLU(),
            nn.Linear(output_dim * width_factor, output_dim)
            ])


    def forward(self, x):
        encoded = self.image_encoder(x)
        B, C, H,W = encoded.size()
        encoded = encoded.view([B, -1, H*W]).transpose(1,2)
        encoded += self.pos_embedding
        encoded  = self.mixer(encoded)

        ## Shuffle indicies
        if not self.args.remove_pixel_shuffle:
            shuffled = torch.zeros_like(encoded)
            for i in range(B):
                idx = torch.randperm(H*W, device=encoded.device)
                for j, k in enumerate(idx):
                    shuffled[i,j] = encoded[i,k]
            encoded = shuffled

        return encoded
## End copied code

def setup(ds):
    """
    Load the datasets to use. Nothing interesting to see.
    """
    global x_train, y_train
    if ds == 'imagenet':
        import torchvision
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor()])
        imagenet_data = torchvision.datasets.ImageNet('/mnt/data/datasets/unpacked_imagenet_pytorch/',
                                                      split='val',
                                                      transform=transform)
        data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size=100,
                                                  shuffle=True,
                                                  num_workers=8)
        r = []
        for x,_ in data_loader:
            if len(r) > 1000: break
            print(x.shape)
            r.extend(x.numpy())
        x_train = np.array(r)
        print(x_train.shape)
    elif ds == 'xray':
        import torchvision
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor()])
        imagenet_data = torchvision.datasets.ImageFolder('CheXpert-v1.0/train',
                                                      transform=transform)
        data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size=100,
                                                  shuffle=True,
                                                  num_workers=8)
        r = []
        for x,_ in data_loader:
            if len(r) > 1000: break
            print(x.shape)
            r.extend(x.numpy())
        x_train = np.array(r)
        print(x_train.shape)
    elif ds == 'challenge':
        x_train = np.load("orig-7.npy")
        print(np.min(x_train), np.max(x_train), x_train.shape)
    else:
        raise


def gen_train_data():
    """
    Generate aligned training data to train a patch similarity function.
    Given some original images, generate lots of encoded versions.
    """
    global encoded_train, original_train

    encoded_train = []
    original_train = []

    args = Xray(True)

    C = 100
    for i in range(30):
        print(i)
        torch.manual_seed(int(time.time()))
        e = PrivateEncoder(args).cuda()
        batch = np.random.randint(0, len(x_train), size=C)
        xin = x_train[batch]

        r = []
        for i in range(0,C,32):
            r.extend(e(torch.tensor(xin[i:i+32]).cuda()).detach().cpu().numpy())
        r = np.array(r)

        encoded_train.append(r)
        original_train.append(xin)

def features_(x, moments=15, encoded=False):
    """
    Compute higher-order moments for patches in an image to use as
    features for the neural network.
    """
    x = np.array(x, dtype=np.float32)
    dim = 2
    arr = np.array([np.mean(x, dim)] + [abs(scipy.stats.moment(x, moment=i, axis=dim))**(1/i) for i in range(1,moments)])
    
    return arr.transpose((1,2,0))


def features(x, encoded):
    """
    Given the original images or the encoded images, generate the
    features to use for the patch similarity function.
    """
    print('start shape',x.shape)
    if len(x.shape) == 3:
        x = x - np.mean(x,axis=0,keepdims=True)
    else:
        # count x 100 x 256 x 768
        print(x[0].shape)
        x = x - np.mean(x,axis=1,keepdims=True)
        # remove per-neural-network dimension
        x = x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
    p = mp.Pool(96)
    B = len(x) // 96
    print(1)
    bs = [x[i:i+B] for i in range(0,len(x),B)]
    print(2)
    r = p.map(features_, bs)
    #r = features_(bs[0][:100])
    print(3)
    p.close()
    #r = np.array(r)
    #print('finish',r.shape)
    return np.concatenate(r, axis=0)
    


def get_train_features():
    """
    Create features for the entire datasets.
    """
    global xs_train, ys_train
    print(x_train.shape)
    original_train_ = np.array(original_train)
    encoded_train_ = np.array(encoded_train)

    print("Computing features")
    ys_train = features(encoded_train_, True)
    
    patch_size = 16
    ss = original_train_.shape[3] // patch_size
    # Okay so this is an ugly transpose block.
    # We are going from [outer_batch, batch_size, channels, width, height
    # to [outer_batch, batch_size, channels, width/patch_size, patch_size, height/patch_size, patch_size]
    # Then we reshape this and flatten so that we end up with
    # [other_batch, batch_size, width/patch_size, height_patch_size, patch_size**2*channels]
    # So that now we can run features on the last dimension
    original_train_ = original_train_.reshape((original_train_.shape[0],
                                               original_train_.shape[1],
                                               original_train_.shape[2],
                                               ss,patch_size,ss,patch_size)).transpose((0,1,3,5,2,4,6)).reshape((original_train_.shape[0], original_train_.shape[1], ss**2, patch_size**2))


    xs_train = features(original_train_, False)

    print(xs_train.shape, ys_train.shape)
    

def train_model():
    """
    Train the patch similarity function
    """
    global ema, model
    
    model = Model()
    def loss(x, y):
        """
        K-way contrastive loss as in SimCLR et al.
        The idea is that we should embed x and y so that they are similar
        to each other, and dis-similar from others. To do this we have a
        softmx loss over one dimension to make the values large on the diagonal
        and small off-diagonal.
        """
        a = model.encode(x)
        b = model.decode(y)

        mat = a@b.T
        return objax.functional.loss.cross_entropy_logits_sparse(
            logits=jn.exp(jn.clip(model.scale.w.value, -2, 4)) * mat,
            labels=np.arange(a.shape[0])).mean()

    ema = objax.optimizer.ExponentialMovingAverage(model.vars(), momentum=0.999)
    gv = objax.GradValues(loss, model.vars())
    
    encode_ema = ema.replace_vars(lambda x: model.encode(x))
    decode_ema = ema.replace_vars(lambda y: model.decode(y))

    def train_op(x, y):
        """
        No one was ever fired for using Adam with 1e-4.
        """
        g, v = gv(x, y)
        opt(1e-4, g)
        ema()
        return v

    opt = objax.optimizer.Adam(model.vars())
    train_op = objax.Jit(train_op, gv.vars() + opt.vars() + ema.vars())

    ys_ = ys_train

    print(ys_.shape)
    
    xs_ = xs_train.reshape((-1, xs_train.shape[-1]))
    ys_ = ys_.reshape((-1, ys_train.shape[-1]))

    # The model scale trick here is taken from CLIP.
    # Let the model decide how confident to make its own predictions.
    model.scale.w.assign(jn.zeros((1,1)))

    valid_size = 1000
    
    print(xs_train.shape)
    # SimCLR likes big batches
    B = 4096
    for it in range(80):
        print()
        ms = []
        for i in range(1000):
            # First batch is smaller, to make training more stable
            bs = [B // 64, B][it>0]
            batch = np.random.randint(0, len(xs_)-valid_size, size=bs)
            r = train_op(xs_[batch], ys_[batch])

            # This shouldn't happen, but if it does, better to bort early
            if np.isnan(r):
                print("Die on nan")
                print(ms[-100:])
                return
            ms.append(r)

        print('mean',np.mean(ms), 'scale', model.scale.w.value)
        print('loss',loss(xs_[-100:], ys_[-100:]))
    
        a = encode_ema(xs_[-valid_size:])
        b = decode_ema(ys_[-valid_size:])
        
        br = b[np.random.permutation(len(b))]
        
        print('score',np.mean(np.sum(a*b,axis=(1)) - np.sum(a*br,axis=(1))),
              np.mean(np.sum(a*b,axis=(1)) > np.sum(a*br,axis=(1))))
    ckpt = objax.io.Checkpoint("saved", keep_ckpts=0)
    ema.replace_vars(lambda: ckpt.save(model.vars(), 0))()
        


def load_challenge():
    """
    Load the challenge datast for attacking
    """
    global xs, ys, encoded, original, ooriginal
    print("SETUP: Loading matrixes")
    # The encoded images
    encoded = np.load("challenge-7.npy")
    # And the original images
    ooriginal = original = np.load("orig-7.npy")

    print("Sizes", encoded.shape, ooriginal.shape)

    # Again do that ugly resize thing to make the features be on the last dimension
    # Look up above to see what's going on.
    patch_size = 16
    ss = original.shape[2] // patch_size
    original = ooriginal.reshape((original.shape[0],1,ss,patch_size,ss,patch_size))
    original = original.transpose((0,2,4,1,3,5))
    original = original.reshape((original.shape[0], ss**2, patch_size**2))


def match_sub(args):
    """
    Find the best way to undo the permutation between two images.
    """
    vec1, vec2 = args
    value = np.sum((vec1[None,:,:] - vec2[:,None,:])**2,axis=2)
    row, col = scipy.optimize.linear_sum_assignment(value)
    return col
    

def recover_local_permutation():
    """
    Given a set of encoded images, return a new encoding without permutations
    """
    global encoded, ys

    p = mp.Pool(96)
    print('recover local')
    local_perm = p.map(match_sub, [(encoded[0], e) for e in encoded])
    local_perm = np.array(local_perm)

    encoded_perm = []

    for i in range(len(encoded)):
        encoded_perm.append(encoded[i][np.argsort(local_perm[i])])

    encoded_perm = np.array(encoded_perm)

    encoded = np.array(encoded_perm)

    p.close()


def recover_better_local_permutation():
    """
    Given a set of encoded images, return a new encoding, but better!
    """
    global encoded, ys

    # Now instead of pairing all images to image 0, we compute the mean l2 vector
    # and then pair all images onto the mean vector. Slightly more noise resistant.
    p = mp.Pool(96)    
    target = encoded.mean(0)
    local_perm = p.map(match_sub, [(target, e) for e in encoded])
    local_perm = np.array(local_perm)

    # Probably we didn't change by much, generally <0.1%
    print('improved changed by', np.mean(local_perm != np.arange(local_perm.shape[1])))
    
    encoded_perm = []

    for i in range(len(encoded)):
        encoded_perm.append(encoded[i][np.argsort(local_perm[i])])

    encoded = np.array(encoded_perm)

    p.close()


def compute_patch_similarity():
    """
    Compute the feature vectors for each patch using the trained neural network.
    """
    global xs, ys, xs_image, ys_image

    print("Computing features")
    ys = features(encoded, encoded=True)
    xs = features(original, encoded=False)

    model = Model()
    ckpt = objax.io.Checkpoint("saved", keep_ckpts=0)
    ckpt.restore(model.vars())
    
    xs_image = model.encode(xs)
    ys_image = model.decode(ys)
    assert xs.shape[0] == xs_image.shape[0]
    print("Done")
    

def match(args, ret_col=False):
    """
    Compute the similarity between image features and encoded features.
    """
    vec1, vec2s = args
    r = []
    open("/tmp/start%d.%d"%(np.random.randint(10000),time.time()),"w").write("hi")
    for vec2 in vec2s:
        value = np.sum(vec1[None,:,:] * vec2[:,None,:],axis=2)

        row, col = scipy.optimize.linear_sum_assignment(-value)
        r.append(value[row,col].mean())
    return r



def recover_global_matching_first():
    """
    Recover the global matching of original to encoded images by doing
    an all-pairs matching problem
    """
    global global_matching, ys_image, encoded

    matrix = []
    p = mp.Pool(96)
    xs_image_ = np.array(xs_image)
    ys_image_ = np.array(ys_image)

    matrix = p.map(match, [(x, ys_image_) for x in xs_image_])
    matrix = np.array(matrix).reshape((xs_image.shape[0],
                                       xs_image.shape[0]))


    row, col = scipy.optimize.linear_sum_assignment(-np.array(matrix))
    global_matching = np.argsort(col)
    print('glob',list(global_matching))
    
    p.close()



def recover_global_permutation():
    """
    Find the way that the encoded images are permuted off of the original images
    """
    global global_permutation

    print("Glob match", global_matching)
    overall = []
    for i,j in enumerate(global_matching):
        overall.append(np.sum(xs_image[j][None,:,:] * ys_image[i][:,None,:],axis=2))
        
    overall = np.mean(overall, 0)
    
    row, col = scipy.optimize.linear_sum_assignment(-overall)

    try:
        print("Changed frac:", np.mean(global_permutation!=np.argsort(col)))
    except:
        pass
    
    global_permutation = np.argsort(col)


def recover_global_matching_second():
    """
    Match each encoded image with its original encoded image,
    but better by relying on the global permutation.
    """
    global global_matching_second, global_matching

    ys_fix = []
    for i in range(ys_image.shape[0]):
        ys_fix.append(ys_image[i][global_permutation])
    ys_fix = np.array(ys_fix)


    print(xs_image.shape)

    sims = []
    for i in range(0,len(xs_image),10):
        tmp = np.mean(xs_image[None,:,:,:] * ys_fix[i:i+10][:,None,:,:],axis=(2,3))
        sims.extend(tmp)
    sims = np.array(sims)
    print(sims.shape)
    
    
    row, col = scipy.optimize.linear_sum_assignment(-sims)

    print('arg',sims.argmax(1))

    print("Same matching frac", np.mean(col == global_matching) )
    print(col)
    global_matching = col


def extract_by_training(resume):
    """
    Final recovery process by extracting the neural network
    """
    global inverse

    device = torch.device('cuda:1')
    
    if not resume:
        inverse = PrivateEncoder(Xray(True)).cuda(device)

    # More adam to train.
    optimizer = torch.optim.Adam(inverse.parameters(), lr=0.0001)

    this_xs = ooriginal[global_matching]
    this_ys = encoded[:,global_permutation,:]

    for i in range(2000):
        idx = np.random.random_integers(0, len(this_xs)-1, 32)
        xbatch = torch.tensor(this_xs[idx]).cuda(device)
        ybatch = torch.tensor(this_ys[idx]).cuda(device)
        
        optimizer.zero_grad()
        
        guess_output = inverse(xbatch)
        # L1 loss because we don't want to be sensitive to outliers
        error = torch.mean(torch.abs(guess_output-ybatch))
        error.backward()

        optimizer.step()

        print(error)



def test_extract():
    """
    Now we can recover the matching much better by computing the estimated
    encodings for each original image.
    """
    global err, global_matching, guessed_encoded, smatrix

    device = torch.device('cuda:1')

    print(ooriginal.shape, encoded.shape)

    out = []
    for i in range(0,len(ooriginal),32):
        print(i)
        out.extend(inverse(torch.tensor(ooriginal[i:i+32]).cuda(device)).cpu().detach().numpy())

    guessed_encoded = np.array(out)


    # Now we have to compare each encoded image with every other original image.
    # Do this fast with some matrix multiplies.
    
    out = guessed_encoded.reshape((len(encoded), -1))
    real = encoded[:,global_permutation,:].reshape((len(encoded), -1))
    @jax.jit
    def foo(x, y):
        return jn.square(x[:,None] - y[None,:]).sum(2)

    smatrix = np.zeros((len(out), len(out)))

    B = 500
    for i in range(0,len(out),B):
        print(i)
        for j in range(0,len(out),B):
            smatrix[i:i+B, j:j+B] = foo(out[i:i+B], real[j:j+B])

    # And the final time you'l have to look at a min weight matching, I promise.
    row, col = scipy.optimize.linear_sum_assignment(np.array(smatrix))
    r = np.array(smatrix)

    print(list(row)[::100])

    print("Differences", np.mean(np.argsort(col) != global_matching))

    global_matching = np.argsort(col)


def perf(steps=[]):
    if len(steps) == 0:
        steps.append(time.time())
    else:
        print("Last Time Elapsed:", time.time()-steps[-1], ' Total Time Elapsed:', time.time()-steps[0])
        steps.append(time.time())
    time.sleep(1)


if __name__ == "__main__":
    if True:
        perf()
        setup('challenge')
        perf()
        gen_train_data()
        perf()
        get_train_features()
        perf()
        train_model()
        perf()

    if True:
        load_challenge()
        perf()
        recover_local_permutation()
        perf()
        recover_better_local_permutation()
        perf()
        compute_patch_similarity()
        perf()
        recover_global_matching_first()
        perf()

    for _ in range(3):
        recover_global_permutation()
        perf()
        recover_global_matching_second()
        perf()

    for i in range(3):
        recover_global_permutation()
        perf()
        extract_by_training(i > 0)
        perf()
        test_extract()
        perf()
    print(perf())
