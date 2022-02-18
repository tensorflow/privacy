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

CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 0 --logdir exp/cifar10 &> logs/log_0 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 1 --logdir exp/cifar10 &> logs/log_1 &
CUDA_VISIBLE_DEVICES='2' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 2 --logdir exp/cifar10 &> logs/log_2 &
CUDA_VISIBLE_DEVICES='3' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 3 --logdir exp/cifar10 &> logs/log_3 &
CUDA_VISIBLE_DEVICES='4' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 4 --logdir exp/cifar10 &> logs/log_4 &
CUDA_VISIBLE_DEVICES='5' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 5 --logdir exp/cifar10 &> logs/log_5 &
CUDA_VISIBLE_DEVICES='6' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 6 --logdir exp/cifar10 &> logs/log_6 &
CUDA_VISIBLE_DEVICES='7' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 7 --logdir exp/cifar10 &> logs/log_7 &
wait;
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 8 --logdir exp/cifar10 &> logs/log_8 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 9 --logdir exp/cifar10 &> logs/log_9 &
CUDA_VISIBLE_DEVICES='2' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 10 --logdir exp/cifar10 &> logs/log_10 &
CUDA_VISIBLE_DEVICES='3' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 11 --logdir exp/cifar10 &> logs/log_11 &
CUDA_VISIBLE_DEVICES='4' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 12 --logdir exp/cifar10 &> logs/log_12 &
CUDA_VISIBLE_DEVICES='5' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 13 --logdir exp/cifar10 &> logs/log_13 &
CUDA_VISIBLE_DEVICES='6' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 14 --logdir exp/cifar10 &> logs/log_14 &
CUDA_VISIBLE_DEVICES='7' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 --num_experiments 16 --expid 15 --logdir exp/cifar10 &> logs/log_15 &
wait;
