Implementation of our reconstruction attack on InstaHide.

Is Private Learning Possible with Instance Encoding?
Nicholas Carlini, Samuel Deng, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Shuang Song, Abhradeep Thakurta, Florian Tramer
https://arxiv.org/abs/2011.05315


## Overview

InstaHide is a recent privacy-preserving machine learning framework.
It takes a (sensitive) dataset and generates encoded images that are privacy-preserving.
Our attack breaks InstaHide and shows it does not offer meaningful privacy.
Given the encoded dataset, we can recover a near-identical copy of the original images.

This repository implements the attack described in our paper. It consists of a number of
steps that shoul be run sequentially. It assumes access to pre-trained neural network
classifiers that should be downloaded following the steps below.


### Requirements

* Python, version &ge; 3.5
* jax
* jaxlib
* objax (https://github.com/google/objax)
* PIL
* sklearn


### Running the attack

To reproduce our results and run the attack, each of the files should be run in turn.

0. Download the necessary dependency files:
- (encryption.npy)[https://www.dropbox.com/sh/8zdsr1sjftia4of/AAA-60TOjGKtGEZrRmbawwqGa?dl=0] and (labels.npy)[https://www.dropbox.com/sh/8zdsr1sjftia4of/AAA-60TOjGKtGEZrRmbawwqGa?dl=0] from the (InstaHide Challenge)[https://github.com/Hazelsuko07/InstaHide_Challenge]
- The (saved models)[https://drive.google.com/file/d/1YfKzGRfnnzKfUKpLjIRXRto8iD4FdwGw/view?usp=sharing] used to run the attack
- Set up all the requirements as above

1. Run `step_1_create_graph.py`. Produce the similarity graph to pair together encoded images that share an original image.

2. Run `step_2_color_graph.py`. Color the graph to find 50 dense cliques.

3. Run `step_3_second_graph.py`. Create a new bipartite similarity graph.

4. Run `step_4_final_graph.py`. Solve the matching problem to assign encoded images to original images.

5. Run `step_5_reconstruct.py`. Reconstruct the original images.

6. Run `step_6_adjust_color.py`. Adjust the color curves to match.

7. Run `step_7_visualize.py`. Show the final resulting images.

## Citation

You can cite this attack at

```
@inproceedings{carlini2021private,
  title={Is Private Learning Possible with Instance Encoding?},
  author={Carlini, Nicholas and Deng, Samuel and Garg, Sanjam and Jha, Somesh and Mahloujifar, Saeed and Mahmoody, Mohammad and Thakurta, Abhradeep and Tram{\`e}r, Florian},
  booktitle={2021 IEEE Symposium on Security and Privacy (SP)},
  pages={410--427},
  year={2021},
  organization={IEEE}
}
```