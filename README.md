![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# rGAN: Label-Noise Robust Generative Adversarial Networks (CVPR 2019)

This repository provides PyTorch implementation for [rGAN](https://arxiv.org/abs/1811.11165).
rGAN can learn a *label-noise robust conditional generator* that can generate an image conditioned on the *clean* label rather than conditioned on the *noisy* label even when the *noisy* labeled data are only available during the training.

<img src="docs/images/examples.png" width=100% alt="rGAN examples">

## Paper

Label-Noise Robust Generative Adversarial Networks.<br>
[Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/), [Yoshitaka Ushiku](https://yoshitakaushiku.net/), and [Tatsuya Harada](https://www.mi.t.u-tokyo.ac.jp/harada/).<br>
In CVPR 2019 (Oral).

[[Project]](https://takuhirok.github.io/rGAN/) [[Paper]](https://arxiv.org/abs/1811.11165)

## Overview

Our task is, when given *noisy* labeled data, to construct a label-noise robust conditional generator that can generate an image conditioned on the *clean* label rather than conditioned on the *noisy* label. Our main idea for solving this problem is to incorporate a **noise transition model** (viewed as orange rectangles in (b) and (d); which represents a probability that a clean label is corrupted to a noisy label) into typical class conditional GANs. In particular, we develop two variants: **rAC-GAN** (b) and **rcGAN** (d) that are extensions of AC-GAN \[1] (a) and cGAN \[2, 3] (c), respectively.

<img src="docs/images/networks.png" width=100% alt="networks">

## Installation

Clone this repository.

```bash
git clone https://github.com/takuhirok/rGAN.git
cd rGAN/
```
First install Python 3.6 and then install [PyTorch](https://pytorch.org/) 1.0 and the other dependencies by

```bash
pip install -r requirements.txt
```

## Train

To train the model, use the following script.

```bash
bash ./scripts/train.sh \
	[acgan/racgan/cgan/rcgan] [sngan/wgangp/ctgan] [cifar10/cifar100] \
	[symmetric/asymmetric] [0/0.1/0.3/0.5/0.7/0.9] output_directory_path
```

Please choose one of the options in the square brackets ([ ]).

### Example 1
To train rcSN-GAN (**rcGAN** + **SN-GAN**) on **CIFAR-10** in the **symmetric** noise with a noise rate **0.5**, run the following script.

```bash
bash ./scripts/train.sh rcgan sngan cifar10 symmetric 0.5 outputs
```

### Example 2
To train rAC-CT-GAN (**rAC-GAN** + **CT-GAN**) on **CIFAR-100** in the **asymmetric** noise with a noise rate **0.7**, run the following script.

```bash
bash ./scripts/train.sh racgan ctgan cifar100 asymmetric 0.7 outputs
```

## Test
To generate images, use the following script.

```bash
python test.py --g_path trained_model_path --out output_directory_path
```

### Example

```bash
python test.py --g_path outputs/netG_iter_100000.pth --out samples
```

**NOTE:**
In our paper, we conducted early stopping based on the FID because some models collapsed before the end of the training.

## Citation
If you use this code for your research, please cite our paper.

```
@inproceedings{kaneko2019rGAN,
  title={Label-Noise Robust Generative Adversarial Networks},
  author={Kaneko, Takuhiro and Ushiku, Yoshitaka and Harada, Tatsuya},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Related work

1. A. Odena, C. Olah, and J. Shlens. Conditional image synthesis with auxiliary classifier GANs. In ICML, 2017. [[arXiv]](https://arxiv.org/abs/1610.09585)
2. M. Mirza and S. Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014. [[arXiv]](https://arxiv.org/abs/1411.1784)
3. T. Miyato and M. Koyama. cGANs with projection discriminator. In ICLR, 2018. [[arXiv]](https://arxiv.org/abs/1802.05637)