## AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights

Official PyTorch implementation of AdamP and SGDP optimizers | [Paper](https://arxiv.org/abs/2006.08217) | [Project page](https://clovaai.github.io/AdamP/)

**Byeongho Heo<sup>\*</sup>, Sanghyuk Chun<sup>\*</sup>, Seong Joon Oh, Dongyoon Han, Sangdoo Yun, Gyuwan Kim, Youngjung Uh, Jung-Woo Ha.** <br>
<sub>\* indicates equal contribution</sub>

NAVER AI LAB, NAVER CLOVA

### Abstract

Normalization techniques are a boon for modern deep learning. They let weights converge more quickly with often better generalization performances. It has been argued that the normalization-induced scale invariance among the weights provides an advantageous ground for gradient descent (GD) optimizers: the effective step sizes are automatically reduced over time, stabilizing the overall training procedure. It is often overlooked, however, that the additional introduction of momentum in GD optimizers results in a far more rapid reduction in effective step sizes for scale-invariant weights, a phenomenon that has not yet been studied and may have caused unwanted side effects in the current practice. This is a crucial issue because arguably the vast majority of modern deep neural networks consist of (1) momentum-based GD (e.g. SGD or Adam) and (2) scale-invariant parameters. In this paper, we verify that the widely-adopted combination of the two ingredients lead to the premature decay of effective step sizes and sub-optimal model performances. We propose a simple and effective remedy, SGDP and AdamP: get rid of the radial component, or the norm-increasing direction, at each optimizer step. Because of the scale invariance, this modification only alters the effective step sizes without changing the effective update directions, thus enjoying the original convergence properties of GD optimizers. Given the ubiquity of momentum GD and scale invariance in machine learning, we have evaluated our methods against the baselines on 13 benchmarks. They range from vision tasks like classification (e.g. ImageNet), retrieval (e.g. CUB and SOP), and detection (e.g. COCO) to language modelling (e.g. WikiText) and audio classification (e.g. DCASE) tasks. We verify that our solution brings about uniform gains in those benchmarks.

## How does it work?

Please visit our [project page](https://clovaai.github.io/AdamP/).

## Updates

- **Aug 27, 2020**: built-in cosine similarity and fix warning (v0.3.0)
- **Jun 19, 2020**: nesterov update (v0.2.0)
- **Jun 15, 2020**: Initial upload (v0.1.0)

## Getting Started

### Installation

```
pip3 install adamp
```

### Usage

Usage is exactly same as [torch.optim](https://pytorch.org/docs/stable/optim.html) library!

```python
from adamp import AdamP

# define your params
optimizer = AdamP(params, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-2)
```

```python
from adamp import SGDP

# define your params
optimizer = SGDP(params, lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)
```

## Arguments
`SGDP` and `AdamP` share arguments with [torch.optim.SGD](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) and [torch.optim.Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam).
There are two additional hyperparameters; we recommend using the default values.
- `delta` : threhold that determines whether a set of parameters is scale invariant or not (default: 0.1)
- `wd_ratio` : relative weight decay applied on _scale-invariant_ parameters compared to that applied on _scale-variant_ parameters (default: 0.1)

Both `SGDP` and `AdamP` support Nesterov momentum.
- `nesterov` : enables Nesterov momentum (default: False)

## License

This project is distributed under MIT license.

```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## How to cite

```
@article{heo2020adamp,
    title={Slowing Down the Weight Norm Increase in Momentum-based Optimizers},
    author={Heo, Byeongho and Chun, Sanghyuk and Oh, Seong Joon and Han, Dongyoon and Yun, Sangdoo and Uh, Youngjung and Ha, Jung-Woo},
    year={2020},
    journal={arXiv preprint arXiv:2006.08217},
}
```
