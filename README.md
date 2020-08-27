## Slowing Down the Weight Norm Increase in Momentum-based Optimizers

Official PyTorch implementation of AdamP and SGDP optimizers | [Paper](https://arxiv.org/abs/2006.08217) | [Project page](https://clovaai.github.io/AdamP/)

**Byeongho Heo<sup>\*</sup>, Sanghyuk Chun<sup>\*</sup>, Seong Joon Oh, Dongyoon Han, Sangdoo Yun, Youngjung Uh, Jung-Woo Ha.** <br>
<sub>\* indicates equal contribution</sub>

Clova AI Research, NAVER Corp.

### Abstract

Normalization techniques, such as batch normalization (BN), have led to significant improvements in deep neural network performances. Prior studies have analyzed the benefits of the resulting scale invariance of the weights for the gradient descent (GD) optimizers: it leads to a stabilized training due to the auto-tuning of step sizes. However, we show that, combined with the momentum-based algorithms, the scale invariance tends to induce an excessive growth of the weight norms. This in turn overly suppresses the effective step sizes during training, potentially leading to sub-optimal performances in deep neural networks. We analyze this phenomenon both theoretically and empirically. We propose a simple and effective solution: at each iteration of momentum-based GD optimizers (e.g., SGD or Adam) applied on scale-invariant weights (e.g., Conv weights preceding a BN layer), we remove the radial component (i.e., parallel to the weight vector) from the update vector. Intuitively, this operation prevents the unnecessary update along the radial direction that only increases the weight norm without contributing to the loss minimization. We verify that the modified optimizers SGDP and AdamP successfully regularize the norm growth and improve the performance of a broad set of models. Our experiments cover tasks including image classification and retrieval, object detection, robustness benchmarks, and audio classification.

## How does it work?

Please visit our [project page](https://clovaai.github.io/AdamP/).

## Updates

- **Jun 19, 2020**: built-in cosine similarity and fix warning (v0.3.0)
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
