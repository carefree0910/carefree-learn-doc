---
id: MNIST
title: MNIST
---

The MNIST dataset is a dataset of handwritten digits that is commonly used as the 'Hello World' dataset in Deep Learning domain. It contains 60,000 training images and 10,000 testing images, and
`carefree-learn` provided a straightforward API to access it.

MNIST dataset can be used for training various image processing systems. In this article, we will demonstrate how to actually utilize `carefree-learn` to solve these different tasks on MNIST dataset.

```python
# preparations

import torch
import cflearn

import numpy as np
import torch.nn as nn

# MNIST dataset could be prepared with this one line of code
data = cflearn.cv.MNISTData(batch_size=16, transform="to_tensor")

# for reproduction
np.random.seed(142857)
torch.manual_seed(142857)
```

:::tip
+ As shown above, the MNIST dataset could be easily turned into a `DLDataModule` instance, which is the common data interface used in `carefree-learn`.
+ The `transform` argument specifies which transform do we want to use to pre-process the input batch. See [Transforms](/docs/user-guides/computer-vision#transforms) section for more details.
:::


## Classification

| Python source code | Jupyter Notebook | Task |
|:---:|:---:|:---:|
| [run_clf.py](https://github.com/carefree0910/carefree-learn/blob/dev/examples/cv/mnist/run_clf.py) | [clf.ipynb](https://nbviewer.jupyter.org/github/carefree0910/carefree-learn/blob/dev/examples/cv/mnist/clf.ipynb) | Computer Vision 🖼️ |

For demo purpose, we are going to build a simple convolution-based classifier:

```python
@cflearn.register_module("simple_conv")
class SimpleConvClassifier(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(128, num_classes),
        )
```

We leveraged the [`register_module`](/docs/developer-guides/computer-vision-customization#customize-models) API here, which can turn a general `nn.Module` instance to a [`ModelProtocol`](/docs/design-principles/#model) in `carefree-learn`. After registered, it can be easily accessed with its name (`"simple_conv"`):

```python
cflearn.api.fit_cv(
    data,
    "simple_conv",
    {"in_channels": 1, "num_classes": 10},
    loss_name="cross_entropy",
    metric_names="acc",
    fixed_epoch=1,                                  # for demo purpose, we only train our model for 1 epoch
    cuda=0 if torch.cuda.is_available() else None,  # use CUDA if possible
)
```

Our model achieves 98.0400% accuracy on validation set within 1 epoch, not bad!
