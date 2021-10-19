---
id: computer-vision-customization
title: Computer Vision 🖼️
---

In this section, we will introduce:
+ How to define a new model & How to use it for training.
+ How to customize transforms (i.e. customize data augmentations).


## Customize Models

Basically, there are no differences between defining our own Computer Vision models and defining an `nn.Module`, so the information provided in the [Customize Models in General](general-customization#customize-models) section is pretty enough.

However, to make things even easier, `carefree-learn` provides `register_module` API, which can directly register an `nn.Module` to a `ModelProtocol`:

```python
import cflearn
from torch.nn import Module

@cflearn.register_module("my_fancy_model")
class MyFancyModel(Module):
    def __init__(self, foo):
        super().__init__()
        self.foo = foo
    
    def forward(self, x):
        return x + 1
```

In this case, we assume that this model receive one tensor as input, and output one tensor. Then, `carefree-learn` will internally convert the batches to input tensors and convert the output tensors to tensor dictionaries.

```python
import torch

m = cflearn.cv.CarefreePipeline("my_fancy_model", {"foo": "bar"})
m.build({})
my_fancy_model = m.model.core
print(my_fancy_model.foo)               # bar
print(my_fancy_model(torch.tensor(0)))  # tensor(1)
```

:::note
Notice that the original `nn.Module` will be constructed as `ModelProtocol.core`.
:::


## Customize Transforms

:::caution
To be continued...
:::

