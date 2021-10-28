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


## Variational Auto Encoder

| Python source code | Jupyter Notebook | Task |
|:---:|:---:|:---:|
| [run_vae.py](https://github.com/carefree0910/carefree-learn/blob/dev/examples/cv/mnist/run_vae.py) | [vae.ipynb](https://nbviewer.jupyter.org/github/carefree0910/carefree-learn/blob/dev/examples/cv/mnist/vae.ipynb) | Computer Vision 🖼️ |

```python
import torch.nn.functional as F

from typing import Any
from typing import Dict
from typing import Optional
from cflearn.types import losses_type
from cflearn.types import tensor_dict_type
from cflearn.protocol import TrainerState
from cflearn.misc.toolkit import interpolate
from cflearn.modules.blocks import Lambda
from cflearn.modules.blocks import UpsampleConv2d
```

For demo purpose, we are going to build a simple convolution-based VAE:

```python
@cflearn.register_module("simple_vae")
class SimpleVAE(nn.Module):
    def __init__(self, in_channels: int, img_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
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
        )
        self.decoder = nn.Sequential(
            Lambda(lambda t: t.view(-1, 4, 4, 4), name="reshape"),
            nn.Conv2d(4, 128, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            UpsampleConv2d(128, 64, kernel_size=3, padding=1, factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            UpsampleConv2d(64, 32, kernel_size=3, padding=1, factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            UpsampleConv2d(32, in_channels, kernel_size=3, padding=1, factor=2),
            Lambda(lambda t: interpolate(t, size=img_size, mode="bilinear")),
        )

    def forward(self, net: torch.Tensor) -> Dict[str, torch.Tensor]:
        net = self.encoder(net)
        mu, log_var = net.chunk(2, dim=1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        net = eps * std + mu
        net = self.decoder(net)
        return {"mu": mu, "log_var": log_var, cflearn.PREDICTIONS_KEY: net}
```

There are quite a few details that worth to be mentioned:
+ We leveraged the [`register_module`](/docs/developer-guides/computer-vision-customization#customize-models) API here, which can turn a general `nn.Module` instance to a [`ModelProtocol`](/docs/design-principles#model) in `carefree-learn`. After registered, it can be easily accessed with its name (`"simple_vae"`)
+ We leveraged some built-in [common blocks](/docs/design-principles#common-blocks) of `carefree-learn` to build our simple VAE:
  + `Lambda`, which can turn a function to an `nn.Module`.
  + `UpsampleConv2d`, which can be used to upsample the input image.
  + `interpolate`, which is a handy function to resize the input image to the desired size.

After we finished implementing our model, we need to implement the special loss used in VAE tasks:

```python
@cflearn.register_loss_module("simple_vae")
@cflearn.register_loss_module("simple_vae_foo")
class SimpleVAELoss(cflearn.LossModule):
    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        # reconstruction loss
        original = batch[cflearn.INPUT_KEY]
        reconstruction = forward_results[cflearn.PREDICTIONS_KEY]
        mse = F.mse_loss(reconstruction, original)
        # kld loss
        mu = forward_results["mu"]
        log_var = forward_results["log_var"]
        kld_losses = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        kld_loss = torch.mean(kld_losses, dim=0)
        # gather
        loss = mse + 0.001 * kld_loss
        return {"mse": mse, "kld": kld_loss, cflearn.LOSS_KEY: loss}
```

+ We used `register_loss_module` to register a general `LossModule` instance to a `LossProtocol` in `carefree-learn`.
+ We can call `register_loss_module` multiple times to assign multiple names to the same loss function.
+ When the loss function shares the same name with the model, we don't need to specify the `loss_name` argument explicitly:

```python
# Notice that we don't need to explicitly specify `loss_name`!
cflearn.api.fit_cv(
    data,
    "simple_vae",
    {"in_channels": 1, "img_size": 28},
    fixed_epoch=1,                                  # for demo purpose, we only train our model for 1 epoch
    cuda=0 if torch.cuda.is_available() else None,  # use CUDA if possible
)
```

Of course, we can still specify `loss_name` explicitly:

```python
cflearn.api.fit_cv(
    data,
    "simple_vae",
    {"in_channels": 1, "img_size": 28},
    loss_name="simple_vae_foo",                     # we used the second registered name here
    fixed_epoch=1,                                  # for demo purpose, we only train our model for 1 epoch
    cuda=0 if torch.cuda.is_available() else None,  # use CUDA if possible
)
```


## Generative Adversarial Network

| Python source code | Jupyter Notebook | Task |
|:---:|:---:|:---:|
| [run_gan.py](https://github.com/carefree0910/carefree-learn/blob/dev/examples/cv/mnist/run_gan.py) | [gan.ipynb](https://nbviewer.jupyter.org/github/carefree0910/carefree-learn/blob/dev/examples/cv/mnist/gan.ipynb) | Computer Vision 🖼️ |

```python
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional
from torch.optim import Optimizer
from cflearn.types import tensor_dict_type
from cflearn.protocol import StepOutputs
from cflearn.protocol import TrainerState
from cflearn.protocol import MetricsOutputs
from cflearn.protocol import DataLoaderProtocol
from cflearn.constants import INPUT_KEY
from cflearn.constants import PREDICTIONS_KEY
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import interpolate
from cflearn.misc.toolkit import toggle_optimizer
from cflearn.modules.blocks import Lambda
from cflearn.modules.blocks import UpsampleConv2d
from torch.cuda.amp.grad_scaler import GradScaler
```

For demo purpose, we are going to build a simple convolution-based GAN. But first, let's build the loss function of GAN:

```python
class GANLoss(nn.Module):
    def __init__(self):  # type: ignore
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))

    def expand_target(self, tensor: Tensor, use_real_label: bool) -> Tensor:
        target = self.real_label if use_real_label else self.fake_label
        return target.expand_as(tensor)  # type: ignore

    def forward(self, predictions: Tensor, use_real_label: bool) -> Tensor:
        target_tensor = self.expand_target(predictions, use_real_label)
        loss = self.loss(predictions, target_tensor)
        return loss
```

Although the concept of GAN is fairly easy, it's pretty complicated if we want to implement it with a 'pre-defined' framework. In order to provide full flexibility, `carefree-learn` exposed two methods for users:
+ `train_step`, which is used to control **ALL** training behaviours, including:
  + calculate losses
  + apply back propagation
  + perform [automatic mixed precision](https://pytorch.org/docs/stable/amp.html), [gradient norm clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) and so on
+ `evaluate_step`, which is used to define the final metric that we want to monitor.

Besides, we also need to define the `forward` method, as usual.

```python
@cflearn.register_custom_module("simple_gan")
class SimpleGAN(cflearn.CustomModule):
    def __init__(self, in_channels: int, img_size: int, latent_dim: int):
        super().__init__()
        if not latent_dim % 16 == 0:
            raise ValueError(f"`latent_dim` ({latent_dim}) should be divided by 16")
        self.latent_dim = latent_dim
        latent_channels = latent_dim // 16
        self.generator = nn.Sequential(
            Lambda(lambda t: t.view(-1, latent_channels, 4, 4), name="reshape"),
            nn.Conv2d(latent_channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            UpsampleConv2d(128, 64, kernel_size=3, padding=1, factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            UpsampleConv2d(64, 32, kernel_size=3, padding=1, factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            UpsampleConv2d(32, in_channels, kernel_size=3, padding=1, factor=2),
            Lambda(lambda t: interpolate(t, size=img_size, mode="bilinear")),
        )
        self.discriminator = nn.Sequential(
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
        )
        self.loss = GANLoss()

    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        optimizers: Dict[str, Optimizer],
        use_amp: bool,
        grad_scaler: GradScaler,
        clip_norm_fn: Callable[[], None],
        scheduler_step_fn: Callable[[], None],
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        net = batch[INPUT_KEY]
        # we will explain where do these keys come from in the following markdown block
        opt_g = optimizers["core.g_parameters"]
        opt_d = optimizers["core.d_parameters"]
        # generator step
        toggle_optimizer(self, opt_g)
        with torch.cuda.amp.autocast(enabled=use_amp):
            sampled = self.sample(len(net))
            pred_fake = self.discriminator(sampled)
            g_loss = self.loss(pred_fake, use_real_label=True)
        grad_scaler.scale(g_loss).backward()
        clip_norm_fn()
        grad_scaler.step(opt_g)
        grad_scaler.update()
        opt_g.zero_grad()
        # discriminator step
        toggle_optimizer(self, opt_d)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_real = self.discriminator(net)
            loss_d_real = self.loss(pred_real, use_real_label=True)
            pred_fake = self.discriminator(sampled.detach().clone())
            loss_d_fake = self.loss(pred_fake, use_real_label=False)
            d_loss = 0.5 * (loss_d_fake + loss_d_real)
        grad_scaler.scale(d_loss).backward()
        clip_norm_fn()
        grad_scaler.step(opt_d)
        grad_scaler.update()
        opt_d.zero_grad()
        # finalize
        scheduler_step_fn()
        forward_results = {PREDICTIONS_KEY: sampled}
        loss_dict = {
            "g": g_loss.item(),
            "d": d_loss.item(),
            "d_fake": loss_d_fake.item(),
            "d_real": loss_d_real.item(),
        }
        return StepOutputs(forward_results, loss_dict)

    def evaluate_step(
        self,
        loader: DataLoaderProtocol,
        portion: float,
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        trainer: Any,
    ) -> MetricsOutputs:
        loss_items: Dict[str, List[float]] = {}
        for i, batch in enumerate(loader):
            if i / len(loader) >= portion:
                break
            batch = to_device(batch, self.device)
            net = batch[INPUT_KEY]
            sampled = self.sample(len(net))
            pred_fake = self.discriminator(sampled)
            g_loss = self.loss(pred_fake, use_real_label=True)
            pred_real = self.discriminator(net)
            d_loss = self.loss(pred_real, use_real_label=True)
            loss_items.setdefault("g", []).append(g_loss.item())
            loss_items.setdefault("d", []).append(d_loss.item())
        # gather
        mean_loss_items = {k: sum(v) / len(v) for k, v in loss_items.items()}
        mean_loss_items[cflearn.LOSS_KEY] = sum(mean_loss_items.values())
        score = weighted_loss_score_fn(mean_loss_items)
        return MetricsOutputs(score, mean_loss_items)

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.generator.parameters())

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.discriminator.parameters())

    def sample(self, num_samples: int) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return self.generator(z)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.sample(len(batch[INPUT_KEY]))}
```

We leveraged the `register_custom_module` API here, which can turn a general `CustomModule` instance to a [`ModelProtocol`](/docs/design-principles#model) in `carefree-learn`. After registered, it can be easily accessed with its name (`"simple_gan"`).

There are two more things that are worth mentioning:
+ When using models with custom steps, we don't need to specify `loss_name` anymore, because the losses are calculated inside `train_step`.
+ The `register_custom_module` API will generate a [`ModelProtocol`](/docs/design-principles#model), whose `core` property points to the original `CustomModule`. From the above codes, we can see that `SimpleGAN` implements `g_parameters` and `d_parameters`, which means the `self.core.g_parameters` and `self.core.d_parameters` of the generated [`ModelProtocol`](/docs/design-principles#model) will be two sets of parameters that we wish to optimize.
  + In this case, the `core.g_parameter` and `core.d_parameters` will be the optimize `scope` of the generated [`ModelProtocol`](/docs/design-principles#model). That's why we access the optimizers with them.
  + Please refer to the [`OptimizerPack`](/docs/getting-started/configurations#optimizerpack) section for more details.

```python
# Notice that we don't need to explicitly specify `loss_name`!
cflearn.api.fit_cv(
    data,
    "simple_gan",
    {"in_channels": 1, "img_size": 28, "latent_dim": 128},
    optimizer_settings={
        "core.g_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
        "core.d_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
    },
    fixed_epoch=1,                                  # for demo purpose, we only train our model for 1 epoch
    cuda=0 if torch.cuda.is_available() else None,  # use CUDA if possible
)
```
