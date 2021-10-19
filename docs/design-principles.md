---
id: design-principles
title: Design Principles
---

`carefree-learn` was designed to support most commonly used methods with *carefree* APIs. Moreover, `carefree-learn` was also designed with interface which is general enough, so that more sophisticated functionality can also be easily integrated in the future. This brings a tension in how to create abstractions in code, which is a challenge for us:

+ On the one hand, it requires a reasonably high-level abstraction so that users can easily work around with it in a standard way, without having to worry too much about the details.
+ On the other hand, it also needs to have a very thin abstraction to allow users to do (many) other things in new ways. Breaking existing abstractions and replacing them with new ones should be fairly easy.

In `carefree-learn`, there are five main design principles that address this tension together:

+ Divide `carefree-learn` into three parts: [`Model`](#model), [`Trainer`](#trainer) and [`Pipeline`](#pipeline).
+ Build some [`Common Blocks`](#common-blocks) which shall be leveraged across different [`Model`](#model)s.
+ Manage models / blocks with `register` mechanism, so they can be accessed via their names (see [Register Mechanism](#register-mechanism)).

We will introduce the details in the following subsections.


## Common Blocks

> Source code: [blocks.py](https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/modules/blocks.py).

`carefree-learn` implements many basic blocks which can directly form famous models, such as **VAE**, **AdaIN**, **CycleGAN**, **BERT**, **ViT**, **FNet**, **StyleGAN**, **U^2 Net** etc. The best of `carefree-learn` is that, it not only reproduces the official implementations, but also reuses everything it could. For example:
- The `Decoder` used in `VAE` and `CycleGAN` is the same (with different args / kwargs).
- The `Transformer` used in `BERT` and `Vit` is the same (with different args / kwargs).
- `Transformer` and `FNet` shares most of the codes, except that `Transformer` uses `Attention` but `FNet` uses fourier transform.
- And much more...


## Configurations

In general, there are three kinds of configurations in `carefree-learn`:
- Model configurations.
- Trainer configurations.
- Pipeline configurations, which is basically constructed by the above two configurations.

See [specifying configurations](getting-started/configurations#specify-configurations) for more details.


## Register Mechanism

> Source code: [`WithRegister`](https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/misc/toolkit.py#L383).

In `carefree-learn`, it is likely to see `@xxx.register(...)` all around. This is very useful when we want to provide many useful defaults for users.

Here's a code snippet that well demonstrates how to use `register` mechanism:

```python
from cflearn.misc.toolkit import WithRegister

foo = {}

class FooBase(WithRegister):
    d = foo

@FooBase.register("bar")
class Bar(FooBase):
    def __init__(self, name="foobar"):
        self.name = name

print(foo["bar"]().name)                             # foobar
print(FooBase.get("bar")().name)                     # foobar
print(FooBase.make("bar", {"name": "barfoo"}).name)  # barfoo
```


## Model

> Source code: [`ModelProtocol`](https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/protocol.py#L109).

In `carefree-learn`, a `Model` should implement the core algorithms. It's basically an `nn.Module`, with some extra useful functions:

```python
class ModelProtocol(nn.Module, WithRegister, metaclass=ABCMeta):
    d = model_dict

    ...

    @property
    def device(self) -> torch.device:
        return list(self.parameters())[0].device

    def onnx_forward(self, batch: tensor_dict_type) -> Any:
        return self.forward(0, batch)

    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
        self.forward(batch_idx, batch)
```

As shown above, there are two special `forward` methods defined in a `Model`, which allows us to customize `onnx` export procedure and `summary` procedure respectively.

:::tip
See [ModelProtocol](developer-guides/general-customization#modelprotocol) section for more details.
:::


## Trainer

> Source code: [`Trainer`](https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/trainer.py#L226).

In `carefree-learn`, a `Trainer` should implement the training loop, which includes updating trainable parameters with an optimizer (or, some optimizers), verbosing metrics / losses, checkpointing, early stopping, logging, etc.

:::note
Although we can construct a `Trainer` from scratch, `carefree-learn` provides [`make_trainer`](https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/misc/internal_/trainer.py#L19) function, which contains many useful default `Trainer` values.
:::


## Pipeline

> Source code: [`DLPipeline`](https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/pipeline.py#L90).

In `carefree-learn`, a `Pipeline` should implement the high-level parts (e.g. `fit`, `predict`, `save`, `load`, etc.), and will be the (internal) user interface. It's basically a 'wrapper' which can use a [`Trainer`](#trainer) to train a [`Model`](#model) properly, and can serialize the necessary information to disk.

:::note
Although `carefree-learn` focuses on Deep Learning tasks, the most general abstraction ([`PipelineProtocol`](https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/pipeline.py#L57)) can actually utilize traditional Machine Learning models, such as `LinearRegression` from `scikit-learn`.
:::
