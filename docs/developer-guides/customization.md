---
id: customization
title: Build Your Own Models
sidebar_label: Customization
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

In this page we will go through some basic concepts we need to know to build our own models in `carefree-learn`. Customizing `carefree-learn` could be very easy if you only want to construct existing blocks to form a new model structure, and should also be fairly straight forward even if you want to implement your own blocks.

:::tip
There's a step-by-step example [here](../user-guides/examples#operations) which will walk you through the most important concepts with sufficient codes and experiments.
:::

:::note
In order to better understand the following contents, it is recommended to first understand the basic concepts mentioned in the [Design Principles](../design-principles#model).
:::


## `Configs`

Before we dive into the details of customization, we should first understand how `carefree-learn` manages its `Configs`. It is actually not more than an ordinary Python `dict`, except it can be *registered* in a certain *hierarchy* format. Basically, a `Configs` need to define a `scope` and a `name` for users to access it, where

+ A `scope` represents a `module`. 
+ A `name` represents the different `version` of the corresponding configuration.

For example, if we want to configure a `module` named `Foo` with different `dummy_value`:

```python
class Foo:
    def __init__(self, dummy_value: float):
        self.dummy = dummy_value
    
    def print(self) -> None:
        print(self.dummy)
```

Then we can leverage `cflearn.register_config` to register different configurations:

```python
import cflearn

@cflearn.register_config("foo", "one")
class FooOneConfig(cflearn.Configs):
    def get_default(self):
        return {"dummy_value": 1.0}

@cflearn.register_config("foo", "two")
class FooTwoConfig(cflearn.Configs):
    def get_default(self):
        return {"dummy_value": 2.0}
```

:::info
Notice that we've implemented `get_default` for each class, which is the only necessary method we need to inherit from `Configs`.
:::

After registration, we can access them through their names, which is very convenient in many use cases (e.g. hyper parameter optimization):

```python
for name in ["one", "two"]:
    cfg = cflearn.Configs.get("foo", name)
    config = cfg.pop()
    Foo(**config).print()

# 1.0
# 2.0
```

:::info
Notice that we used `Configs.pop` to generate a Python `dict` for further usages.
:::

What's going on under the hood is that `carefree-learn` maintains a global `configs_dict` with following hierarchy:

```python
{
    "scope_0": {
        "name_00": config_class_00,
        "name_01": config_class_01,
    },
    "scope_1": {
        "name_10": config_class_10,
        "name_11": config_class_11,
    },
    ...,
    "scope_k": {
        "name_k0": config_class_k0,
        "name_k1": config_class_k1,
    },
}
```

So after the registration mentioned above, this global `configs_dict` will be updated to:

```python
{
    ...,
    "foo": {
        "one": FooOneConfig,
        "two": FooTwoConfig,
    }
}
```

### `HeadConfigs`

A `HeadConfigs` inherits from `Configs` and holds more information. The reason why we implement an extra sub-class of `Configs` is that we usually need more information in `head` than in `transform` and `extractor`. For instance, we need to know the data dimensions to inference the default `output_dim`.


## Constructing Existing Blocks

With the help of `Configs`, constructing existing blocks is pretty easy because we can access different configurations by specifying their `scope` and `name`. In fact, as mentioned in [`Design Principles`](../design-principles#examples), `carefree-learn` itself is actually implementing its models by such similar process:

<Tabs
  groupId="models"
  defaultValue="linear"
  values={[
    {label: 'Linear', value: 'linear'},
    {label: 'FCNN', value: 'fcnn'},
    {label: 'Wide & Deep', value: 'wnd'},
    {label: 'RNN', value: 'rnn'},
    {label: 'Transformer', value: 'transformer'},
  ]
}>
<TabItem value="linear">

```python
@ModelBase.register("linear")
@ModelBase.register_pipe("linear")
class LinearModel(ModelBase):
    pass
```

</TabItem>
<TabItem value="fcnn">

```python
@ModelBase.register("fcnn")
@ModelBase.register_pipe("fcnn")
class FCNN(ModelBase):
    pass
```

</TabItem>
<TabItem value="wnd">

```python
@ModelBase.register("wnd")
@ModelBase.register_pipe("fcnn", transform="embedding")
@ModelBase.register_pipe("linear", transform="one_hot_only")
class WideAndDeep(ModelBase):
    pass
```

</TabItem>
<TabItem value="rnn">

```python
@ModelBase.register("rnn")
@ModelBase.register_pipe("rnn", head="fcnn")
class RNN(ModelBase):
    pass
```

</TabItem>
<TabItem value="transformer">

```python
@ModelBase.register("transformer")
@ModelBase.register_pipe("transformer", head="fcnn")
class Transformer(ModelBase):
    pass
```

</TabItem>
</Tabs>

### `ModelBase.register`

In `carefree-learn` we implemented an alias for `ModelBase.register`:

```python
def register_model(name: str) -> Callable[[Type], Type]:
    return ModelBase.register(name)
```

It can be used to register a new model and access it through its name, which is very convenient in many use cases (e.g. hyper parameter optimization).

### `ModelBase.register_pipe`

In `carefree-learn` we implemented an alias for `ModelBase.register_pipe`:

```python
def register_pipe(
    key: str,
    *,
    transform: str = "default",
    extractor: Optional[str] = None,
    head: Optional[str] = None,
    extractor_config: str = "default",
    head_config: str = "default",
    extractor_meta_scope: Optional[str] = None,
    head_meta_scope: Optional[str] = None,
) -> Callable[[Type], Type]:
    return ModelBase.register_pipe(
        key,
        transform=transform,
        extractor=extractor,
        head=head,
        extractor_config=extractor_config,
        head_config=head_config,
        extractor_meta_scope=extractor_meta_scope,
        head_meta_scope=head_meta_scope,
    )
```

In this definition, the `extractor` and `head` arguments represent the corresponding `scope`, while `transform`, `extractor_config` and `head_config` represent the corresponding `name`. In other words, this definition actually means:

```python
transform_cfg = cflearn.Configs.get("transform", transform)
extractor_cfg = cflearn.Configs.get(extractor, extractor_config)
head_cfg = cflearn.HeadConfigs.get(head, head_config)
```

:::note
+ There is only one `scope` for `transform` because the number of choices of `transform` is limited (see [transform](../design-principles#transform) for more details).
+ We are using `cflearn.HeadConfigs` to fetch configurations for `head`, as mentioned in [HeadConfigs](#headconfigs) section.
:::

Besides these, there still remains a `key` argument in `register_pipe` and this is where many default logics hide under the hood:

```python
if head is None:
    head = key
elif extractor is None:
    extractor = key
if extractor is None:
    extractor = "identity"
```

These logics simplify the definitions of some common structures, so in `carefree-learn` we only care about the `key` argument in most cases.

:::tip
For the `key` itself, we only need to guarantee that different [pipe](../design-principles#pipe) corresponds to different `key`.
:::

At the last part of this section, we will demonstrate how could we build a new model with following properties:

+ Use one hot features to train a `DNDF` `head`.
+ Use one hot features and numerical features to train a `linear` `head`.
+ Use numerical features to train an `fcnn` `head`.
+ Use embedding features to train an `fcnn` `head`.

```python
import cflearn

@cflearn.register_model("brand_new_model")
@cflearn.register_pipe("dndf", transform="one_hot_only")
@cflearn.register_pipe("linear", transform="one_hot")
@cflearn.register_pipe("fcnn", transform="numerical")
@cflearn.register_pipe("fcnn2", transform="embedding_only", extractor="identity", head="fcnn")
class BrandNewModel(cflearn.ModelBase):
    pass
```

We can actually play with it:

```python
import numpy as np

numerical = np.random.random([10000, 5])
categorical = np.random.randint(0, 10, [10000, 5])
x = np.hstack([numerical, categorical])
y = np.random.random([10000, 1])
m = cflearn.make("brand_new_model").fit(x, y)
print(m.model)
```

<details><summary><b>Which yields</b></summary>
<p>

```text
BrandNewModel(
  (pipes): Pipes(
    (fcnn2): embedding_only_identity_default -> fcnn_default
    (fcnn): numerical_identity_default -> fcnn_default
    (linear): one_hot_identity_default -> linear_default
    (dndf): one_hot_only_identity_default -> dndf_default
  )
  (loss): L1Loss()
  (encoder): Encoder(
    (embeddings): ModuleList(
      (0): Embedding(
        (core): Lambda(embedding: 50 -> 4)
      )
    )
    (one_hot_encoders): ModuleList(
      (0): OneHot(
        (core): Lambda(one_hot_10)
      )
      (1): OneHot(
        (core): Lambda(one_hot_10)
      )
      (2): OneHot(
        (core): Lambda(one_hot_10)
      )
      (3): OneHot(
        (core): Lambda(one_hot_10)
      )
      (4): OneHot(
        (core): Lambda(one_hot_10)
      )
    )
    (embedding_dropout): Dropout(keep=0.8)
  )
  (transforms): ModuleDict(
    (embedding_only): Transform(
      (use_one_hot): False
      (use_embedding): True
      (only_categorical): True
    )
    (numerical): Transform(
      (use_one_hot): False
      (use_embedding): False
      (only_categorical): False
    )
    (one_hot): Transform(
      (use_one_hot): True
      (use_embedding): False
      (only_categorical): False
    )
    (one_hot_only): Transform(
      (use_one_hot): True
      (use_embedding): False
      (only_categorical): True
    )
  )
  (extractors): ModuleDict(
    (embedding_only_identity_default): Identity()
    (numerical_identity_default): Identity()
    (one_hot_identity_default): Identity()
    (one_hot_only_identity_default): Identity()
  )
  (heads): ModuleDict(
    (fcnn2): FCNNHead(
      (mlp): MLP(
        (mappings): ModuleList(
          (0): Mapping(
            (linear): Linear(
              (linear): Linear(in_features=20, out_features=64, bias=False)
            )
            (bn): BN(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace=True)
            (dropout): Dropout(keep=0.5)
          )
          (1): Mapping(
            (linear): Linear(
              (linear): Linear(in_features=64, out_features=64, bias=False)
            )
            (bn): BN(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace=True)
            (dropout): Dropout(keep=0.5)
          )
          (2): Linear(
            (linear): Linear(in_features=64, out_features=1, bias=True)
          )
        )
      )
    )
    (fcnn): FCNNHead(
      (mlp): MLP(
        (mappings): ModuleList(
          (0): Mapping(
            (linear): Linear(
              (linear): Linear(in_features=5, out_features=64, bias=False)
            )
            (bn): BN(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace=True)
            (dropout): Dropout(keep=0.5)
          )
          (1): Mapping(
            (linear): Linear(
              (linear): Linear(in_features=64, out_features=64, bias=False)
            )
            (bn): BN(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): ReLU(inplace=True)
            (dropout): Dropout(keep=0.5)
          )
          (2): Linear(
            (linear): Linear(in_features=64, out_features=1, bias=True)
          )
        )
      )
    )
    (linear): LinearHead(
      (linear): Linear(
        (linear): Linear(in_features=55, out_features=1, bias=True)
      )
    )
    (dndf): DNDFHead(
      (dndf): DNDF(
        (tree_proj): Linear(
          (linear): Linear(in_features=50, out_features=310, bias=True)
          (pruner): Pruner(method='auto_prune')
        )
      )
    )
  )
)
```

</p>
</details>


## Customizing New Modules


