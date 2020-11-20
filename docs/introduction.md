---
id: introduction
title: Introduction
slug: /
---

## Advantages

Like many similar projects, `carefree-learn` can be treated as a high-level library to help with training neural networks in PyTorch. However, `carefree-learn` does less and more than that.

+ `carefree-learn` focuses on tabular (structured) datasets, instead of unstructured datasets (e.g. NLP datasets or CV datasets).
+ `carefree-learn` provides an end-to-end pipeline on tabular datasets, including **AUTOMATICALLY** deal with (this part is mainly handled by [`carefree-data`](https://github.com/carefree0910/carefree-data), though):
    + Detection of redundant feature columns which can be excluded (all SAME, all DIFFERENT, etc).
    + Detection of feature columns types (whether a feature column is string column / numerical column / categorical column).
    + Imputation of missing values.
    + Encoding of string columns and categorical columns (Embedding or One Hot Encoding).
    + Pre-processing of numerical columns (Normalize, Min Max, etc.).
    + And much more...
+ `carefree-learn` can help you deal with almost **ANY** kind of tabular datasets, no matter how 'dirty' and 'messy' it is. It can be either trained directly with some numpy arrays, or trained indirectly with some files locate on your machine. This makes `carefree-learn` stand out from similar projects.
+ `carefree-learn` is **highly customizable** for developers. We have already wrapped (almost) every single functionality / process into a single module (a Python class), and they can be replaced or enhanced either directly from source codes or from local codes with the help of some pre-defined functions provided by `carefree-learn` (see [`Registration`](introduction.md#registration)).
+ `carefree-learn` supports easy-to-use saving and loading. By default, everything will be wrapped into a zip file!
+ `carefree-learn` supports [`Distributed Training`](introduction.md#distributed-training).

:::info
From the discriptions above, you might notice that `carefree-learn` is more of a minimal **Automatic Machine Learning** (AutoML) solution than a pure Machine Learning package.
:::

:::tip
When we say **ANY**, it means that `carefree-learn` can even train on dataset with only one single sample:

```python
import cflearn

toy = cflearn.make_toy_model()
data = toy.tr_data.converted
print(f"x={data.x}, y={data.y}")  # x=[[0.]], y=[[1.]]
```

This is especially useful when we need to do unittests or to verify whether our custom modules (e.g. custom pre-processes) are correctly integrated into `carefree-learn`, for example:

```python
import cflearn
import numpy as np

# here we implement a custom processor
@cflearn.register_processor("plus_one")
class PlusOne(cflearn.Processor):
    @property
    def input_dim(self) -> int:
        return 1

    @property
    def output_dim(self) -> int:
        return 1

    def fit(self, columns: np.ndarray) -> cflearn.Processor:
        return self

    def _process(self, columns: np.ndarray) -> np.ndarray:
        return columns + 1

    def _recover(self, processed_columns: np.ndarray) -> np.ndarray:
        return processed_columns - 1

# we need to specify that we use the custom process method to process our labels
config = {"data_config": {"label_process_method": "plus_one"}}
toy = cflearn.make_toy_model(config=config)
y = toy.tr_data.converted.y
processed_y = toy.tr_data.processed.y
print(f"y={y}, new_y={processed_y}")  # y=[[1.]], new_y=[[2.]]
```
:::

There is one more thing we'd like to mention: `carefree-learn` is '[Pandas](https://pandas.pydata.org/)-free'. The reasons why we excluded [Pandas](https://pandas.pydata.org/) are listed in [`carefree-data`](https://github.com/carefree0910/carefree-data).

## Configurations

In `carefree-learn`, we have few args and kwargs in each module. Instead, we'll use one single argument `config` which takes in a (shared) Python dict to configure those modules. That's why we can easily support JSON configuration, which is very useful when you need to share your models to others or reproduce others' work.

### Scopes

Since we have many stand-alone modules that provide corresponding functionalities, our configuration (which is a Python dict) will be designed 'hierarchically', and each module can read its specified configuration under its specific 'scope'. If needed, they can also access configurations defined under other 'scopes' easily because the whole configuration dict will be passed to each module.

:::info
Currently we mainly have 6 scopes in use: `root` (`pipeline_config`), `model_config`, `trainer_config`, `data_config`, `metric_config` and `optimizers`.
:::

Suppose we have a Python dict named `config` now:

+ The keys in `root` scope of `config` are those which are directly stored in the `config`.
+ The keys in other scopes (e.g. `data_config`) of `config` are those which are stored in a sub-dict, and this sub-dict is the value of the scope-name-key in `config` (e.g. `:::python "data_config"`).

Here is an example:

```python
config = {
    # `root` scope
    "foo": ...,
    "dummy": ...,
    # `model_config` scope
    "model_config": {
        "...": ...
    },
    # `trainer_config` scope
    "trainer_config": {
        "...": ...
    },
    ...
}
```

We will introduce what kinds of algorithm-agnostic configurations are available in each scope and how to specify them in the [`Configurations`](getting-started/configurations) section.


## Data Loading Strategy

Since `carefree-learn` focuses on tabular datasets, the data loading strategy is very different from unstructured datasets' strategy. For instance, it is quite common that a CV dataset is a bunch of pictures located in a folder, and we will either read them sequentially or read them in parallel. Nowadays, almost every famous deep learning framework has their own solution to load unstructured datasets efficiently, e.g. PyTorch officially implements [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) to support multi-process loading and other features.

Although we know that RAM speed is (almost) always faster than I/O operations, we still prefer leveraging multi-process to read files than loading them all into RAM at once. This is because unstructured datasets are often too large to allocate them all to RAM. However, when it comes to tabular datasets, we prefer to load everything into RAM at the very beginning. The main reasons are listed below:

+ Tabular datasets are often quite small and are able to put into RAM at once.
+ Network structures for tabular datasets are often much smaller, which means using multi-process loading will cause a much heavier overhead.
+ We need to take [`Distributed Training`](introduction.md#distributed-training) into account. If we stick to multi-process loading, there would be too many threads in the pool which is not a good practice.


## Distributed Training

In `carefree-learn`, **`Distributed Training`** doesn't mean training your model on multiple GPUs or multiple machines, because `carefree-learn` focuses on tabular datasets which are often not as large as unstructured datasets. Instead, **`Distributed Training`** in `carefree-learn` means **training multiple models** at the same time. This is important because:

+ Deep Learning models suffer from randomness, so we need to train multiple models with the same algorithm and calculate the mean / std of the performances to estimate the algorithm's capacity and stability.
+ Ensembling these models (trained with the same algorithm) can boost the algorithm's performance without making any changes to the algorithm itself.
+ Parameter searching will be easier & faster.


## Terminologies

In `carefree-learn`, there are some frequently used terminologies, and we will introduce them in this section. If you are confused by some other terminologies in `carefree-learn` when you are using it, feel free to edit this list:

### step

One **`step`** in the training process means that one mini-batch passed through our model.

### epochs

In most deep learning processes, training is structured into epochs. An epoch is one iteration over the entire input data, which is constructed by several **`step`**s.

### batch_size

It is a good practice to slice the data into smaller batches and iterates over these batches during training, and **`batch_size`** specifies the size of each batch. Be aware that the last batch may be smaller if the total number of samples is not divisible by the **`batch_size`**.

### config

A **`config`** indicates the main part (or, the shared part) of the configuration.

### increment_config

An **`increment_config`** indicates the configurations that you want to update on **`config`**.

:::info
This is very useful when you only want to tune a single configuration and yet you have tons of other configurations need to be fixed. In this case, you can set other configurations as 
:::**`config`**, and adjust the target configuration in **`increment_config`**.

### forward

A **`forward`** method is a common method required by (almost) all PyTorch modules.

:::info
[Here](https://discuss.pytorch.org/t/about-the-nn-module-forward/20858) is a nice discussion.
:::

### task_type

We use `:::python task_type = "clf"` to indicate a classification task, and `:::python task_type = "reg"` to indicate a regression task.

:::info
And we'll convert them into [`:::python cfdata.tabular.TaskTypes`](https://github.com/carefree0910/carefree-data/blob/770de34ed2e49ed81fa00be629d61f5b05233a9b/cfdata/tabular/types.
:::py#L91) under the hood.

### tr, cv & te

In most cases, we use:

+ `x_tr`, `x_cv` and `x_te` to represent `training`, `cross validation` and `test` **features**.
+ `y_tr`, `y_cv` and `y_te` to represent `training`, `cross validation` and `test` **labels**.

### metrics

Although `losses` are what we optimize directly during training, `metrics` are what we 'actually' want to optimize (e.g. `acc`, `auc`, `f1-score`, etc.). Sometimes we may want to take multiple `metrics` into consideration, and we may also want to eliminate the fluctuation comes with mini-batch training by applying EMA on the metrics. To make things clearer, we decided to introduce the **`metric_config`** scope (under the **`trainer_config`** scope). By default:

+ `mae` & `mse` is used for regression tasks, while `auc` & `acc` is used for classification tasks.
+ An EMA with `:::python decay = 0.1` will be used.
+ Every metrics will be treated as equal. 

So `carefree-learn` will construct the following configurations for you by default (take classification tasks as an example):

```json
{
    ...,
    "trainer_config": {
        ...,
        "metric_config": {
            "decay": 0.1,
            "types": ["auc", "acc"],
            "weights": {"auc": 1.0, "acc": 1.0}
        }
    }
}
```

It's worth mentioning that `carefree-learn` also supports using losses as metrics:

```json
{
    ...,
    "trainer_config": {
        ...,
        "metric_config": {
            "decay": 0.1,
            "types": ["loss"]
        }
    }
}
```

### optimizers

Sometimes we may want to have different optimizers to optimize different group of parameters. In order to make things easier with flexibility and control, we decided to introduce the **`optimizers`** scope (under the **`trainer_config`** scope). By default, **all** parameters will be optimized via one single optimizer, so `carefree-learn` will construct the following configurations for you by default:

```json
{
    ...,
    "trainer_config": {
        ...,
        "optimizers": {
            "all": {
                "optimizer": "adam",
                "optimizer_config": {"lr": 1e-3},
                "scheduler": "plateau",
                "scheduler_config": {"mode": "max", ...}
            }
        }
    }
}
```

If we need to apply different optimizers on different parameters (which is quite common in GANs), we need to walk through the following two steps:

+ Define a `property` in your `Model` which returns a list of parameters you want to optimize.
+ Define the corresponding optimizer configs with `property`'s name as the dictionary key.

Here's an example:

```python
from cflearn.models.base import ModelBase

@ModelBase.register("foo")
class Foo(ModelBase):
    @property
    def params1(self):
        return [self.p1, self.p2, ...]
    
    @property
    def params2(self):
        return [self.p1, self.p3, ...]
```

```json
{
    ...,
    "trainer_config": {
        ...,
        "optimizers": {
            "params1": {
                "optimizer": "adam",
                "optimizer_config": {"lr": 3e-4},
                "scheduler": null
            },
            "params2": {
                "optimizer": "nag",
                "optimizer_config": {"lr": 1e-3, "momentum": 0.9},
                "scheduler": "plateau",
                "scheduler_config": {"mode": "max", ...}
            }
        }
    }
}
```


[^1]: [**D**eep **N**eural **D**ecision **F**orests](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf)

[^2]: [**D**eep **D**istribution **R**egression](https://arxiv.org/pdf/1911.05441.pdf)