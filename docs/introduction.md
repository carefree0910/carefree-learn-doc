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
+ `carefree-learn` can help you deal with almost **ANY** kind of tabular datasets, no matter how *dirty* and *messy* it is. It can be either trained directly with some numpy arrays, or trained indirectly with some files locate on your machine. This makes `carefree-learn` stand out from similar projects.
+ `carefree-learn` is **highly customizable** for developers. We have already wrapped (almost) every single functionality / process into a single module (a Python class), and they can be replaced or enhanced either directly from source codes or from local codes with the help of some pre-defined functions provided by `carefree-learn` (see [`Registration`](design-principles#registration)).
+ `carefree-learn` supports easy-to-use saving and loading. By default, everything will be wrapped into a zip file!
+ `carefree-learn` supports [`Distributed Training`](user-guides/distributed#distributed-training).

:::info
From the discriptions above, you might notice that `carefree-learn` is more of a minimal **Automatic Machine Learning** (AutoML) solution than a pure Machine Learning package.
:::

:::tip
When we say **ANY**, it means that `carefree-learn` can even train on one single sample.

<details><summary><b>For example</b></summary>
<p>

```python
import cflearn

toy = cflearn.make_toy_model()
data = toy.tr_data.converted
print(f"x={data.x}, y={data.y}")  # x=[[0.]], y=[[1.]]
```

</p>
</details>
<br />

This is especially useful when we need to do unittests or to verify whether our custom modules (e.g. custom pre-processes) are correctly integrated into `carefree-learn`.

<details><summary><b>For example</b></summary>
<p>

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

</p>
</details>
:::

There is one more thing we'd like to mention: `carefree-learn` is *[Pandas](https://pandas.pydata.org/)-free*. The reasons why we excluded [Pandas](https://pandas.pydata.org/) are listed in [`carefree-data`](https://github.com/carefree0910/carefree-data).


## Configurations

In `carefree-learn`, we have few args and kwargs in each module. Instead, we'll use one single argument [`Environment`](getting-started/configurations#environment) to handle the configurations. That's why we can easily support JSON configuration, which is very useful when you need to share your models to others or reproduce others' works.


## Data Loading Strategy

Since `carefree-learn` focuses on tabular datasets, the data loading strategy is very different from unstructured datasets' strategy. For instance, it is quite common that a CV dataset is a bunch of pictures located in a folder, and we will either read them sequentially or read them in parallel. Nowadays, almost every famous deep learning framework has their own solution to load unstructured datasets efficiently, e.g. PyTorch officially implements [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) to support multi-process loading and other features.

Although we know that RAM speed is (almost) always faster than I/O operations, we still prefer leveraging multi-process to read files than loading them all into RAM at once. This is because unstructured datasets are often too large to allocate them all to RAM. However, when it comes to tabular datasets, we prefer to load everything into RAM at the very beginning. The main reasons are listed below:

+ Tabular datasets are often quite small and are able to put into RAM at once.
+ Network structures for tabular datasets are often much smaller, which means using multi-process loading will cause a much heavier overhead.
+ We need to take [`Distributed Training`](user-guides/distributed#distributed-training) into account. If we stick to multi-process loading, there would be too many threads in the pool which is not a good practice.


## Terminologies

In `carefree-learn`, there are some frequently used terminologies, and we will introduce them in this section. If you are confused by some other terminologies in `carefree-learn` when you are using it, feel free to edit this list:

### step

One **`step`** in the training process means that one mini-batch passed through our model.

### epoch

In most deep learning processes, training is structured into epochs. An epoch is one iteration over the entire input data, which is constructed by several **`step`**s.

### batch_size

It is a good practice to slice the data into smaller batches and iterates over these batches during training, and **`batch_size`** specifies the size of each batch. Be aware that the last batch may be smaller if the total number of samples is not divisible by the **`batch_size`**.

### config

A **`config`** indicates the main part (or, the shared part) of the configuration.

### increment_config

An **`increment_config`** indicates the configurations that you want to update on **`config`**.

:::tip
This is very useful when you only want to tune a single configuration and yet you have tons of other configurations need to be fixed. In this case, you can set other configurations as **`config`**, and adjust the target configuration in **`increment_config`**.
:::

### forward

A **`forward`** method is a common method required by (almost) all PyTorch modules.

:::info
[Here](https://discuss.pytorch.org/t/about-the-nn-module-forward/20858) is a nice discussion.
:::

### task_type

We use `task_type = "clf"` to indicate a classification task, and `task_type = "reg"` to indicate a regression task.

:::info
And we'll convert them into [cfdata.tabular.TaskTypes](https://github.com/carefree0910/carefree-data/blob/82f158be82ced404a1f4ac37e7a669a50470b109/cfdata/tabular/misc.py#L126) under the hood.
:::

### tr, cv & te

In most cases, we use:

+ `x_tr`, `x_cv` and `x_te` to represent `training`, `cross validation` and `test` **features**.
+ `y_tr`, `y_cv` and `y_te` to represent `training`, `cross validation` and `test` **labels**.

### metrics

Although `losses` are what we optimize directly during training, `metrics` are what we *actually* want to optimize (e.g. `acc`, `auc`, `f1-score`, etc.). Sometimes we may want to take multiple `metrics` into consideration, and we may also want to eliminate the fluctuation comes with mini-batch training by applying EMA on the metrics.

:::tip
Please refer to [metrics](getting-started/configurations#metrics) and see how to customize the behaviour of `metrics` in `carefree-learn`.
:::

### optimizers

In PyTorch (and other deep learning framework) we have `optimizers` to *optimize* the parameters of our model. We sometimes need to divide the parameters into several groups and optimize them individually (which is quite common in GANs).

:::tip
Please refer to [optimizers](getting-started/configurations#optimizers) and see how to control the behaviour of `optimizers` in `carefree-learn`.
:::


[^1]: [**D**eep **N**eural **D**ecision **F**orests](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf)

[^2]: [**D**eep **D**istribution **R**egression](https://arxiv.org/pdf/1911.05441.pdf)
