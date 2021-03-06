---
id: configurations
title: Configurations
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Although it is possible to get a rather good performance with default configurations, performance might be gained easily by specifying configurations with our prior knowledges.

We've already mentioned the basic ideas on how to configure `carefree-learn` in [`Introduction`](../#configurations), so we will focus on introducing how to actually configure `Environment` in this page. 


## Environment

In many high-level Machine Learning modules (e.g. [scikit-learn](https://scikit-learn.org/stable/)), configurations are directly specified by using args and kwargs to instantiate an object of the corresponding algorithm. In `carefree-learn`, however, since we've wrapped many procedures together (in [`Pipeline`](../design-principles#pipeline)) to provide a more *carefree* usage, we cannot put all those configurations in the definition of the class because that will be too much and too messy. Instead, we will share an `Environment` instance across different components to specify their configurations.

There are several advantages by doing so, as listed below:

+ It's much more flexible and easier to extend.
+ It's much easier to reproduce other's work, because a single JSON file will be enough.
+ It's much easier to share configurations between different modules. This is especially helpful in `carefree-learn` because we've tried hard to do elegant abstractions, which lead us to implement many individual modules to handle different problems. In this case, some *global* information will be hard to access if we don't share configurations.

:::info
In most cases we won't instantiate an `Environment` instance explicitly, but will leverage [High Level APIs](#high-level-apis) (e.g. [Elements](#elements) & [cflearn.make](#make)) to make things easier and clearer.
:::


## Specify Configurations

There are two ways to specify configurations in `carefree-learn`: directly with a Python dict or indirectly with a JSON file.

<Tabs
  defaultValue="dict"
  values={[
    {label: 'Python dict', value: 'dict'},
    {label: 'JSON file', value: 'json'},
  ]
}>
<TabItem value="dict">

```python
import cflearn

# specify any configurations
config = {"foo": 0, "dummy": 1}
fcnn = cflearn.make(**config)

assert fcnn.foo == 0
assert fcnn.dummy == 1
```

</TabItem>
<TabItem value="json">

In order to use a JSON file as configuration, suppose you want to run `my_script.py`, and it contains the following codes:

```python
import cflearn

config = "./configs/basic.json"
increment_config = {"foo": 2}
fcnn = cflearn.make(config=config, increment_config=increment_config)
```

Since `config` is set to `"./configs/basic.json"`, the file structure should be:

```text
-- my_script.py
-- configs
 |-- basic.json
```

Suppose `basic.json` contains following stuffs:

```json
{
    "foo": 0,
    "dummy": 1
}
```

Then we should have:

```python
assert fcnn.foo == 2
assert fcnn.dummy == 1
```

It is OK to get rid of `increment_config`, in which case the configuration will be completely controlled by `basic.json`:

```python
import cflearn

config = "./configs/basic.json"
fcnn = cflearn.make(config=config)

assert fcnn.foo == 0
assert fcnn.dummy == 1
```

</TabItem>
</Tabs>


## High Level APIs

In order to manage default configurations, `carefree-learn` introduced `Elements`, which is a `NamedTuple`, to organize the logics. With the help of `Elements`, defining high-level APIs could be fairly easy and straight forward.

:::info
`Elements` is more of an internal structure used by [make](#make) API than an external API, but we can also use it directly, as shown in the [Example](#example) section.
:::

### Elements

Since some fields in `Elements` need to be inferenced with other information, their `default` values are ones assigned in `Elements.make`.

```python
class Elements(NamedTuple):
    model: str = "fcnn"
    aggregator: str = "sum"
    aggregator_config: Optional[Dict[str, Any]] = None
    data_config: Optional[Dict[str, Any]] = None
    task_type: Optional[task_type_type] = None
    use_simplify_data: bool = False
    ts_config: Optional[TimeSeriesConfig] = None
    aggregation: Optional[str] = None
    aggregation_config: Optional[Dict[str, Any]] = None
    ts_label_collator_config: Optional[Dict[str, Any]] = None
    delim: Optional[str] = None
    has_column_names: Optional[bool] = None
    read_config: Optional[Dict[str, Any]] = None
    logging_folder: Optional[str] = None
    logging_file: Optional[str] = None
    batch_size: int = 128
    cv_split: Optional[Union[float, int]] = None
    use_amp: bool = False
    min_epoch: Optional[int] = None
    num_epoch: Optional[int] = None
    max_epoch: Optional[int] = None
    fixed_epoch: Optional[int] = None
    max_snapshot_file: int = 5
    clip_norm: float = 0.0
    ema_decay: float = 0.0
    model_config: Optional[Dict[str, Any]] = None
    loss: str = "auto"
    loss_config: Optional[Dict[str, Any]] = None
    default_encoding_init_method: Optional[str] = None
    metrics: Union[str, List[str]] = "auto"
    metric_config: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None
    optimizer: Optional[str] = "adamw"
    scheduler: Optional[str] = "warmup"
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    optimizers: Optional[Dict[str, Any]] = None
    verbose_level: int = 2
    use_tqdm: bool = True
    trigger_logging: bool = False
    cuda: Optional[Union[int, str]] = None
    mlflow_config: Optional[Dict[str, Any]] = None
    extra_config: Optional[Dict[str, Any]] = None
```

+ **`model`** [default = `"fcnn"`]
    + Specify which model we're going to use.
    + Currently `carefree-learn` supports:
        + `"linear"`, `"fcnn"`, `"wnd"`, `"nnb"`, `"ndt"`, `"tree_linear"`, `"tree_stack"`, `"tree_dnn"` and `"ddr"` for basic usages.
        + `"rnn"` and `"transformer"` for time series usages.
+ **`aggregator`** [default = `"sum"`]
    + Specify the [`Aggregator`](../design-principles#aggregator) we're going to use.
+ **`aggregator_config`** [default = `{}`]
    + Specify the config for the [`Aggregator`](../design-principles#aggregator).
+ **`data_config`** [default = `{}`]
    + kwargs used in [`cfdata.tabular.TabularData`](https://github.com/carefree0910/carefree-data/blob/82f158be82ced404a1f4ac37e7a669a50470b109/cfdata/tabular/wrapper.py#L31).
+ **`task_type`** [default = `None`]
    + Specify the task type.
        + If not provided, `carefree-learn` will try to inference it with the help of `carefree-data`.
+ **`use_simplify_data`** [default = `False`]
    + Specify whether use a simplified `TabularData` (without any pre-processing).
+ **`ts_config`** [default = `None`]
    + Specify the time series config (experimental).
+ **`aggregation`** [default = `None`]
    + Specify the aggregation used in time series tasks (experimental).
+ **`aggregation_config`** [default = `None`]
    + Specify the configuration of aggregation used in time series tasks (experimental).
+ **`ts_label_collator_config`** [default = `None`]
    + Specify the configuration of the label collator used in time series tasks (experimental).
+ **`delim`** [default = `None`]
    + Specify the delimiter of the dataset file.
        + If not provided, `carefree-learn` will try to inference it with the help of `carefree-data`.
    + Only take effects when we are using file datasets.
+ **`has_column_names`** [default = `None`]
    + Specify whether the elements of the first row are column names.
        + If not provided, `carefree-learn` will try to inference it with the help of `carefree-data`.
    + Only take effects when we are using file datasets.
+ **`read_config`** [default = `{}`]
    + kwargs used in [`cfdata.tabular.TabularData.read`](https://github.com/carefree0910/carefree-data/blob/82f158be82ced404a1f4ac37e7a669a50470b109/cfdata/tabular/wrapper.py#L769).
+ **`logging_folder`** [default = `None`]
    + Specify the logging folder.
    + If not provided, `carefree-learn` will try to inference it automatically.
+ **`logging_file`** [default = `None`]
    + Specify the logging file.
    + If not provided, `carefree-learn` will try to inference it automatically.
+ **`batch_size`** [default = `128`]
    + Specify the number of samples in each batch.
+ **`cv_split`** [default = `None`]
    + Specify the split of the cross validation dataset.
        + If `cv_split < 1`, it will be the 'ratio' comparing to the whole dataset.
        + If `cv_split > 1`, it will be the exact 'size'.
        + If `cv_split == 1`, `cv_split == "ratio" if isinstance(cv_split, float) else "size"`
    + If not provided, `carefree-learn` will try to inference it with `min_cv_split`, `max_cv_split` and `max_cv_split_ratio`. See [cv_split](#cv_split) for more details.
+ **`use_amp`** [default = `False`]
    + Specify whether use the [`amp`](https://pytorch.org/docs/stable/amp.html) technique or not.
+ **`min_epoch`** [default = `0`]
    + Specify the minimum number of epoch.
+ **`num_epoch`** [default = `40`]
    + Specify number of epoch. 
    + Notice that in most cases this will not be the final epoch number.
+ **`max_epoch`** [default = `200`]
    + Specify the maximum number of epoch.
+ **`fixed_epoch`** [default = `None`]
    + Specify the (fixed) number of epoch.
    + If sepcified, then `min_epoch`, `num_epoch` and `max_epoch` will all be set to it.
+ **`max_snapshot_file`** [default = `5`]
    + Specify the maximum number of checkpoint files we could save during training.
+ **`clip_norm`** [default = `0.0`]
    + Given a gradient `g`, and the **`clip_norm`** value, we will normalize `g` so that its L2-norm is less than or equal to **`clip_norm`**.
    + If `0.0`, then no gradient clip will be performed.
+ **`ema_decay`** [default = `0.0`]
    + When training a model, it is often beneficial to maintain **E**xponential **M**oving **A**verages with a certain decay rate (**`ema_decay`**) of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.
    + If `0.0`, then no EMA will be used.
+ **`model_config`** [default = `{}`]
    + Configurations used in [`Model`](../design-principles#model).
+ **`loss`** [default = `"auto"`]
    + Loss that we'll use for training.
    + Currently `carefree-learn` supports `mae`, `mse`, `quantile`, `cross_entropy`, `label_smooth_cross_entropy` and `focal`.
+ **`loss_config`** [default = `None`]
    + Configurations of the corresponding loss.
+ **`default_encoding_init_method`** [default = `None`]
    + Default init method for encoding.
+ **`metrics`** [default = `"auto"`]
    + Specify which metric(s) are we going to use to monitor our training process
+ **`metric_config`** [default = `{}`]
    + Specify the fine grained configurations of metrics. See [`metrics`](#metrics) for more details.
+ **`lr`** [default = `None`]
    + Default learning rate.
    + If not specified, `carefree-learn` will try to infer the best default value.
+ **`optimizer`** [default = `"adam"`]
    + Specify which optimizer will be used.
+ **`scheduler`** [default = `"plateau"`]
    + Specify which learning rate scheduler will be used.
+ **`optimizer_config`** [default = `{}`]
    + Specify optimizer's configuration.
+ **`scheduler_config`** [default = `{}`]
    + Specify scheduler's configuration.
+ **`optimizers`** [default = `{}`]
    + Specify the fine grained configurations of optimizers and schedulers. See [`optimizers`](#optimizers) for more details.
+ **`verbose_level`** [default = `2`]
    + Specify the verbose level.
+ **`use_tqdm`** [default = `True`]
    + Whether utilize the `tqdm` progress bar or not.
+ **`trigger_logging`** [default = `False`]
    + Whether log messages into a log file.
+ **`cuda`** [default = `None`]
    + Specify the working GPU.
    + If not provided, `carefree-learn` will try to inference it automatically.
+ **`mlflow_config`** [default = `None`]
    + Specify the configuration of `mlflow`.
    + If `None`, then we will not utilize `mlflow`.
+ **`extra_config`** [default = `{}`]
    + Other configurations.

### make

In order to provide out of the box tools, `carefree-learn` implements high level APIs for training, evaluating, distributed, HPO, etc. In this section we'll introduce `cflearn.make` because [other APIs](../user-guides/apis#cflearnapibasic) depend on it more or less.

```python
def make(
    model: str = "fcnn",
    config: general_config_type = None,
    increment_config: general_config_type = None,
    **kwargs: Any,
) -> Pipeline:
    kwargs["model"] = model
    parsed_config = update_dict(_parse_config(config), kwargs)
    parsed_increment_config = _parse_config(increment_config)
    return Pipeline.make(parsed_config, parsed_increment_config)
```

Notice that we've used `Pipeline.make`, whose definition is:

```python
@classmethod
def make(
    cls,
    config: Dict[str, Any],
    increment_config: Dict[str, Any],
) -> "Pipeline":
    return cls(Environment.from_elements(Elements.make(config, increment_config)))
```

The reason why we introduce the `make` API is that we hope we can provide APIs as *carefree* as possible. So we've listed the most commonly used configurations in `Elements`, and supported users to specify them through `kwargs` (which is the most intuitive way) in `make`.

:::info
The correspondence between `make` and `Elements` is illustrated in the [Example](#example) section.
:::

For example, if we want to switch our device to `"cpu"`, we can simply:

```python
import cflearn

m = cflearn.make(cuda="cpu")
print(m.device)  # cpu
```

### Example

As mentioned above, in most cases we can simply use the [`make`](#make) API, but we can also use [`Elements`](#elements) directly. For instance, the following snippet

```python
import cflearn

m = cflearn.make(cuda="cpu")
print(m.device)  # cpu
```

is equivalent to

```python
from cflearn.configs import *
from cflearn.pipeline import Pipeline

elements = Elements(cuda="cpu")
environment = Environment.from_elements(elements)
m = Pipeline(environment)
print(m.device)  # cpu
```


## Configuration Details

:::caution
This section is a work in progress.
:::

In this section we'll introduce some default configurations used in `carefree-learn`, as well as how to configure them (with some examples). The default settings have already been tuned on variety tabular datasets and should be able to achieve a good performance, as mentioned at the beginning.

### metrics

By default:
+ `mae` & `mse` is used for regression tasks, while `auc` & `acc` is used for classification tasks.
+ An EMA with `decay = 0.1` will be used.
+ Every metric will be treated as equal. 

So `carefree-learn` will construct the following configurations for you (take classification tasks as an example):

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

By default, **all** parameters will be optimized via one single optimizer, so `carefree-learn` will construct the following configurations for you:

```json
{
    ...,
    "trainer_config": {
        ...,
        "optimizers": {
            "all": {
                "optimizer": "adamw",
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
import cflearn

@cflearn.register_model("foo")
class Foo(cflearn.ModelBase):
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

### cv_split

It is important to split out a cross validation dataset from the training dataset if it is not explicitly provided, because a cv set could help us monitor the generalization error, hence prevent overfitting. However, unlike unstructured datasets, the sample number of tabular datasets could vary dramatically (roughly $10^2$ ~ $10^8$). Therefore, it is not trivial to decide how many samples should we use for cross validation. In `carefree-learn`, we use `min_cv_split`, `max_cv_split` and `max_cv_split_ratio` to help us make this decision automatically:

```python
default_cv_split = 0.1
cv_split_num = int(round(default_cv_split * num_data))
cv_split_num = max(self.min_cv_split, cv_split_num)
max_cv_split = int(round(num_data * self.max_cv_split_ratio))
max_cv_split = min(self.max_cv_split, max_cv_split)
return min(cv_split_num, max_cv_split)
```

:::note default settings
+ `min_cv_split`: 100
+ `max_cv_split`: 10000
+ `max_cv_split_ratio`: 0.5
:::
