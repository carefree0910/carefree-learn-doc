---
id: configurations
title: Configurations
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Although it is possible to get a rather good performance with default configurations, performance might be gained easily by specifying configurations with our prior knowledges.

We've already mentioned the basic ideas on how to configure `carefree-learn` in [`Introduction`](../#configurations), so we will focus on introducing how to actually configure `Pipeline`s in this page. 


## Specify Configurations

There are three ways to specify configurations in `carefree-learn`:
- Construct a `Pipeline` from scratch.
- Leverage `DLZoo` to construct a `Pipeline` with a JSON file.
- Utilize `cflearn.api` (**recommended!**).

Let's say we want to construct a `Pipeline` to train `resnet18` on MNIST dataset, here are three different ways to achieve this:

<Tabs
  defaultValue="scratch"
  values={[
    {label: 'From Scratch', value: 'scratch'},
    {label: 'DLZoo', value: 'zoo'},
    {label: 'cflearn.api', value: 'api'},
  ]
}>

<TabItem value="scratch">

```python
m = cflearn.cv.CarefreePipeline(
    "clf",
    {
        "in_channels": 1,
        "num_classes": 10,
        "latent_dim": 512,
        "encoder1d": "backbone",
        "encoder1d_config": {
            "name": "resnet18",
            "pretrained": False,
        },
    },
    loss_name="cross_entropy",
    metric_names="acc",
)
```

</TabItem>

<TabItem value="zoo">

```python
m = cflearn.DLZoo.load_pipeline("clf/resnet18.gray", num_classes=10)
```

</TabItem>

<TabItem value="api">

```python
m = cflearn.api.resnet18_gray(10)
```

</TabItem>

</Tabs>

We'll describe some details in the following sections.

### Configure from Scratch

Since `carefree-learn` exposed almost every parameter to users, we can actually control every part of the `Pipeline` through args and kwargs of `__init__`.

:::info
Although Machine Learning, Computer Vision and Natural Language Processing are very different, they share many things in common when they are solved by Deep Learning. Therefore in `carefree-learn`, we implement `DLPipeline` to handle these shared stuffs.
:::

```python
class DLPipeline(PipelineProtocol, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        loss_name: str,
        loss_config: Optional[Dict[str, Any]] = None,
        # trainer
        state_config: Optional[Dict[str, Any]] = None,
        num_epoch: int = 40,
        max_epoch: int = 1000,
        fixed_epoch: Optional[int] = None,
        fixed_steps: Optional[int] = None,
        log_steps: Optional[int] = None,
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        cudnn_benchmark: bool = False,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: Optional[Dict[str, Any]] = None,
        use_losses_as_metrics: Optional[bool] = None,
        loss_metrics_weights: Optional[Dict[str, float]] = None,
        recompute_train_losses_in_eval: bool = True,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
        optimizer_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        workplace: str = "_logs",
        finetune_config: Optional[Dict[str, Any]] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
    )
```

+ **`loss_name`**
    + Loss that we'll use for training.
    + Currently `carefree-learn` supports
        + Common losses: `mae`, `mse`, `quantile`, `cross_entropy`, `focal`, ...
        + Task specific losses: `vae`, `vqvae`, ...
+ **`loss_config`** [default = `{}`]
    + Configurations of the corresponding loss.
+ **`state_config`** [default = `{}`]
    + Configurations of the [`TrainerState`](#trainerstate).
+ **`num_epoch`** [default = `40`]
    + Specify number of epochs. 
    + Notice that in most cases this will not be the final epoch number.
+ **`max_epoch`** [default = `1000`]
    + Specify the maximum number of epochs.
+ **`fixed_epoch`** [default = `None`]
    + Specify the (fixed) number of epochs.
    + If sepcified, then `num_epoch` and `max_epoch` will all be set to it.
+ **`fixed_steps`** [default = `None`]
    + Specify the (fixed) number of steps.
+ **`log_steps`** [default = `None`]
    + Specify the (fixed) number of steps to do loggings.
+ **`valid_portion`** [default = `1.0`]
    + Specify how much data from validation set do we want to use for monitoring.
+ **`amp`** [default = `False`]
    + Specify whether use the [`amp`](https://pytorch.org/docs/stable/amp.html) technique or not.
+ **`clip_norm`** [default = `0.0`]
    + Given a gradient `g`, and the **`clip_norm`** value, we will normalize `g` so that its L2-norm is less than or equal to **`clip_norm`**.
    + If `0.0`, then no gradient clip will be performed.
+ **`cudnn_benchmark`** [default = `False`]
    + Specify whether use the [`cudnn.benchmark`](https://pytorch.org/docs/stable/backends.html) technique or not.
+ **`metric_names`** [default = `None`]
    + Specify what metrics do we want to use for monitoring.
    + If `None`, then no metrics will be used, and losses will be treated as metrics.
+ **`metric_configs`** [default = `{}`]
    + Configurations of the corresponding metrics.
+ **`use_losses_as_metrics`** [default = `None`]
    + Specify whether use losses as metrics or not.
    + It will always be `True` if `metric_names` is `None`.
+ **`loss_metrics_weights`** [default = `None`]
    + Specify the weight of each loss when they are used as metrics.
+ **`recompute_train_losses_in_eval`** [default = `True`]
    + Specify whether should we recompute losses on training set in monitor steps when validation set is not provided.
+ **`monitor_names`** [default = `None`]
    + Specify what monitors do we want to use for monitoring.
    + If `None`, then [`BasicMonitor`](#basicmonitor) will be used.
+ **`monitor_configs`** [default = `{}`]
    + Configurations of the corresponding monitors.
+ **`callback_names`** [default = `None`]
    + Specify what callbacks do we want to use during training.
    + If `None`, then [`_LogMetricsMsgCallback`](#_logmetricsmsgcallback) will be used.
+ **`callback_configs`** [default = `{}`]
    + Configurations of the corresponding callbacks.
+ **`lr`** [default = `None`]
    + Default learning rate.
    + If not specified, `carefree-learn` will try to infer the best default value.
+ **`optimizer_name`** [default = `"None"`]
    + Specify which optimizer will be used.
    + If not specified, `carefree-learn` will try to infer the best default value.
+ **`scheduler_name`** [default = `"None"`]
    + Specify which learning rate scheduler will be used.
    + If not specified, `carefree-learn` will try to infer the best default value.
+ **`optimizer_config`** [default = `{}`]
    + Specify the optimizer's configuration.
+ **`scheduler_config`** [default = `{}`]
    + Specify the scheduler's configuration.
+ **`optimizer_settings`** [default = `None`]
    + Specify the fine grained configurations of optimizers and schedulers.
    + We should not specify `optimizer_name`, ... if we want to specify `optimizer_settings`.
    + See [`OptimizerPack`](#optimizerpack) for more details.
+ **`workplace`** [default = `"_logs"`]
    + Specify the workplace of the whole training process.
    + In general, `carefree-learn` will create a folder (with timestamp as its name) in the workplace, and will dump everything generated in the training process to it.
+ **`finetune_config`** [default = `None`]
    + Specify the finetune configurations.
    + If `None`, then we'll not utilize the finetune mechanism supported by `carefree-learn`.
    + See [`finetune_config`](#finetune_config) for more details.
+ **`tqdm_settings`** [default = `None`]
    + Specify the `tqdm` configurations.
    + See [`TqdmSettings`](#tqdmsettings) for more details.
+ **`in_loading`** [default = `False`]
    + In most cases this is an internal property handled by `carefree-learn` itself.

### Configure `DLZoo`

Since it will be tedious to re-define similar configurations over and over, `carefree-learn` provides `DLZoo` to improve user experience. Internally, `DLZoo` will read configurations from `cflearn/api/zoo/configs`, which contains a bunch of JSON files:

```python
# This will read the  cflearn/api/zoo/configs/clf/resnet18/default.json  file
m = cflearn.DLZoo.load_pipeline("clf/resnet18", num_classes=10)
# This will read the  cflearn/api/zoo/configs/clf/resnet18/gray.json     file
m = cflearn.DLZoo.load_pipeline("clf/resnet18.gray", num_classes=10)
```

### Configure `cflearn.api`

Since `DLZoo` mainly depends on JSON files which cannot provide useful auto-completion, `carefree-learn` further provides `cflearn.api`, which is a thin wrapper of `DLZoo`, as the recommended user interface.

Configuring `cflearn.api` will be exactly the same as configuring `DLZoo`, except that it can utilize auto-completion which significantly improves user experience.

```python
m = cflearn.api.resnet18_gray(10, metric_names="acc")
```


## Configuration Details

### `make_multiple` mechanism

> This mechanism is based on the [`Register Mechanism`](../design-principles#register-mechanism).

`make_multiple` mechanism is useful when we need to use either one single instance or multiple instances (e.g. use one metric / use multiple metrics to monitor the training process):
+ When we need one single instance, only one single name (`str`) and the corresponding config is required.
+ When we need multiple instances, their names (`List[str]`) are required, and the configs should be a dictionary, where:
    + The keys should be the names.
    + The values should be the corresponding configs.

The source codes well demonstrate how it works:

```python
@classmethod
def make_multiple(
    cls,
    names: Union[str, List[str]],
    configs: configs_type = None,
) -> Union[T, List[T]]:
    if configs is None:
        configs = {}
    if isinstance(names, str):
        assert isinstance(configs, dict)
        return cls.make(names, configs)  # type: ignore
    if not isinstance(configs, list):
        configs = [configs.get(name, {}) for name in names]
    return [
        cls.make(name, shallow_copy_dict(config))
        for name, config in zip(names, configs)
    ]
```

### `TrainerState`

```python
class TrainerState:
    def __init__(
        self,
        loader: DataLoaderProtocol,
        *,
        num_epoch: int,
        max_epoch: int,
        fixed_steps: Optional[int] = None,
        extension: int = 5,
        enable_logging: bool = True,
        min_num_sample: int = 3000,
        snapshot_start_step: Optional[int] = None,
        max_snapshot_file: int = 5,
        num_snapshot_per_epoch: int = 2,
        num_step_per_log: int = 350,
        num_step_per_snapshot: Optional[int] = None,
        max_step_per_snapshot: int = 1000,
    )
```

+ **`loader`**
    + This will be handled by `carefree-learn` internally.
+ **`num_epoch`**
    + Specify number of epochs. 
    + Notice that in most cases this will not be the final epoch number.
+ **`max_epoch`**
    + Specify the maximum number of epochs.
+ **`fixed_steps`** [default = `None`]
    + Specify the (fixed) number of steps.
+ **`extension`** [default = `None`]
    + Specify the number of the extended epochs per extension.
    + So basically, we'll not extend the epoch for more than $$\frac{\mathrm{max\_epoch}-\mathrm{num\_epoch}}{\mathrm{extension}}$$ times.
+ **`enable_logging`** [default = `True`]
    + Whether enable logging stuffs or not.
+ **`min_num_sample`** [default = `3000`]
    + We'll not start monitoring until the model has already seen `min_num_sample` samples.
    + This can avoid monitors from stopping too early, when the model is still trying to optimize its initial state.
+ **`snapshot_start_step`** [default = `None`]
    + Specify the number of steps when we start to take snapshots.
    + If not specified, `carefree-learn` will try to infer the best default value.
+ **`max_snapshot_file`** [default = `5`]
    + Specify the maximum number of checkpoint files we could save during training.
+ **`num_snapshot_per_epoch`** [default = `2`]
    + Indicates how many snapshots we would like to take per epoch.
    + The final behaviour will be affected by `max_step_per_snapshot`.
+ **`num_step_per_log`** [default = `350`]
    + Indicates the number of steps of each logging period.
+ **`num_step_per_snapshot`** [default = `None`]
    + Specify the number of steps of each snapshot period.
    + If not specified, `carefree-learn` will try to infer the best default value.
+ **`max_step_per_snapshot`** [default = `1000`]
    + Specify the maximum number of steps of each snapshot period.

### `BasicMonitor`

This is the default monitor of `carefree-learn`. It's fairly simple, but quite useful in practice:
+ It will take a snapshot when SOTA is achieved.
+ It will terminate the training after `patience` steps, if the new score is even worse than the worst score.
+ It will not punish extension

:::info
So in most cases, `BasicMonitor` will not early-stop until `max_epoch` is reached.
:::

```python
@TrainerMonitor.register("basic")
class BasicMonitor(TrainerMonitor):
    def __init__(self, patience: int = 25):
        super().__init__()
        self.patience = patience
        self.num_snapshot = 0
        self.best_score = -math.inf
        self.worst_score: Optional[float] = None

    def snapshot(self, new_score: float) -> bool:
        self.num_snapshot += 1
        if self.worst_score is None:
            self.worst_score = new_score
        else:
            self.worst_score = min(new_score, self.worst_score)
        if new_score > self.best_score:
            self.best_score = new_score
            return True
        return False

    def check_terminate(self, new_score: float) -> bool:
        if self.num_snapshot <= self.patience:
            return False
        if self.worst_score is None:
            return False
        return new_score <= self.worst_score

    def punish_extension(self) -> None:
        return None
```

### `_LogMetricsMsgCallback`

This is the default callback of `carefree-learn`. It will report the validation metrics to the console periodically, along with the current steps / epochs, and the execution time since last report. It will also write these information to disk.

:::info
When writing to disk, `_LogMetricsMsgCallback` will also write the `lr` (learning rate) of the corresponding steps.
:::

### `OptimizerPack`

```python
class OptimizerPack(NamedTuple):
    scope: str
    optimizer_name: str
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
```

+ **`scope`**
    + Specify the parameter 'scope' of this pack.
    + If `scope="all"`, all trainable parameters will be considered.
    + Else, it represents the attribute of the model, and:
        + If this attribute is an `nn.Module`, then its parameters will be considered.
        + Else, this attribute should be a list of parameters, which will be considered.
+ **`optimizer_name`**
    + Specify which optimizer will be used.
+ **`scheduler_name`** [default = `"None"`]
    + Specify which learning rate scheduler will be used.
    + If not specified, no scheduler will be used.
+ **`optimizer_config`** [default = `{}`]
    + Specify optimizer's configuration.
+ **`scheduler_config`** [default = `{}`]
    + Specify scheduler's configuration.

Since directly constructing `OptimizerPack`s will be troublesome, `carefree-learn` provides many convenient interface for users to specify optimizer settings. For instance, these configurations will have same effects:

<Tabs
  defaultValue="kwargs"
  values={[
    {label: 'Via `kwargs`', value: 'kwargs'},
    {label: 'Via `optimizer_settings`', value: 'settings'},
  ]
}>

<TabItem value="kwargs">

```python
m = cflearn.cv.CarefreePipeline(
    ...,
    lr=1.0e-3,
    optimizer_name="adamw",
    scheduler_name="plateau",
    optimizer_config={"weight_decay": 1.0e-3},
)
```

</TabItem>

<TabItem value="settings">

```python
m = cflearn.cv.CarefreePipeline(
    ...,
    optimizer_settings={
        "all": dict(
            optimizer_name="adamw",
            scheduler_name="plateau",
            optimizer_config={"lr": 1.0e-3, "weight_decay": 1.0e-3},
        ),
    },
)
```

</TabItem>

</Tabs>

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

```python
m = cflearn.cv.CarefreePipeline(
    ...,
    optimizer_settings={
        "params1": {
            "optimizer": "adam",
            "optimizer_config": {"lr": 3.0e-4},
            "scheduler": None,
        },
        "params2": {
            "optimizer": "nag",
            "optimizer_config": {"lr": 1.0e-3, "momentum": 0.9},
            "scheduler": "plateau",
            "scheduler_config": {"mode": "max", ...},
        },
    },
)
```

### `finetune_config`

> Source code: [`_init_finetune`](https://github.com/carefree0910/carefree-learn/blob/d039183c803f23266101b65c3863528e97940bc8/cflearn/trainer.py#L435).

`carefree-learn` supports finetune mechanism, and we can specify:
+ The initial states we want to start training from.
+ What parameters should we freeze / train during the finetune process, and Regex is supported!

#### Example

```python
m = cflearn.api.u2net(
    ...,
    finetune_config={
        "pretrained_ckpt": "/path/to/your/pretrained.pt",
        # We'll freeze the parameters whose name follows the regex expression
        "freeze": "some.regex.expression",
        # We'll freeze the parameters whose name doesn't follow the regex expression
        "freeze_except": "some.regex.expression",
    },
)
```

:::info
`freeze` & `freeze_except` should not be provided simultaneously
:::

### `TqdmSettings`

```python
class TqdmSettings(NamedTuple):
    use_tqdm: bool = False
    use_step_tqdm: bool = False
    use_tqdm_in_validation: bool = False
    in_distributed: bool = False
    position: int = 0
    desc: str = "epoch"
```

+ **`use_tqdm`** [default = `False`]
    + Whether enable `tqdm` progress bar or not.
+ **`use_step_tqdm`** [default = `False`]
    + Whether enable `tqdm` progress bar on steps or not.
+ **`use_tqdm_in_validation`** [default = `False`]
    + Whether enable `tqdm` progress bar in validation procedure or not.
+ **`in_distributed`** [default = `False`]
    + This will be handled by `carefree-learn` internally.
+ **`position`** [default = `0`]
    + This will be handled by `carefree-learn` internally.
+ **`desc`** [default = `"epoch"`]
    + This will be handled by `carefree-learn` internally.
