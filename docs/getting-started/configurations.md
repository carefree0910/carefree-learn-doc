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

### `DLPipeline`

Since `carefree-learn` exposed almost every parameter to users, we can actually control every part of the `Pipeline` through args and kwargs of `__init__`.

Although Machine Learning, Computer Vision and Natural Language Processing are very different, they share many things in common when they are solved by Deep Learning. Therefore in `carefree-learn`, we implement `DLPipeline` to handle these shared stuffs.

:::note
The `DLPipeline` serves as the base class of all `Pipeline`s, and for specific domain, we need to inherit `DLPipeline` and implement its own `Pipeline` class.
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

### `dl.SimplePipeline`

This `Pipeline` aims to solve general deep learning tasks.

```python
@DLPipeline.register("dl.simple")
class SimplePipeline(DLPipeline):
    def __init__(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        *,
        # The rest is the same as `DLPipeline`
```

+ **`model_name`**
    + Model that we'll use for training.
+ **`model_config`** [default = `{}`]
    + Configurations of the corresponding model.

### `dl.CarefreePipeline`

This `Pipeline` will provide some useful default settings on top of [`dl.SimplePipeline`](#dlsimplepipeline).

```python
@DLPipeline.register("dl.carefree")
class CarefreePipeline(SimplePipeline):
    def __init__(
        self,
        # The rest is the same as `dl.SimplePipeline`
```

### `cv.SimplePipeline`

This `Pipeline` is exactly the same as [`dl.SimplePipeline`](#dlsimplepipeline), just an alias.

### `cv.CarefreePipeline`

This `Pipeline` is exactly the same as [`dl.CarefreePipeline`](#dlcarefreepipeline), just an alias.

### `ml.SimplePipeline`

This `Pipeline` aims to solve tabular tasks. It will always use `MLModel` as its model, and we can only specify the `core` of the `MLModel`.

:::info
The reason why `carefree-learn` does so is that in tabular tasks, there are many common practices which shall be applied everytime, such as:
+ Encode the categorical columns (to `one_hot` / `embedding` format, required).
+ Pre-process the numerical columns (with `min_max` / `normalize` / ... method, optional).
+ Decide the binary threshold in binary classification tasks.
+ ......

In order to prevent users from re-implementing these stuffs over and over again, `carefree-learn` decides to provide `MLModel` which wraps everything up. In this case, we can focus on the core algorithms, without concerning the rest.
:::

```python
@DLPipeline.register("ml.simple")
class SimplePipeline(DLPipeline):
    def __init__(
        self,
        core_name: str = "fcnn",
        core_config: Optional[Dict[str, Any]] = None,
        *,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        loss_name: str = "auto",
        loss_config: Optional[Dict[str, Any]] = None,
        # encoder
        only_categorical: bool = False,
        encoder_config: Optional[Dict[str, Any]] = None,
        encoding_methods: Optional[Dict[str, List[str]]] = None,
        encoding_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        default_encoding_methods: Optional[List[str]] = None,
        default_encoding_configs: Optional[Dict[str, Any]] = None,
        # misc
        pre_process_batch: bool = True,
        num_repeat: Optional[int] = None,
        # The rest is the same as `DLPipeline`
```

+ **`core_name`** [default = `"fcnn"`]
    + Core of `MLModel` that we'll use for training.
+ **`core_config`** [default = `{}`]
    + Configurations of the corresponding core.
+ **`input_dim`** [default = `None`]
    + Input dimension of the task.
    + If not provided, then `cf_data` should be provided in `MLData` which we want to train on.
+ **`output_dim`** [default = `None`]
    + Output dimension of the task.
    + If not provided, then `cf_data` should be provided in `MLData` which we want to train on.
+ **`loss_name`** [default = `"auto"`]
    + Loss that we'll use for training.
    + As default (`"auto"`), `carefree-learn` will use:
        + `"mae"` for regression tasks.
        + `"focal"` for classification tasks.
+ **`loss_config`** [default = `{}`]
    + Configurations of the corresponding loss.
+ **`only_categorical`** [default = `False`]
    + Specify whether all columns in the task are categorical columns or not.
+ **`encoder_config`** [default = `{}`]
    + Configurations of `Encoder`.
+ **`encoding_methods`** [default = `None`]
    + Encoding methods we will use to encode the categorical columns.
+ **`encoding_configs`** [default = `{}`]
    + Configurations of the corresponding methods.
+ **`default_encoding_methods`** [default = `["embedding"]`]
    + Default encoding methods we will use to encode the categorical columns.
+ **`default_encoding_configs`** [default = `{}`]
    + Default configurations of the corresponding methods.
+ **`pre_process_batch`** [default = `False`]
    + Specify whether should we pre-process the input batch or not.
+ **`num_repeat`** [default = `None`]
    + In most cases this is an internal property handled by `carefree-learn` itself.

### `ml.CarefreePipeline`

This `Pipeline` will provide some useful default settings on top of [`ml.SimplePipeline`](#mlsimplepipeline).

```python
@DLPipeline.register("ml.carefree")
class CarefreePipeline(SimplePipeline):
    def __init__(
        self,
        # The rest is the same as `ml.SimplePipeline`
```

### Configure `DLZoo`

Since it will be tedious to re-define similar configurations over and over, `carefree-learn` provides `DLZoo` to improve user experience. Internally, `DLZoo` will read configurations from `cflearn/api/zoo/configs`, which contains a bunch of JSON files:

```python
# This will read the  cflearn/api/zoo/configs/clf/resnet18/default.json  file
m = cflearn.DLZoo.load_pipeline("clf/resnet18", num_classes=10)
# This will read the  cflearn/api/zoo/configs/clf/resnet18/gray.json     file
m = cflearn.DLZoo.load_pipeline("clf/resnet18.gray", num_classes=10)
```

The general usage of `DLZoo` should be as follows:

```python
m = cflearn.DLZoo.load_pipeline("task/model.type", **kwargs)
```

+ **`task`**
    + Specify the task we want to work with.
    + See [Supported Models](#supported-models) for more details.
+ **`model`**
    + Specify the model we want to use.
    + See [Supported Models](#supported-models) for more details.
+ **`type`**
    + Specify the model type we want to use.
    + If not provided, we will use `default` as the model type.
+ **`kwargs`**
    + Specify the keyword arguments of the `Pipeline`, described above.
    + See [Example](#example) section for more details.

#### `__requires__`

Although `carefree-learn` wants to make everything as easy as possible, there are still some properties that `carefree-learn` cannot make decisions for you (e.g. `img_size`, `num_classes`, etc.). These properties will be presented in the `__requires__` field of each JSON file.

For example, in `resnet18`, we will need you to provide the `num_classes` property, so the corresponding JSON file will be:

```json
{
  "__requires__": {
    "model_config": ["num_classes"]
  },
  ...
}
```

Which means we need to specify `num_classes` if we want to use `resnet18`:

```python
m = cflearn.DLZoo.load_pipeline("clf/resnet18", num_classes=10)
```

:::info
In fact, the 'original' configuration should be:

```python
m = cflearn.DLZoo.load_pipeline("clf/resnet18", model_config=dict(num_classes=10))
```

Because `num_classes` should be defined under the `model_config` scope.

Since this is quite troublesome, we decided to allow users to specify these 'requirements' directly by the names, which makes `DLZoo` more readable and easier to use!
:::

#### Example

The following two code snippets have same effects:

<Tabs
  defaultValue="scratch"
  values={[
    {label: 'From Scratch', value: 'scratch'},
    {label: 'DLZoo', value: 'zoo'},
  ]
}>

<TabItem value="scratch">

```python {3-7}
m = cflearn.cv.CarefreePipeline(
    "clf",
    model_config={
        "num_classes": 10,
        ...
    },
    loss_name="focal",
    ...
)
```

</TabItem>

<TabItem value="zoo">

```python {3-7}
m = cflearn.DLZoo.load_pipeline(
    "clf/resnet18.gray",
    model_config={
        "num_classes": 10,
        ...
    },
    loss_name="focal",
    ...
)
```

</TabItem>

<TabItem value="api">

```python
m = cflearn.api.resnet18_gray(10)
```

</TabItem>

</Tabs>

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


## Supported Models

:::info
In this section, we will:
+ Use `load` to represent `cflearn.DLZoo.load_pipeline`.
+ Use `key=...` to represent the `__requires__` field.
:::

:::tip
It's also recommended to browse the `cflearn/api/zoo/configs` folder, from which you can see all the JSON files that represent the corresponding supported models.
:::

### `clf`

#### `cct`
##### default
```python
# cflearn/api/zoo/configs/clf/cct/default.json
m = load("clf/cct", img_size=..., num_classes=...)
```
##### large
```python
# cflearn/api/zoo/configs/clf/cct/large.json
m = load("clf/cct.large", img_size=..., num_classes=...)
```
##### large_224
```python
# cflearn/api/zoo/configs/clf/cct/large_224.json
m = load("clf/cct.large_224", num_classes=...)
```
##### large_384
```python
# cflearn/api/zoo/configs/clf/cct/large_384.json
m = load("clf/cct.large_384", num_classes=...)
```
##### lite
```python
# cflearn/api/zoo/configs/clf/cct/lite.json
m = load("clf/cct.lite", img_size=..., num_classes=...)
```

#### `resnet101`
##### default
```python
# cflearn/api/zoo/configs/clf/resnet101/default.json
m = load("clf/resnet101", num_classes=...)
```

#### `resnet18`
##### default
```python
# cflearn/api/zoo/configs/clf/resnet18/default.json
m = load("clf/resnet18", num_classes=...)
```
##### gray
```python
# cflearn/api/zoo/configs/clf/resnet18/gray.json
m = load("clf/resnet18.gray", num_classes=...)
```

### `gan`

#### `siren`
##### default
```python
# cflearn/api/zoo/configs/gan/siren/default.json
m = load("gan/siren", img_size=...)
```
##### gray
```python
# cflearn/api/zoo/configs/gan/siren/gray.json
m = load("gan/siren.gray", img_size=...)
```

#### `vanilla`
##### default
```python
# cflearn/api/zoo/configs/gan/vanilla/default.json
m = load("gan/vanilla", img_size=...)
```
##### gray
```python
# cflearn/api/zoo/configs/gan/vanilla/gray.json
m = load("gan/vanilla.gray", img_size=...)
```

### `generator`

#### `cycle_gan_generator`
##### default
```python
# cflearn/api/zoo/configs/generator/cycle_gan_generator/default.json
m = load("generator/cycle_gan_generator")
```

#### `pixel_cnn`
##### default
```python
# cflearn/api/zoo/configs/generator/pixel_cnn/default.json
m = load("generator/pixel_cnn", num_classes=...)
```

#### `style_gan2_generator`
##### 1024
```python
# cflearn/api/zoo/configs/generator/style_gan2_generator/1024.json
m = load("generator/style_gan2_generator.1024")
```
##### default
```python
# cflearn/api/zoo/configs/generator/style_gan2_generator/default.json
m = load("generator/style_gan2_generator")
```
##### ffhq
```python
# cflearn/api/zoo/configs/generator/style_gan2_generator/ffhq.json
m = load("generator/style_gan2_generator.ffhq")
```
##### metfaces
```python
# cflearn/api/zoo/configs/generator/style_gan2_generator/metfaces.json
m = load("generator/style_gan2_generator.metfaces")
```

#### `vqgan_generator`
##### default
```python
# cflearn/api/zoo/configs/generator/vqgan_generator/default.json
m = load("generator/vqgan_generator")
```

### `multimodal`

#### `clip`
##### default
```python
# cflearn/api/zoo/configs/multimodal/clip/default.json
m = load("multimodal/clip")
```

#### `clip_vqgan_aligner`
##### default
```python
# cflearn/api/zoo/configs/multimodal/clip_vqgan_aligner/default.json
m = load("multimodal/clip_vqgan_aligner")
```

### `segmentor`

#### `aim`
##### default
```python
# cflearn/api/zoo/configs/segmentor/aim/default.json
m = load("segmentor/aim")
```

#### `u2net`
##### default
```python
# cflearn/api/zoo/configs/segmentor/u2net/default.json
m = load("segmentor/u2net")
```
##### finetune
```python
# cflearn/api/zoo/configs/segmentor/u2net/finetune.json
m = load("segmentor/u2net.finetune", pretrained_ckpt=...)
```
##### finetune_lite
```python
# cflearn/api/zoo/configs/segmentor/u2net/finetune_lite.json
m = load("segmentor/u2net.finetune_lite", pretrained_ckpt=...)
```
##### lite
```python
# cflearn/api/zoo/configs/segmentor/u2net/lite.json
m = load("segmentor/u2net.lite")
```
##### refine
```python
# cflearn/api/zoo/configs/segmentor/u2net/refine.json
m = load("segmentor/u2net.refine", lv1_model_ckpt_path=...)
```
##### refine_lite
```python
# cflearn/api/zoo/configs/segmentor/u2net/refine_lite.json
m = load("segmentor/u2net.refine_lite", lv1_model_ckpt_path=...)
```

### `ssl`

#### `dino`
##### default
```python
# cflearn/api/zoo/configs/ssl/dino/default.json
m = load("ssl/dino", img_size=...)
```

### `style_transfer`

#### `adain`
##### default
```python
# cflearn/api/zoo/configs/style_transfer/adain/default.json
m = load("style_transfer/adain")
```

### `vae`

#### `siren`
##### default
```python
# cflearn/api/zoo/configs/vae/siren/default.json
m = load("vae/siren", img_size=...)
```
##### gray
```python
# cflearn/api/zoo/configs/vae/siren/gray.json
m = load("vae/siren.gray", img_size=...)
```

#### `style`
##### default
```python
# cflearn/api/zoo/configs/vae/style/default.json
m = load("vae/style", img_size=...)
```
##### gray
```python
# cflearn/api/zoo/configs/vae/style/gray.json
m = load("vae/style.gray", img_size=...)
```

#### `vanilla`
##### 2d
```python
# cflearn/api/zoo/configs/vae/vanilla/2d.json
m = load("vae/vanilla.2d", img_size=...)
```
##### 2d_gray
```python
# cflearn/api/zoo/configs/vae/vanilla/2d_gray.json
m = load("vae/vanilla.2d_gray", img_size=...)
```
##### default
```python
# cflearn/api/zoo/configs/vae/vanilla/default.json
m = load("vae/vanilla", img_size=...)
```
##### gray
```python
# cflearn/api/zoo/configs/vae/vanilla/gray.json
m = load("vae/vanilla.gray", img_size=...)
```

#### `vq`
##### default
```python
# cflearn/api/zoo/configs/vae/vq/default.json
m = load("vae/vq", img_size=...)
```
##### gray
```python
# cflearn/api/zoo/configs/vae/vq/gray.json
m = load("vae/vq.gray", img_size=...)
```
##### gray_lite
```python
# cflearn/api/zoo/configs/vae/vq/gray_lite.json
m = load("vae/vq.gray_lite", img_size=...)
```
##### lite
```python
# cflearn/api/zoo/configs/vae/vq/lite.json
m = load("vae/vq.lite", img_size=...)
```
