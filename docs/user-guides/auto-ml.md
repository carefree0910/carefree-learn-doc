---
id: auto-ml
title: AutoML
---

`carefree-learn` provides `cflearn.Auto` API for out-of-the-box usages.

```python
import cflearn

from cfdata.tabular import *

# prepare iris dataset
iris = TabularDataset.iris()
iris = TabularData.from_dataset(iris)
# split 10% of the data as validation data
split = iris.split(0.1)
train, valid = split.remained, split.split
x_tr, y_tr = train.processed.xy
x_cv, y_cv = valid.processed.xy
data = x_tr, y_tr, x_cv, y_cv


if __name__ == '__main__':
    # standard usage
    fcnn = cflearn.make().fit(*data)

    # 'overfit' validation set
    # * `clf` indicates this is a classification task
    # * for regression tasks, use `reg` instead
    auto = cflearn.Auto("clf").fit(*data, num_jobs=2)

    # evaluate manually
    predictions = auto.predict(x_cv)
    print("accuracy:", (y_cv == predictions).mean())

    # evaluate with `cflearn`
    cflearn.evaluate(
        x_cv,
        y_cv,
        pipelines=fcnn,
        other_patterns={"auto": auto.pattern},
    )
```

Which yields

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          auto          | -- 1.000000 -- | -- 0.000000 -- | -- 1.000000 -- | -- 1.000000 -- | -- 0.000000 -- | -- 1.000000 -- |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.933333    | -- 0.000000 -- |    0.933333    |    0.993333    | -- 0.000000 -- |    0.993333    |
================================================================================================================================
```


## Explained

`cflearn.Auto.fit` will run through the following steps:
1. define the model space automatically (or manually; see [Define Model Space](#define-model-space) for more details).
2. fetch hyperparameters search spaces automatically (or manually; see [Define Search Spaces](#define-search-spaces) for more details) for each model with the help of `OptunaPresetParams` (and inject manual configurations, if provided; see [Define Extra Configurations](#define-extra-configurations) for more details).
3. leverage `optuna` with `cflearn.optuna_tune` to perform hyperparameters optimization.
4. use searched hyperparameters to train each model multiple times (separately).
5. ensemble all trained models (with `cflearn.Ensemble.stacking`).
6. record all these results to corresponding attributes.

So after `cflearn.Auto.fit`, we can perform visualizations provided by `optuna` easily:

```python
export_folder = "iris_vis"
auto.plot_param_importances("fcnn", export_folder=export_folder)
auto.plot_intermediate_values("fcnn", export_folder=export_folder)
```

:::note
It is also worth mentioning that we can pass file datasets into `cflearn.Auto` as well. See [test_auto_file](https://github.com/carefree0910/carefree-learn/blob/3fb03dbfc3e2b494f2ab03b9d8f07683fe30e7ef/tests/usages/test_basic.py#L221) for more details.
:::


## Configurations

Although `cflearn.Auto` could achieve acceptable performances, we can manually adjust its behaviour for even better ones as well.

### Define Model Space

Model space could be defined by specifying the `models` in `cflearn.Auto`:

```python
auto = cflearn.Auto(..., models="fcnn")
```

or

```python
auto = cflearn.Auto(..., models=["linear", "fcnn"])
```

:::info
By default, `cflearn.Auto` will use a large model space and hope for the best:

```python
if models == "auto":
    models = ["linear", "fcnn", "tree_dnn"]
    parsed_task_type = parse_task_type(task_type)
    # time series tasks
    if parsed_task_type.is_ts:
        models += ["rnn", "transformer"]
    # classification tasks
    elif parsed_task_type.is_clf:
        models += ["nnb", "ndt"]
    # regression tasks
    else:
        models.append("ddr")
```

We recommend to use `models="fcnn"` before actually dive into this bunch of models 🤣
:::

### Define Search Spaces

Search spaces could be defined by specifying the `params` in `fit`:

```python
auto.fit(..., params={...})
```

However customizing `params` requires some more steps, and we'll illustrate how to do so in the following sub-sections.

#### `OptunaParam`

As shown in the [`make`](../getting-started/configurations#make) API, we can specify configurations through `kwargs`. Customizing `params` is actually no more than customizing this `kwargs`, except it should turn the target hyperparameter from a specific value to an `OptunaParam`.

:::info
+ If you are more interested in codes than a step by step tutorial, you can jump to the [In a Nut Shell](#in-a-nut-shell) section directly.
+ If you are more interested in the API documentations, you can jump to the [APIs](#apis) section directly.
:::

For example, if we want to use `sgd` instead of the default `adamw`, we can simply

```python
import cflearn
from cfdata.tabular import TabularDataset

# We'll use the famous iris dataset
x, y = TabularDataset.iris().xy
m = cflearn.make(optimizer="sgd").fit(x, y)
print(m.trainer.optimizers["all"])  # SGD(...)
```

but we are not sure which one is better, and here's where AutoML could help. Since we should choose from either `sgd` or `adamw`, the search space is pretty simple:

```python
optimizer_param = cflearn.OptunaParam(
    "opt",             # this should be the unique identifier of this search space
    ["sgd", "adamw"],  # here are the parameters of this search space
    "categorical",     # this is the type of this search space
)
params = {
    # since different model may require different search space
    # we should specify which model does this search space belong to
    # * the "optimizer" argument here should correspond to the one in `make`
    "linear": {"optimizer": optimizer_param},
}
```

After which we can perform AutoML on this search space:

```python
# notice that we've constraint the model space to `linear`
# because we've only defined the search space for `linear`
auto = cflearn.Auto("clf", models="linear").fit(x, y, params=params)
```

Which yields

```text
[I 2020-12-22 19:45:13,597] A new study created in memory with name: linear_optuna
[I 2020-12-22 19:45:14,461] Trial 0 finished with value: 0.401367224752903 and parameters: {'opt': 'adamw'}. Best is trial 0 with value: 0.401367224752903.
[I 2020-12-22 19:45:15,222] Trial 1 finished with value: 0.37729840725660324 and parameters: {'opt': 'sgd'}. Best is trial 0 with value: 0.401367224752903.
[I 2020-12-22 19:45:16,028] Trial 2 finished with value: 0.6434845961630344 and parameters: {'opt': 'adamw'}. Best is trial 2 with value: 0.6434845961630344.
[I 2020-12-22 19:45:16,818] Trial 3 finished with value: 0.14388968795537949 and parameters: {'opt': 'adamw'}. Best is trial 2 with value: 0.6434845961630344.
......
```

As shown above, [`optuna`](https://optuna.org/) will try to search the best hyperparameters with the defined search space for us. Since our search space only contains two possible choices (`{'opt': 'sgd'}` and `{'opt': 'adamw'}`), [`optuna`](https://optuna.org/) will jump between these two choices over and over again.

After the searching we can obtain the searched optimizer via `best_params`:

```python
print(auto.best_params["linear"]["optimizer"])  # adamw
```

Great! Now we know that `adamw` may be better than `sgd`. But soon we'll encounter another issue: what learning rate (`lr`) should we use? Since the default `lr` is `1e-3`, it's hard to tell whether `adamw` will always better than `sgd`, or it is better only if `lr=1e-3`. And again, here's where AutoML could help. Since `lr` should be searched in a logarithm way, we should define the search space as follows:

```python
# the parameters of this search space should define the 'floor' and the 'ceiling'
# in this example, we are specifying 1e-5 <= lr <= 0.1
# and we are using {"log": True} to indicate that we are searching in a logarithm way
lr_param = cflearn.OptunaParam("lr", [1e-5, 0.1], "float", {"log": True})
params = {
    "linear": {
        "optimizer": optimizer_param,
        "optimizer_config": {"lr": lr_param},
    }
}
```

After which we can perform AutoML on this search space:

```python
auto = cflearn.Auto("clf", models="linear").fit(x, y, params=params)
```

Which yields

```text
[I 2020-12-22 19:57:22,893] A new study created in memory with name: linear_optuna
[I 2020-12-22 19:57:23,766] Trial 0 finished with value: 0.12851058691740036 and parameters: {'opt': 'adamw', 'lr': 0.002884297316991861}. Best is trial 0 with value: 0.12851058691740036.
[I 2020-12-22 19:57:24,615] Trial 1 finished with value: 0.26402048021554947 and parameters: {'opt': 'adamw', 'lr': 3.506510110282046e-05}. Best is trial 1 with value: 0.26402048021554947.
[I 2020-12-22 19:57:25,493] Trial 2 finished with value: 0.6952096559107304 and parameters: {'opt': 'adamw', 'lr': 0.01178820649139017}. Best is trial 2 with value: 0.6952096559107304.
[I 2020-12-22 19:57:26,356] Trial 3 finished with value: 0.9276982471346855 and parameters: {'opt': 'adamw', 'lr': 0.08356996061205905}. Best is trial 3 with value: 0.9276982471346855.
......
```

As shown above, this time we are searching for the best `lr` as well. We can futher obtain the searched optimizer and `lr` via `best_params`:

```python
print(auto.best_params["linear"]["optimizer"])               # adamw
print(auto.best_params["linear"]["optimizer_config"]["lr"])  # 0.09311529866070806
```

Great! Now we know the best hyperparameter combination for `linear` model on `iris` dataset: `adamw` with `lr=0.0931`.

#### In a Nut Shell

To recap, if we want to search for a certain hyperparameter, instead of specifying a concrete value, we need to define its search space via `OptunaParam`. For example:

##### the `optimizer` search space

We need to turn

```python
params = {"optimizer": "sgd"}
cflearn.make(config=params).fit(x, y)
```

into

```python
# you can change "opt" into any other (unique) identifier
optimizer_param = cflearn.OptunaParam("opt", ["sgd", "adamw"], "categorical")
# xxx should be your model
params = {"xxx": {"optimizer": optimizer_param}}
cflearn.Auto(..., models="xxx").fit(x, y, params=params)
```

##### the `lr` search space

We need to turn

```python
params = {"optimizer_config": {"lr": 1e-3}}
cflearn.make(config=params).fit(x, y)
```

into

```python
# you can change "lr" into any other (unique) identifier
lr_param = cflearn.OptunaParam("lr", [1e-5, 0.1], "float", {"log": True})
# xxx should be your model
params = {"xxx": {"optimizer_config": {"lr": lr_param}}}
cflearn.Auto(..., models="xxx").fit(x, y, params=params)
```

#### The Default Search Spaces

The default search spaces of `carefree-learn` have already provided examples on how to define search spaces for some critical hyperparameters (e.g. optimizer, learning rate, etc.). To be concrete, `carefree-learn` will use `OptunaPresetParams` to manage a set of default search spaces:

```python
class Auto:
    def __init__(self, ...):
        ...
        self.preset_params = OptunaPresetParams(...)
    
    def fit(self, ...):
        if params is not None:
            model_params = params[model]
        else:
            model_params = self.preset_params.get(model)
```

And the definition of `OptunaPresetParams` is:

```python
class OptunaPresetParams:
    def __init__(
        self,
        *,
        tune_lr: bool = True,
        tune_optimizer: bool = True,
        tune_scheduler: bool = True,
        ...,
        **kwargs: Any,
    ) -> None:
        self.base_params: optuna_params_type = {}
        if tune_lr:
            # update base_params with learning rate search space
            ...
        if tune_optimizer:
            # update base_params with optimizer search space
            ...
        ...

    def get(self, model: str) -> optuna_params_type:
        # Will execute self._{model}_preset() (e.g. self._fcnn_preset()) here
        ...
    
    def _linear_preset(self):
        # Will return the default search spaces for `linear` model
        # * since `linear` model is very simple, we can return the base_params directly
        return shallow_copy_dict(self.base_params)

    def _fcnn_preset(self):
        # Will return the default search spaces for `fcnn` model
        ...
```

You can inspect the source code [here](https://github.com/carefree0910/carefree-learn/blob/752f4190aab49e6fa44e3926c01aeec5dc9a129a/cflearn/api/hpo.py#L774-L814) to see how `carefree-learn` defines its default search spaces, as well as [here](https://github.com/carefree0910/carefree-learn/blob/752f4190aab49e6fa44e3926c01aeec5dc9a129a/cflearn/api/hpo.py#L836-L859) to see how `carefree-learn` defines search spaces for `fcnn`. These two snippets should be able to cover most of the common use cases.

### Define Extra Configurations

If we want to change some default behaviours of `cflearn.Auto`, we can specify the `extra_configs` in `fit`:

```python
auto.fit(..., extra_config={...})
```

And the usage of `extra_config` should be equivalent to the usage of `config` in [`make`](../getting-started/configurations#make) API.

:::note
`extra_config` is not able to overwrite the hyperparameters generated by the search spaces, so in fact the options we can play with it are limited 🤣
:::


## Production

What's facinating is that we can pack the models trained by `cflearn.Auto` into a zip file for production:

```python
auto.pack("pack")
```

Please refer to [AutoML in Production](production#automl-in-production) for more details.


# APIs

In this section, we'll introduce some APIs related to `cflearn.Auto` in details.


## `OptunaParam`

> Source code: [hpo.py -> class OptunaParam](https://github.com/carefree0910/carefree-learn/blob/752f4190aab49e6fa44e3926c01aeec5dc9a129a/cflearn/api/hpo.py#L344).

General interface for defining search spaces of hyperparameters.

```python
class OptunaParam(NamedTuple):
    name: str
    values: Any
    dtype: str  # [int | float | categorical]
    config: Optional[Dict[str, Any]] = None
```

+ **`name`**
    + Specify the **unique** identifier of the current search space.
+ **`values`**
    + Indicate the parameters of this search space.
    + If **`dtype`** is `"int"` or `"float"`, then **`values`** should represent the lower bound and upper bound of this search space.
    + If **`dtype`** is `"categorical"`, then **`values`** should represent the possible choices of this search space.
+ **`dtype`**
    + Indicate the type of this search space.
    + If `"int"`, then this search space will be finite and will only pop integer values.
    + If `"float"`, then this search space will be infinite and will only pop float values.
    + If `"categorical"`, then this search space will be finite and will only pop values specified by **`values`**.
+ **`config`** [default = `None`]
    + Specify other configurations used in `optuna`.
    + There's only one common use case, namely if we need to search in a logarithm way, that we need to specify `config={"log": True}`.

### Examples

```python
import cflearn
from cfdata.tabular import TabularDataset

x, y = TabularDataset.iris().xy
lr_param = cflearn.OptunaParam("lr", [1e-5, 0.1], "float", {"log": True})
params = {"linear": {"optimizer_config": {"lr": lr_param}}}
auto = cflearn.Auto("clf", models="linear").fit(x, y, params=params)
```


## `OptunaPresetParams`

> Source code: [hpo.py -> class OptunaPresetParams](https://github.com/carefree0910/carefree-learn/blob/752f4190aab49e6fa44e3926c01aeec5dc9a129a/cflearn/api/hpo.py#L760).

Structure for defining default search spaces of each model.

```python
class OptunaPresetParams:
    def __init__(
        self,
        *,
        tune_lr: bool = True,
        tune_optimizer: bool = True,
        tune_scheduler: bool = True,
        tune_ema_decay: bool = True,
        tune_clip_norm: bool = True,
        tune_batch_size: bool = True,
        tune_init_method: bool = True,
        **kwargs: Any,
    ):
```

+ **`tune_lr`** [default = `True`]
    + Specify whether we should include the search space of `lr`.
+ **`tune_optimizer`** [default = `True`]
    + Specify whether we should include the search space of `optimizer`.
+ **`tune_ema_decay`** [default = `True`]
    + Specify whether we should include the search space of `ema_decay`.
+ **`tune_clip_norm`** [default = `True`]
    + Specify whether we should include the search space of `clip_norm`.
+ **`tune_init_method`** [default = `True`]
    + Specify whether we should include the search space of `init_method`.
+ **`kwargs`** [default = `{}`]
    + Specify other configs that may be used in the definitions of search spaces of each model.

### `get`

Method for getting the default search spaces of the specific model.

```python
def get(self, model: str) -> optuna_params_type:
```

+ **`model`**
    + Specify which model's default search spaces that we want to get.

#### Examples

```python
import cflearn

preset = cflearn.OptunaPresetParams()
print(preset.get("linear"))  # ...
```


## `cflearn.Auto`

> Source code: [auto.py -> class Auto](https://github.com/carefree0910/carefree-learn/blob/752f4190aab49e6fa44e3926c01aeec5dc9a129a/cflearn/api/auto.py#L42)

`Auto` implement the high-level parts, and should be able to cover the life cycle of an AutoML task.

```python
class Auto:
    def __init__(
        self,
        task_type: task_type_type,
        *,
        models: Union[str, List[str]] = "auto",
        tune_lr: bool = True,
        tune_optimizer: bool = True,
        tune_ema_decay: bool = True,
        tune_clip_norm: bool = True,
        tune_init_method: bool = True,
        **kwargs: Any,
    ):
```

+ **`task_type`**
    + Specify the task type.
    + `"clf"` for classification tasks.
    + `"reg"` for regression tasks.
+ **`models`** [default = `"auto"`]
    + Specify the [Model Space](#define-model-space).
+ **`tune_lr`** [default = `True`]
    + Specify whether we should include the search space of `lr`.
+ **`tune_optimizer`** [default = `True`]
    + Specify whether we should include the search space of `optimizer`.
+ **`tune_ema_decay`** [default = `True`]
    + Specify whether we should include the search space of `ema_decay`.
+ **`tune_clip_norm`** [default = `True`]
    + Specify whether we should include the search space of `clip_norm`.
+ **`tune_init_method`** [default = `True`]
    + Specify whether we should include the search space of `init_method`.
+ **`kwargs`** [default = `{}`]
    + Specify other configs used in [`OptunaPresetParams`](#optunapresetparams).

### `fit`

Method for running the training process of an AutoML task.

```python
def fit(
    self,
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    study_config: Optional[Dict[str, Any]] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Union[str, List[str]]] = None,
    params: Optional[Dict[str, optuna_params_type]] = None,
    num_jobs: int = 1,
    num_trial: int = 20,
    num_repeat: int = 5,
    num_parallel: int = 1,
    timeout: Optional[float] = None,
    score_weights: Optional[Dict[str, float]] = None,
    estimator_scoring_function: Union[str, scoring_fn_type] = default_scoring,
    temp_folder: str = "__tmp__",
    num_final_repeat: int = 20,
    extra_config: general_config_type = None,
    cuda: Optional[Union[str, int]] = None,
) -> "Auto":
```

+ **`x`**
    + Specify the training features.
    + Could be either a `ndarray` or a `file`.
+ **`y`** [default = `None`]
    + Specify the training labels.
    + If **`x`** is a `file`, then we should leave **`y`** unspecified.
+ **`x_cv`** [default = `None`]
    + Specify the cross validation features (if provided).
    + If **`x`** is a `file`, then **`x_cv`** should also be a `file`
+ **`y_cv`** [default = `None`]
    + Specify the cross validation labels (if provided).
    + If **`x_cv`** is a `file`, then we should leave **`y_cv`** unspecified.
+ **`study_config`** [default = `None`]
    + Configs that will be passed into [`optuna.create_study`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html).
+ **`predict_config`** [default = `None`]
    + Configs that will be passed into [`Pipeline.predict`](../user-guides/apis#predict).
+ **`metrics`** [default = `None`]
    + Specify which metric(s) are we going to use to monitor our training process.
    + If not provided, we'll use the [default settings](../getting-started/configurations#metrics).
+ **`params`** [default = `None`]
    + Specify the [Search Spaces](#define-search-spaces).
    + If not provided, we'll use [the default search spaces](#the-default-search-spaces).
+ **`num_jobs`** [default = 1]
    + Number of processes run in parallel.
    + If set to value greater than `1`, we'll use distributed training.
+ **`num_trial`** [default = 20]
    + Number of trials we will run through.
    + This is equivalent to *how many combinations of hyperparameters will we try*.
+ **`num_repeat`** [default = 5]
    + Number of models we train in each trial.
+ **`num_parallel`** [default = 1]
    + Number of processes run in parallel, in each trial.
    + It is recommended to leave it as `1`, unless **`num_repeat`** has been set to a very large number.
+ **`timeout`** [default = `None`]
    + The `timeout` argument used in [`optuna.study.Study.optimize`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).
+ **`score_weights`** [default = `None`]
    + Specify the weights of each metric.
    + It is recommended to leave it as `None`, which means we will treat every metric as equal.
+ **`estimator_scoring_function`** [default = `"default"`]
    + Specify the scoring function we'll use to aggregate every score and get the final score.
+ **`temp_folder`** [default = `"__tmp__"`]
    + Temporary folder in which we used to store the intermediate results.
+ **`num_final_repeat`** [default = `20`]
    + Specify the final `num_repeat` we'll use in [`repeat_with`](../user-guides/apis#repeat_with).
+ **`extra_config`** [default = `None`]
    + Specify the [Extra Configurations](#define-extra-configurations).
+ **`cuda`** [default = `None`]
    + Specify the working GPU.
    + If not provided, `carefree-learn` will try to inference it automatically.

### `best_params`

Property that stores the best hyperparameters searched in the `fit` process.

#### Examples

```python
import cflearn
from cfdata.tabular import TabularDataset

x, y = TabularDataset.iris().xy
auto = cflearn.Auto("clf", models=["linear", "fcnn"]).fit(x, y)
print(auto.best_params)
```

Which yields

```json
{
    "linear": {
        "optimizer_config": {
            "lr": 0.08741785470275337
        },
        "optimizer": "adamw",
        "scheduler": "plateau",
        "model_config": {
            "ema_decay": 0.0,
            "default_encoding_configs": {
                "init_method": "truncated_normal"
            }
        },
        "trainer_config": {
            "clip_norm": 1.0146174647714874
        },
        "batch_size": 32
    },
    "fcnn": {
        "optimizer_config": {
            "lr": 0.035924214329005666
        },
        "optimizer": "rmsprop",
        "scheduler": "plateau",
        "model_config": {
            "ema_decay": 0.0,
            "default_encoding_configs": {
                "init_method": "truncated_normal",
                "embedding_dim": "auto"
            },
            "hidden_units": [
                8
            ],
            "mapping_configs": {
                "batch_norm": false,
                "dropout": 0.4412946560665756,
                "pruner_config": null
            }
        },
        "trainer_config": {
            "clip_norm": 0.0
        },
        "batch_size": 32
    }
}
```

### `pack`

Please refer to [AutoML in Production](production#automl-in-production) for more details.

### `unpack`

Please refer to [AutoML in Production](production#automl-in-production) for more details.
