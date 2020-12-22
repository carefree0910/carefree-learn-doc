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
2. fetch pre-defined hyper-parameters search space of each model from `OptunaPresetParams` (and inject manual configurations, if provided; see [Define Extra Configurations](#define-extra-configurations) for more details).
3. leverage `optuna` with `cflearn.optuna_tune` to perform hyper-parameters optimization.
4. use searched hyper-parameters to train each model multiple times (separately).
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
If you are more interested in codes than a step by step tutorial, you can jump to the [In a Nut Shell](#in-a-nut-shell) section directly.
:::

For example, if we want to use `sgd` instead of the default `adamw`, we can simply

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
m = cflearn.make(optimizer="sgd").fit(x, y)
print(m.trainer.optimizers["all"])  # SGD(...)
```

but we are not sure which one is better, and here's AutoML that comes to help. Since we should choose from either `sgd` or `adamw`, the search space is pretty simple:

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
auto = cflearn.Auto("reg", models="linear").fit(x, y, params=params)
```

Which yields

```text
[I 2020-12-22 18:16:44,946] A new study created in memory with name: linear_optuna
[I 2020-12-22 18:16:49,200] Trial 0 finished with value: -1.174601199105382 and parameters: {'opt': 'adamw'}. Best is trial 0 with value: -1.174601199105382.
[I 2020-12-22 18:16:55,643] Trial 1 finished with value: -1.1822145022451878 and parameters: {'opt': 'sgd'}. Best is trial 0 with value: -1.174601199105382.
[I 2020-12-22 18:17:02,122] Trial 2 finished with value: -1.2407564036548138 and parameters: {'opt': 'sgd'}. Best is trial 0 with value: -1.174601199105382.
[I 2020-12-22 18:17:05,218] Trial 3 finished with value: -1.2113975938409567 and parameters: {'opt': 'adamw'}. Best is trial 0 with value: -1.174601199105382.
......
```

As shown above, `optuna` will try to search the best hyperparameters with the defined search space for us. Since our search space only contains two possible choices (`{'opt': 'sgd'}` and `{'opt': 'adamw'}`), `optuna` will jump between these two choices over and over again.

After the searching we can obtain the searched optimizer via `best_params`:

```python
print(auto.best_params["optimizer"])
```

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
# xxx here is your model
params = {"xxx": {"optimizer": optimizer_param}}
```

#### Default Search Space

The default search space of `carefree-learn` have already provided examples on how to define search space for some critical hyperparameters (e.g. optimizer, learning rate, etc.). To be concrete, `carefree-learn` will use `OptunaPresetParams` to manage a set of default search spaces:

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
        # Will return the default search space for `linear` model
        # * since `linear` model is very simple, we can return the base_params directly
        return shallow_copy_dict(self.base_params)

    def _fcnn_preset(self):
        # Will return the default search space for `fcnn` model
        ...
```

### Define Extra Configurations

If we want to change some default behaviours of `cflearn.Auto`, we can specify the `extra_configs` in `fit`:

```python
auto.fit(..., extra_config={...})
```

And the usage of `extra_config` should be equivalent to the usage of `config` in [`make`](../getting-started/configurations#make) API.

:::note
`extra_config` is not able to overwrite the hyperparameters generated by the search space, so in fact the options we can play with it are limited 🤣
:::


## Production

What's facinating is that we can pack the models trained by `cflearn.Auto` into a zip file for production:

```python
auto.pack("pack")
```

Please refer to [AutoML in Production](production#automl-in-production) for more details.
