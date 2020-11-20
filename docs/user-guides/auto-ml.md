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

Then you will see something like this:

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

### Explained

`cflearn.Auto.fit` will run through the following steps:
1. define the model space automatically (or manually)
2. fetch pre-defined hyper-parameters search space of each model from `OptunaPresetParams`.
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
