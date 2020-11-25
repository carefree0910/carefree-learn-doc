---
id: distributed
title: Distributed
---


## Distributed Training

In `carefree-learn`, **Distributed Training** doesn't mean training your model on multiple GPUs or multiple machines, because `carefree-learn` focuses on tabular datasets (or, structured datasets) which are often not as large as unstructured datasets. Instead, **Distributed Training** in `carefree-learn` means **training multiple models** at the same time. This is important because:

+ Deep Learning models suffer from randomness, so we need to train multiple models with the same algorithm and calculate the mean / std of the performances to evaluate the algorithm's capacity and stability.
+ Ensemble these models (which are trained with the same algorithm) can boost the algorithm's performance without making any changes to the algorithm itself.
+ Parameter searching will be easier & faster.

```python
import cflearn
from cfdata.tabular import TabularDataset

if __name__ == '__main__':
    x, y = TabularDataset.iris().xy
    # Notice that 3 fcnn were trained simultaneously with this line of code
    results = cflearn.repeat_with(x, y, num_repeat=3, num_jobs=0)
    patterns = results.patterns["fcnn"]
    # And it is fairly straight forward to apply stacking ensemble
    ensemble = cflearn.Ensemble.stacking(patterns)
    patterns_dict = {"fcnn_3": patterns, "fcnn_3_ensemble": ensemble}
    cflearn.evaluate(x, y, metrics=["acc", "auc"], other_patterns=patterns_dict)
```

:::note
It is necessary to wrap codes under `__main__` on WINDOWS when running distributed codes.
:::

Which yields:

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|         fcnn_3         |    0.937778    |    0.017498    |    0.920280    | -- 0.993911 -- |    0.000274    |    0.993637    |
--------------------------------------------------------------------------------------------------------------------------------
|    fcnn_3_ensemble     | -- 0.953333 -- | -- 0.000000 -- | -- 0.953333 -- |    0.993867    | -- 0.000000 -- | -- 0.993867 -- |
================================================================================================================================
```

:::info
You might notice that the best results of each column is highlighted with a pair of '--'.
:::


## Benchmarking

`carefree-learn` has a related repository (namely [`carefree-learn-benchmark`](https://github.com/carefree0910/carefree-learn-benchmark)) which implemented some sophisticated benchmarking functionalities. However, for many common use cases, `carefree-learn` provides `cflearn.repeat_with` and `cflearn.evaluate` for quick benchmarking. For example, if we want to compare the `linear` model and the `fcnn` model by running them `3` times, we can simply:

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
result = cflearn.repeat_with(x, y, models=["linear", "fcnn"], num_repeat=3)
cflearn.evaluate(x, y, pipelines=result.pipelines)
```

Which yields

```text
================================================================================================================================
|        metrics         |                       mae                        |                       mse                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          | -- 0.251717 -- | -- 0.002158 -- | -- -0.25387 -- | -- 0.086110 -- | -- 0.002165 -- | -- -0.08827 -- |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.283154    |    0.015341    |    -0.29849    |    0.118122    |    0.016185    |    -0.13430    |
================================================================================================================================
```

We can also leverage distributed training supported in `carefree-learn` for faster benchmarking by specifying `num_jobs` to a higher value:

```python
if __name__ == "__main__":
    result = cflearn.repeat_with(x, y, models=["linear", "fcnn"], num_repeat=3, num_jobs=2)
    cflearn.evaluate(x, y, pipelines=result.pipelines)
```

:::caution
It is not recommended to enable distributed training unless:
+ There are plenty of tasks that we need to run. 
+ Running each task is quite costly in time.
+ `num_jobs` could be set to a relatively high value (e.g., `8`).

Otherwise the overhead brought by launching distributed training might actually hurt the overall performance.

However, there are no 'golden rules' of whether we should use distributed training or not for us to follow, so the best practice is to actually try it out in a smaller scale 🤣
:::


## Hyper Parameter Optimization (HPO)

Although `carefree-learn` has already provided an [`AutoML`](auto-ml) API, we can still play with the **HPO** APIs manually:

```python
import cflearn
from cfdata.tabular import TabularDataset
 
if __name__ == '__main__':
    x, y = TabularDataset.iris().xy
    # Bayesian Optimization (BO) will be used as default
    hpo = cflearn.tune_with(
        x, y,
        task_type="clf",
        num_repeat=2, num_parallel=0, num_search=10
    )
    # We can further train our model with the best hyper-parameters we've obtained:
    m = cflearn.make(**hpo.best_param).fit(x, y)
    cflearn.evaluate(x, y, pipelines=m)
```

Then you will see something like this:

```text
~~~  [ info ] Results
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|        0659e09f        |    0.943333    |    0.016667    |    0.926667    |    0.995500    |    0.001967    |    0.993533    |
--------------------------------------------------------------------------------------------------------------------------------
|        08a0a030        |    0.796667    |    0.130000    |    0.666667    |    0.969333    |    0.012000    |    0.957333    |
--------------------------------------------------------------------------------------------------------------------------------
|        1962285c        |    0.950000    |    0.003333    |    0.946667    |    0.997467    |    0.000533    |    0.996933    |
--------------------------------------------------------------------------------------------------------------------------------
|        1eb7f2a0        |    0.933333    |    0.020000    |    0.913333    |    0.994833    |    0.003033    |    0.991800    |
--------------------------------------------------------------------------------------------------------------------------------
|        4ed5bb3b        |    0.973333    |    0.013333    |    0.960000    |    0.998733    |    0.000467    |    0.998267    |
--------------------------------------------------------------------------------------------------------------------------------
|        5a652f3c        |    0.953333    | -- 0.000000 -- |    0.953333    |    0.997400    |    0.000133    |    0.997267    |
--------------------------------------------------------------------------------------------------------------------------------
|        82c35e77        |    0.940000    |    0.020000    |    0.920000    |    0.995467    |    0.002133    |    0.993333    |
--------------------------------------------------------------------------------------------------------------------------------
|        a9ef52d0        | -- 0.986667 -- |    0.006667    | -- 0.980000 -- | -- 0.999200 -- | -- 0.000000 -- | -- 0.999200 -- |
--------------------------------------------------------------------------------------------------------------------------------
|        ba2e179a        |    0.946667    |    0.026667    |    0.920000    |    0.995633    |    0.001900    |    0.993733    |
--------------------------------------------------------------------------------------------------------------------------------
|        ec8c0837        |    0.973333    | -- 0.000000 -- |    0.973333    |    0.998867    |    0.000067    |    0.998800    |
================================================================================================================================

~~~  [ info ] Best Parameters
----------------------------------------------------------------------------------------------------
acc  (a9ef52d0) (0.986667 ± 0.006667)
----------------------------------------------------------------------------------------------------
{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.005810863965757382}}
----------------------------------------------------------------------------------------------------
auc  (a9ef52d0) (0.999200 ± 0.000000)
----------------------------------------------------------------------------------------------------
{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.005810863965757382}}
----------------------------------------------------------------------------------------------------
best (a9ef52d0)
----------------------------------------------------------------------------------------------------
{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.005810863965757382}}
----------------------------------------------------------------------------------------------------

~~  [ info ] Results
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.980000    |    0.000000    |    0.980000    |    0.998867    |    0.000000    |    0.998867    |
================================================================================================================================
```

You might notice that:

+ The final results obtained by **HPO** is even better than the stacking ensemble results mentioned above.
+ We search for `optimizer` and `lr` as default. In fact, we can manually passed `params` into `cflearn.tune_with`. If not, then `carefree-learn` will execute following codes:
```python
from cftool.ml.param_utils import *

params = {
    "optimizer": String(Choice(values=["sgd", "rmsprop", "adam"])),
    "optimizer_config": {
        "lr": Float(Exponential(1e-5, 0.1))
    }
}
```
