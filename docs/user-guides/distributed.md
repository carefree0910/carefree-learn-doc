---
id: distributed
title: Distributed
---


## Distributed Training

In `carefree-learn`, **Distributed Training** doesn't mean training your model on multiple GPUs or multiple machines, because `carefree-learn` focuses on tabular datasets (or, structured datasets) which are often not as large as unstructured datasets. Instead, **Distributed Training** in `carefree-learn` means **training multiple models** at the same time. This is important because:

+ Deep Learning models suffer from randomness, so we need to train multiple models with the same algorithm and calculate the mean / std of the performances to evaluate the algorithm's capacity and stability.
+ Ensemble these models (which are trained with the same algorithm) can boost the algorithm's performance without making any changes to the algorithm itself.
+ Parameter searching will be easier & faster.

There are two ways to perform distributed training in `carefree-learn`: through high-level API [`cflearn.repeat_with`](apis#repeat_with) or through helper class [`Experiment`](#experiment). We'll introduce their usages in the following sections.

### `repeat_with`

`repeat_with` is the general method for training multiple neural networks on fixed datasets. It can be used in either *sequential* mode or *distributed* mode. If *distributed* mode is enabled, it will leverage the helper class [`Experiment`](#experiment) internally (here are the pseudo codes):

```python
experiment = Experiment()
for model in models:
    for _ in range(num_repeat):
        experiment.add_task(
            model=model,
            config=fetch_config(model),
            data_folder=data_folder,
        )
```

`repeat_with` is very useful when we want to quickly inspect some statistics of our model (e.g. bias and variance), because you can distributedly perform the same algorithm over the same datasets, and then [`cflearn.evaluate`](apis#evaluate) will handle the statistics for you:

```python
results = cflearn.repeat_with(x, y, num_repeat=..., num_jobs=...)
cflearn.evaluate(x, y, metrics=...)
```

:::info
+ See [Benchmarking](#benchmarking) section for more details.
+ See [here](apis#repeat_with) for the detailed API documentation.
:::

### `Experiment`

If we want to customize the distributed training process (instead of simply replicating), `carefree-learn` also provides an `Experiment` class for us to control every experiment setting, including:
+ Which model should we use for a specific task.
+ Which dataset should we use for a specific task.
+ Which configuration should we use for a specific task.
+ And everything else...

Here are two examples that may frequently appear in real scenarios:

#### Training Multiple Models on Same Dataset

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 10])

experiment = cflearn.Experiment()
# Since we will train every model on x & y, we should dump them to a `data_folder` first.
# After that, every model can access this dataset by reading `data_folder`.
data_folder = experiment.dump_data_bundle(x, y)
# We can add task which will train a model on the dataset.
for model in ["linear", "fcnn", "tree_dnn"]:
    # Don't forget to specify the `data_folder`!
    experiment.add_task(model=model, data_folder=data_folder)
# After adding the tasks, we can run our tasks easily.
# Remember to specify the `task_loader` if we want to fetch the `pipeline_dict`.
results = experiment.run_tasks(task_loader=cflearn.task_loader)
print(results.pipelines)  # [FCNN(), LinearModel(), TreeDNN()]
```

#### Training Same Model on Different Datasets

```python
import cflearn
import numpy as np

x1 = np.random.random([1000, 10])
y1 = np.random.random([1000, 10])
x2 = np.random.random([1000, 10])
y2 = np.random.random([1000, 10])

experiment = cflearn.Experiment()
# What's going under the hood here is that `carefree-learn` will 
#  call `dump_data_bundle` internally to manage the datasets
experiment.add_task(x1, y1)
experiment.add_task(x2, y2)
results = experiment.run_tasks(task_loader=cflearn.task_loader)
print(results.pipelines)  # [FCNN(), FCNN()]
```

### Conclusions

+ If we want to train same model on same dataset multiple times, use `repeat_with`.
+ Otherwise, use `Experiment`, and keep in mind that:
    + If we need to share data, use `dump_data_bundle` to dump the shared data to a `data_folder`, then specify this `data_folder` when we call `add_task`.
    + If we want to add a rather 'isolated' task, simply call `add_task` with the corresponding dataset will be fine.
    + Specify `task_loader=cflearn.task_loader` if we want to fetch the `pipeline_dict`.

:::info
`Experiment` supports much more customizations (e.g. customize configurations) than those mentioned above. Please refer to [Advanced Benchmarking](#advanced-benchmarking) for more details.
:::


## Benchmarking

`carefree-learn` has a related repository (namely [`carefree-learn-benchmark`](https://github.com/carefree0910/carefree-learn-benchmark)) which implemented some sophisticated benchmarking functionalities. However, for many common use cases, `carefree-learn` provides [`cflearn.repeat_with`](apis#repeat_with) and [`cflearn.evaluate`](apis#evaluate) for quick benchmarking. For example, if we want to compare the `linear` model and the `fcnn` model by running them `3` times, we can simply:

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])

if __name__ == "__main__":
    # Notice that there will always be 2 models training simultaneously with `num_jobs=2`
    result = cflearn.repeat_with(x, y, models=["linear", "fcnn"], num_repeat=3, num_jobs=2)
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

:::note
+ It is necessary to wrap codes under `__main__` on WINDOWS when running distributed codes.
+ You might notice that the best results of each column is highlighted with a pair of '--'.
:::

:::caution
It is not recommended to enable distributed training unless:
+ There are plenty of tasks that we need to run. 
+ Running each task is quite costly in time.
+ `num_jobs` could be set to a relatively high value (e.g., `8`).

Otherwise the overhead brought by launching distributed training might actually hurt the overall performance.

However, there are no 'golden rules' of whether we should use distributed training or not for us to follow, so the best practice is to actually try it out in a smaller scale 🤣
:::

### Advanced Benchmarking

In order to serve as a carefree tool, `carefree-learn` is able to perform advanced benchmarking (e.g. compare with scikit-learn models) in a few lines of code (in a distributed mode, if needed).

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])

experiment = cflearn.Experiment()
data_folder = experiment.dump_data_bundle(x, y)

# Add carefree-learn tasks
for model in ["linear", "fcnn", "tree_dnn"]:
    experiment.add_task(model=model, data_folder=data_folder)
# Add scikit-learn tasks
run_command = "python run_sklearn.py"
experiment.add_task(model="svr", run_command=run_command, data_folder=data_folder)
experiment.add_task(model="linear_svr", run_command=run_command, data_folder=data_folder)
```

Notice that we specified `run_command="python run_sklearn.py"` for scikit-learn tasks, which means [`Experiment`](#experiment) will try to execute this command in the current working directory for training scikit-learn models. The good news is that we do not need to speciy any command line arguments, because [`Experiment`](#experiment) will handle those for us.

Here is basically what a `run_sklearn.py` should look like:

```python
import os
import pickle

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from cflearn.dist.runs._utils import get_info

if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    # data
    data_list = info.data_list
    x, y = data_list[:2]
    # model
    model = kwargs["model"]
    sk_model = (SVR if model == "svr" else LinearSVR)()
    # train & save
    sk_model.fit(x, y.ravel())
    with open(os.path.join(info.workplace, "sk_model.pkl"), "wb") as f:
        pickle.dump(sk_model, f)
```

With `run_sklearn.py` defined, we should run those tasks without `task_loader` (because `sk_model` cannot be loaded by `carefree-learn` internally):

```python
results = experiment.run_tasks()
```

After finished running, we should be able to see this file structure in the current working directory:

```text
|--- __experiment__
   |--- __data__
      |-- x.npy
      |-- y.npy
   |--- fcnn/0
      |-- _logs
      |-- __meta__.json
      |-- cflearn^_^fcnn^_^0000.zip
   |--- linear/0
      |-- ...
   |--- tree_dnn/0
      |-- ...
   |--- linear_svr/0
      |-- __meta__.json
      |-- sk_model.pkl
   |--- svr/0
      |-- ...
```

As we expected, `carefree-learn` models are saved into zip files, while scikit-learn models are saved into `sk_model.pkl`. We can further inspect these models with `cflearn.evaluate`:

```python
import os
import pickle

pipelines = {}
scikit_patterns = {}
for workplace, workplace_key in zip(results.workplaces, results.workplace_keys):
    model = workplace_key[0]
    if model not in ["svr", "linear_svr"]:
        pipelines[model] = cflearn.task_loader(workplace)
    else:
        model_file = os.path.join(workplace, "sk_model.pkl")
        with open(model_file, "rb") as f:
            sk_model = pickle.load(f)
            # In `carefree-learn`, we treat labels as column vectors.
            # So we need to reshape the outputs from the scikit-learn models.
            sk_predict = lambda x: sk_model.predict(x).reshape([-1, 1])
            sk_pattern = cflearn.ModelPattern(predict_method=sk_predict)
            scikit_patterns[model] = sk_pattern

cflearn.evaluate(x, y, pipelines=pipelines, other_patterns=scikit_patterns)
```

Which yields

```text
~~~  [ info ] Results
================================================================================================================================
|        metrics         |                       mae                        |                       mse                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.246332    | -- 0.000000 -- |    -0.24633    |    0.082304    | -- 0.000000 -- |    -0.08230    |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.251605    | -- 0.000000 -- |    -0.25160    |    0.087469    | -- 0.000000 -- |    -0.08746    |
--------------------------------------------------------------------------------------------------------------------------------
|       linear_svr       | -- 0.168027 -- | -- 0.000000 -- | -- -0.16802 -- | -- 0.043490 -- | -- 0.000000 -- | -- -0.04349 -- |
--------------------------------------------------------------------------------------------------------------------------------
|          svr           | -- 0.168027 -- | -- 0.000000 -- | -- -0.16802 -- | -- 0.043490 -- | -- 0.000000 -- | -- -0.04349 -- |
--------------------------------------------------------------------------------------------------------------------------------
|        tree_dnn        |    0.246306    | -- 0.000000 -- |    -0.24630    |    0.082190    | -- 0.000000 -- |    -0.08219    |
================================================================================================================================
```


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

Which yields

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
