---
id: apis
title: APIs
---

`carefree-learn` provides many useful APIs for out-of-the-box usages.

:::note
+ For configurations guide, please refer to the [Configurations](../getting-started/configurations) section.
+ For development APIs, please refer to the [Build Your Own Models](../developer-guides/customization) section.
+ For AutoML usages, please refer to the [cflearn.Auto](auto-ml) API.
+ For production usages, please refer to the [cflearn.Pack](production) API.
+ For benchmarking usages, please refer to the [Benchmarking](distributed#benchmarking) section.
:::


## `cflearn.types`

> Source code: [types.py](https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/types.py).

Define some types which are commonly used in `carefree-learn`.

### `pipelines_type`

We'll use `pipelines_type` in most of the APIs of `carefree-learn`.

```python
pipelines_type = Union[
    Pipeline,
    List[Pipeline],
    Dict[str, Pipeline],
    Dict[str, List[Pipeline]],
]
```


## `cflearn.Pipeline`

> Source code: [pipeline.py](https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/pipeline.py).

[`Pipeline`](../design-principles#pipeline) implement the high-level parts, other high-level APIs utilize its methods more or less.

### `fit`

Method for training current `Pipeline` with some given data.

```python
def fit(
    self,
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    sample_weights: Optional[np.ndarray] = None,
) -> "Pipeline":
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
+ **`sample_weights`** [default = `None`]
    + Specify the sample weights of the input data.
    + If **`x`** and **`x_cv`** are both provided, then the length of **`sample_weights`** should be the sum of the length of **`x`** and **`x_cv`**.

### `predict`

Method for making predictions with current `Pipeline` on some given data.

```python
def predict(
    self,
    x: data_type,
    *,
    return_all: bool = False,
    contains_labels: bool = False,
    requires_recover: bool = True,
    returns_probabilities: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
```

+ **`x`**
    + Specify the input features that we want to make predictions on.
    + Could be either a `ndarray` or a `file`.
+ **`return_all`** [default = `False`]
    + Specify whether returns all the predictions.
+ **`contains_labels`** [default = `False`]
    + Specify whether **`x`** contains labels (if it is a file).
+ **`requires_recover`** [default = `True`]
    + Specify whether the model outputs need to be recovered.
    + In most cases this should be left as-is.
+ **`returns_probabilities`** [default = `False`]
    + Specify whether returns the probability predictions (if it is a classification task).
+ **`kwargs`** [default = `{}`]
    + Other `kwargs` that will be passed into [`Model.forward`](https://github.com/carefree0910/carefree-learn/blob/79c9b7fd67fdd8fd874d1c5e85e64448b72424b9/cflearn/models/base.py#L371).


### `predict_prob`

Method for making probability predictions with current `Pipeline` on some given data.

:::caution
This method should not be called if it is a regression task.
:::

```python
def predict_prob(
    self,
    x: data_type,
    *,
    return_all: bool = False,
    contains_labels: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
```

+ **`x`**
    + Specify the input features that we want to make predictions on.
    + Could be either a `ndarray` or a `file`.
+ **`return_all`** [default = `False`]
    + Specify whether returns all the predictions.
+ **`contains_labels`** [default = `False`]
    + Specify whether **`x`** contains labels (if it is a file).
+ **`kwargs`** [default = `{}`]
    + Other `kwargs` that will be passed into [`Model.forward`](https://github.com/carefree0910/carefree-learn/blob/79c9b7fd67fdd8fd874d1c5e85e64448b72424b9/cflearn/models/base.py#L371).

### `to_pattern`

Transform `Pipeline` to [`ModelPattern`](https://github.com/carefree0910/carefree-toolkit/blob/1b406c9c142c5097ad3ed66d02a3cf9c2ae40507/cftool/ml/utils.py#L501).

```python
def to_pattern(
    self,
    *,
    pre_process: Optional[Callable] = None,
    **predict_kwargs: Any,
) -> ModelPattern:
```

+ **`pre_process`** [default = `None`]
    + If not `None`, we'll use it to pre-process the input data before we actually call [`predict`](#predict).
+ **`predict_kwargs`** [default = `{}`]
    + `kwargs` that will be passed into [`predict`](#predict).


## `cflearn.api.basic`

> Source code: [basic.py](https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/api/basic.py)

### `make`

General method for training neural networks with `carefree-learn`.

:::note
Please refer to [`make`](../getting-started/configurations#make) section for more details.
:::

### `evaluate`

Used for evaluating models from `carefree-learn` or from other sources.

```python
def evaluate(
    x: data_type,
    y: data_type = None,
    *,
    contains_labels: bool = True,
    pipelines: Optional[pipelines_type] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Union[str, List[str]]] = None,
    metric_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    other_patterns: Optional[Dict[str, patterns_type]] = None,
    comparer_verbose_level: Optional[int] = 1,
) -> Comparer
```

+ **`x`**
    + Specify the input features that we want to evaluate on.
    + Could be either a `ndarray` or a `file`, it depends on what kinds of data did you train your model with.
+ **`y`** [default = `None`]
    + Specify the input labels that we want to evaluate on.
    + If **`x`** is a `file`, then we should leave **`y`** unspecified.
+ **`contains_labels`** [default = `True`]
    + Specify whether **`x`** contains labels (if it is a file).
    + This should always be `True` because we need target labels for evaluating.
+ **`pipelines`** [default = `None`]
    + Specify the `cflearn` models we want to evaluate with.
    + This can be used together with [`cflearn.repeat_with`](#repeat_with), as shown in the [`Benchmarking`](distributed#benchmarking) section.
+ **`predict_config`** [default = `None`]
    + Configs that will be passed into [`Pipeline.predict`](#predict).
+ **`metrics`** [default = `None`]
    + Metrics that we'll use for evaluating.
    + If not provided, then **`pipelines`** must be provided. In this case, we'll use metrics of the first [`Pipeline`](../design-principles#pipeline) presented in **`pipelines`** as our **`metrics`**.
+ **`metric_configs`** [default = `None`]
    + Metric configs that we'll use for corresponding metrics.
+ **`other_patterns`** [default = `None`]
    + Other models we want to evaluate with.
    + A common use case is to compare `cflearn` models with `sklearn` models ([`examples`](#examples)).
+ **`comparer_verbose_level`** [default = `1`]
    + `verbose_level` used in [`Comparer.compare`](https://github.com/carefree0910/carefree-toolkit/blob/1b406c9c142c5097ad3ed66d02a3cf9c2ae40507/cftool/ml/utils.py#L871).

:::caution
When utilizing **`other_patterns`**, we need to make sure that the given [ModelPattern](https://github.com/carefree0910/carefree-toolkit/blob/1b406c9c142c5097ad3ed66d02a3cf9c2ae40507/cftool/ml/utils.py#L501) has `predict_method` that outputs 2d arrays, as illustrated [here](https://github.com/carefree0910/carefree-toolkit/blob/1b406c9c142c5097ad3ed66d02a3cf9c2ae40507/cftool/ml/utils.py#L521-L522).
:::

#### Examples

```python
import cflearn
import numpy as np
from cftool.ml import ModelPattern
from sklearn.svm import LinearSVR

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
m = cflearn.make("linear").fit(x, y)
skm = LinearSVR().fit(x, y.ravel())
sk_pattern = ModelPattern(predict_method=lambda x: skm.predict(x).reshape([-1, 1]))
cflearn.evaluate(x, y, pipelines=m, other_patterns={"sklearn": sk_pattern})
```

Which yields

```text
================================================================================================================================
|        metrics         |                       mae                        |                       mse                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.319494    | -- 0.000000 -- |    -0.31949    |    0.151017    | -- 0.000000 -- |    -0.15101    |
--------------------------------------------------------------------------------------------------------------------------------
|        sklearn         | -- 0.246084 -- | -- 0.000000 -- | -- -0.24608 -- | -- 0.083311 -- | -- 0.000000 -- | -- -0.08331 -- |
================================================================================================================================
```

### `save`

General method for saving neural networks trained with `carefree-learn`.

```python
def save(
    pipelines: pipelines_type,
    identifier: str = "cflearn",
    saving_folder: Optional[str] = None,
    *,
    retain_data: bool = True,
) -> Dict[str, List[Pipeline]]:
```

+ **`pipelines`**
    + Models trained in `carefree-learn`.
+ **`identifier`** [default = `"cflearn"`]
    + Specify the identifier of the saved model.
    + In most cases this should be left as-is.
+ **`saving_folder`** [default = `None`]
    + Specify the saving folder of the models.
    + If not provided, the models will be saved in the current working directory.
+ **`retain_data`** [default = `True`]
    + Whether retain the data information. In most cases this should not be set manually because `carefree-learn` has provided [`cflearn.Pack`](production) API for production.

#### Examples

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
m = cflearn.make().fit(x, y)
cflearn.save(m)  # this will generate a `cflearn^_^fcnn^_^0000.zip` in the current working directory
```

### `load`

General method for loading neural networks trained with `carefree-learn`.

```python
def load(
    identifier: str = "cflearn",
    saving_folder: Optional[str] = None,
) -> Dict[str, List[Pipeline]]:
```

+ **`identifier`** [default = `"cflearn"`]
    + Specify the identifier of the loaded model.
    + In most cases this should be left as-is.
+ **`saving_folder`** [default = `None`]
    + Specify the loading folder of the models.
    + If not provided, the models will be loaded from the current working directory.

#### Examples

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
m = cflearn.make().fit(x, y)
cflearn.save(m)
ms = cflearn.load()
print(ms)  # {'fcnn': [FCNN()]}
assert np.allclose(m.predict(x), ms["fcnn"][0].predict(x))
```

### `repeat_with`

General method for training multiple neural networks on fixed datasets with `carefree-learn`.

```python
def repeat_with(
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    models: Union[str, List[str]] = "fcnn",
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    identifiers: Optional[Union[str, List[str]]] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    sequential: Optional[bool] = None,
    num_jobs: int = 1,
    num_repeat: int = 5,
    temp_folder: str = "__tmp__",
    return_patterns: bool = True,
    use_tqdm: bool = True,
    **kwargs: Any,
) -> RepeatResult:
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
+ **`models`** [default = `"fcnn"`]
    + Specify the model(s) we want to train.
    + It could either be an `str` or be a list of `str`, which is useful for [`Benchmarking`](distributed#benchmarking).
+ **`model_configs`** [default = `None`]
    + Configurations for each model.
+ **`identifiers`** [default = `None`]
    + **`identifiers`** that will be used in [`cflearn.save`](#save) for each model.
+ **`predict_config`** [default = `None`]
    + Configs that will be passed into [`Pipeline.predict`](#predict).
+ **`sequential`** [default = `None`]
    + Specify whether force not to use distributed training.
    + If not provided, it will be determined by `num_jobs` (`sequential = num_jobs <= 1`).
+ **`num_jobs`** [default = 1]
    + Number of processes run in parallel.
    + If set to value greater than `1`, we'll use distributed training.
+ **`num_repeat`** [default = 5]
    + Number of models we train for each specified model.
+ **`temp_folder`** [default = `"__tmp__"`]
    + Temporary folder in which we used to store intermediate results generated by sequential / distributed training.
+ **`return_patterns`** [default = `True`]
    + Specify whether generate [`ModelPattern`](https://github.com/carefree0910/carefree-toolkit/blob/1b406c9c142c5097ad3ed66d02a3cf9c2ae40507/cftool/ml/utils.py#L501)s of corresponding [`Pipeline`](../design-principles#pipeline)s.
    + If `True`, we'll utilize [`to_pattern`](#to_pattern) to generate the patterns.
+ **`use_tqdm`** [default = `True`]
    + Whether utilize the `tqdm` progress bar or not.
+ **`kwargs`** [default = `{}`]
    + Specify other `kwargs` that will be passed into [`make`](../getting-started/configurations#make).

#### Examples

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
result = cflearn.repeat_with(x, y)
print(result.pipelines)  # {'fcnn': [FCNN(), FCNN(), FCNN(), FCNN(), FCNN()]}
```
