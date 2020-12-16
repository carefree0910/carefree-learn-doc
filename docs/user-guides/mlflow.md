---
id: mlflow
title: mlflow Integration
---

:::caution
This section is a work in progress.
:::

[`mlflow`](https://mlflow.org) is an open source platform for the machine learning lifecycle. Here are the main features supported (these descriptions are copied from the [official documentation](https://mlflow.org/docs/latest/index.html) of [`mlflow`](https://mlflow.org)):
+ **MLflow Tracking**. The MLflow Tracking component is an API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results. MLflow Tracking lets you log and query experiments using Python API.
+ **MLflow Projects**. An MLflow Project is a format for packaging data science code in a reusable and reproducible way, based primarily on conventions. In addition, the Projects component includes an API and command-line tools for running projects, making it possible to chain together projects into workflows.
+ **MLflow Models**. An MLflow Model is a standard format for packaging machine learning models that can be used in a variety of downstream tools—for example, real-time serving through a REST API or batch inference on Apache Spark. The format defines a convention that lets you save a model in different “flavors” that can be understood by different downstream tools.


## Project Component

The `carefree-learn` itself is served as an **MLflow Project**, as you can find a [`MLproject`](https://github.com/carefree0910/carefree-learn/blob/dev/MLproject) file in the root directory. We've registered some common use cases as command-line tools (e.g. basic training, inference and unittesting), so once you've installed `conda` and `mlflow`, you can play with it by running the following command anywhere you like:

```bash
mlflow run https://github.com/carefree0910/carefree-learn -e test -v dev
```

What's going under the hood is:
1. `mlflow` will install a conda environment with `carefree-learn` (dev branch) for you.
2. The CLI defined in [`MLproject`](https://github.com/carefree0910/carefree-learn/blob/dev/MLproject) will execute the `pytest -v --cov` command (as shown [here](https://github.com/carefree0910/carefree-learn/blob/dev/MLproject#L65)).


## Tracking Component

**Tracking** can be enabled easily in by specifying `mlflow_config` (an empty `dict` will be enough):

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
m = cflearn.make(mlflow_config={}).fit(x, y)
```

After which, we can execute `mlflow ui` in the current working directory to inspect the tracking results (e.g. loss curve, metric curve, etc.).


## Model Component

If we want to serve our models, we can enable the **Model** component by specifying `production`

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
m = cflearn.make(production=...).fit(x, y)
```

where

+ `production="pack"` means [`cflearn.Pack`](production) will be used as the serialzing backend.
+ `production="pipeline"` means [`cflearn.save`](apis#save) will be used as the serialzing backend.
