---
id: Iris
title: Iris
---

| Python source code | Jupyter Notebook |
|:---:|:---:|
| [iris.py](https://github.com/carefree0910/carefree-learn/blob/dev/examples/iris/iris.py) | [iris.ipynb](https://github.com/carefree0910/carefree-learn/blob/dev/examples/iris/iris.ipynb) |

Here are some of the information provided by the official website:

```text
This is perhaps the best known database to be found in the pattern recognition literature.
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
Predicted attribute: class of iris plant.
```

And here's the pandas-view of the raw data:

```text
      f0   f1   f2   f3           label
0    5.1  3.5  1.4  0.2     Iris-setosa
1    4.9  3.0  1.4  0.2     Iris-setosa
2    4.7  3.2  1.3  0.2     Iris-setosa
3    4.6  3.1  1.5  0.2     Iris-setosa
4    5.0  3.6  1.4  0.2     Iris-setosa
..   ...  ...  ...  ...             ...
145  6.7  3.0  5.2  2.3  Iris-virginica
146  6.3  2.5  5.0  1.9  Iris-virginica
147  6.5  3.0  5.2  2.0  Iris-virginica
148  6.2  3.4  5.4  2.3  Iris-virginica
149  5.9  3.0  5.1  1.8  Iris-virginica

[150 rows x 5 columns]
```

:::note
+ You can download the raw data (`iris.data`) with [this link](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).
+ We didn't use pandas in our code, but it is convenient to visualize some data with it though 🤣
:::


## Basic Usages

Traditionally, we need to process the raw data before we feed them into our machine learning models (e.g. encode the label column, which is a string column, into an ordinal column). In `carefree-learn`, however, we can train neural networks directly on files without worrying about the rest:

```python
import cflearn

m = cflearn.make().fit("iris.data")
```

What's going under the hood is that `carefree-learn` will try to parse the `iris.data` automatically (with the help of [carefree-data](https://github.com/carefree0910/carefree-data)), split the data into training set and validation set, with which we'll train a fully connected neural network (fcnn).

We can further inspect the processed data if we want to know how `carefree-learn` actually parsed the input data:

```python
print(m.tr_data.raw.x[0])
print(m.tr_data.raw.y[0])
print(m.tr_data.processed.x[0])
print(m.tr_data.processed.y[0])
```

Which yields

```text
['4.6', '3.6', '1.0', '0.2']
['Iris-setosa']
[-1.5065205  1.2634597 -1.5687355 -1.3129768]
[0]
```

It shows that the raw data is carefully normalized into numerical data that neural networks can accept. You may also notice that the first elements are not identical with the first line of the raw data, this is caused by the auto-shuffle mechanism introduced in [carefree-data](https://github.com/carefree0910/carefree-data).

What's more, by saying *normalized*, it means that the input features will be automatically normalized to `mean=0.0` and `std=1.0`:

```python
import numpy as np

tr_x = m.tr_data.processed.x
cv_x = m.cv_data.processed.x
stacked = np.vstack([tr_x, cv_x])
print(stacked.mean(0))
print(stacked.std(0))
```

Which yields

```text
[ 3.1739475e-08 -3.7471455e-07 -1.9907951e-07 -8.0267590e-08]
[0.99999976 0.9999997  1.0000002  0.9999999 ]
```

:::info
The results shown above means we first normalized the data before we actually split it into train & validation set.
:::

After training on files, `carefree-learn` can predict & evaluate on files directly as well. We'll handle the data parsing and normalization for you automatically:

```python
# `contains_labels` is set to True because `iris.data` itself contains labels
predictions = m.predict("iris.data", contains_labels=True)
# evaluations could be achieved easily with cflearn.evaluate
cflearn.evaluate("iris.data", pipelines=m)
```

Which yields

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.926667    |    0.000000    |    0.926667    |    0.994800    |    0.000000    |    0.994800    |
================================================================================================================================
```


## Benchmarking

As we know, neural networks are trained with **_stochastic_** gradient descent (and its variants), which will introduce some randomness to the final result, even if we are training on the same dataset. In this case, we need to repeat the same task several times in order to obtain the bias & variance of our neural networks. Fortunately, `carefree-learn` introduced [`repeat_with`](../user-guides/distributed#repeat_with) API, which can achieve this goal easily with only a few lines of code:

```python
# With num_repeat=3 specified, we'll train 3 models on `iris.data`.
result = cflearn.repeat_with("iris.data", num_repeat=3)
cflearn.evaluate("iris.data", pipelines=result.pipelines)
```

Which yields

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.902222    |    0.019116    |    0.883106    |    0.985778    |    0.004722    |    0.981055    |
================================================================================================================================
```

We can also compare the performances across different models:

```python
# With models=["linear", "fcnn"], we'll train both linear models and fcnn models.
models = ["linear", "fcnn"]
result = cflearn.repeat_with("iris.data", models=models, num_repeat=3)
cflearn.evaluate("iris.data", pipelines=result.pipelines)
```

Which yields

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          | -- 0.915556 -- | -- 0.027933 -- | -- 0.887623 -- | -- 0.985467 -- | -- 0.004121 -- | -- 0.981345 -- |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.620000    |    0.176970    |    0.443030    |    0.733778    |    0.148427    |    0.585351    |
================================================================================================================================
```

It is worth mentioning that `carefree-learn` supports [`Distributed Training`](../user-guides/distributed#distributed-training), which means when we need to perform large scale benchmarking (e.g. train 100 models), we could accelerate the process through multiprocessing:

:::info
In `carefree-learn`, [Distributed Training](../user-guides/distributed#distributed-training) doesn't mean training your model on multiple GPUs or multiple machines. Instead, [Distributed Training](../user-guides/distributed#distributed-training) in `carefree-learn` means training multiple models at the same time.
:::

```python
# With num_jobs=2, we will launch 2 processes to run the tasks in a distributed way.
result = cflearn.repeat_with("iris.data", num_repeat=10, num_jobs=2)
```

:::caution
On iris dataset, however, launching distributed training will actually hurt the speed because iris dataset only contains 150 samples, so the relative overhead brought by distributed training will be too large. Please refer the **CAUTION** section of the [Benchmarking](../user-guides/distributed#benchmarking) section for more details.
:::

## Advanced Benchmarking

But this is not enough, because we want to know whether other models (e.g. scikit-learn models) could achieve a better performance than `carefree-learn` models. In this case, we can perform an advanced benchmarking with the [`Experiment`](../user-guides/distributed#experiment) helper class.

```python
experiment = cflearn.Experiment()
tr_x, tr_y = m.tr_data.processed.xy
cv_x, cv_y = m.cv_data.processed.xy
data_folder = experiment.dump_data_bundle(tr_x, tr_y, cv_x, cv_y)

# Add carefree-learn tasks
for model in ["linear", "fcnn"]:
    experiment.add_task(model=model, data_folder=data_folder)
# Add scikit-learn tasks
run_command = f"python run_sklearn.py"
common_kwargs = {"run_command": run_command, "data_folder": data_folder}
experiment.add_task(model="decision_tree", **common_kwargs)
experiment.add_task(model="random_forest", **common_kwargs)
```

Notice that we specified `run_command="python run_sklearn.py"` for scikit-learn tasks, which means [`Experiment`](../user-guides/distributed#experiment) will try to execute this command in the current working directory for training scikit-learn models. The good news is that we do not need to speciy any command line arguments, because [`Experiment`](../user-guides/distributed#experiment) will handle those for us.

Here is basically what a `run_sklearn.py` should look like ([source code]([run_sklearn.py](https://github.com/carefree0910/carefree-learn/blob/dev/examples/iris/run_sklearn.py))):

```python
import os
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from cflearn.dist.runs._utils import get_info

if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    # data
    data_list = info.data_list
    x, y = data_list[:2]
    # model
    model = kwargs["model"]
    if model == "decision_tree":
        base = DecisionTreeClassifier
    elif model == "random_forest":
        base = RandomForestClassifier
    else:
        raise NotImplementedError
    sk_model = base()
    # train & save
    sk_model.fit(x, y.ravel())
    with open(os.path.join(info.workplace, "sk_model.pkl"), "wb") as f:
        pickle.dump(sk_model, f)
```

With `run_sklearn.py` defined, we could run those tasks with one line of code:

```python
results = experiment.run_tasks()
```

After finished running with this, we should be able to see the following file structure in the current working directory:

```text
|--- __experiment__
   |--- __data__
      |-- x.npy
      |-- y.npy
      |-- x_cv.npy
      |-- y_cv.npy
   |--- fcnn/0
      |-- _logs
      |-- __meta__.json
      |-- cflearn^_^fcnn^_^0000.zip
   |--- linear/0
      |-- ...
   |--- decision_tree/0
      |-- __meta__.json
      |-- sk_model.pkl
   |--- random_forest/0
      |-- ...
```

As we expected, `carefree-learn` models are saved into zip files, while scikit-learn models are saved into `sk_model.pkl` files. Since these models are not yet loaded, we should manually load them into our environment:

```python
import os
import pickle

pipelines = {}
sk_patterns = {}
for workplace, workplace_key in zip(results.workplaces, results.workplace_keys):
        model = workplace_key[0]
        if model not in ["decision_tree", "random_forest"]:
            pipelines[model] = cflearn.task_loader(workplace)
        else:
            model_file = os.path.join(workplace, "sk_model.pkl")
            with open(model_file, "rb") as f:
                sk_model = pickle.load(f)
                # In `carefree-learn`, we treat labels as column vectors.
                # So we need to reshape the outputs from the scikit-learn models.
                sk_predict = lambda x: sk_model.predict(x).reshape([-1, 1])
                sk_predict_prob = lambda x: sk_model.predict_proba(x)
                sk_pattern = cflearn.ModelPattern(
                    predict_method=sk_predict,
                    predict_prob_method=sk_predict_prob,
                )
                sk_patterns[model] = sk_pattern
```

After which we can finally perform benchmarking on these models:

```python
cflearn.evaluate(cv_x, cv_y, pipelines=pipelines, other_patterns=sk_patterns)
```

Which yields

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|     decision_tree      | -- 0.960000 -- | -- 0.000000 -- | -- 0.960000 -- | -- 0.998667 -- | -- 0.000000 -- | -- 0.998667 -- |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          | -- 0.960000 -- | -- 0.000000 -- | -- 0.960000 -- |    0.994133    | -- 0.000000 -- |    0.994133    |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.466667    | -- 0.000000 -- |    0.466667    |    0.725600    | -- 0.000000 -- |    0.725600    |
--------------------------------------------------------------------------------------------------------------------------------
|     random_forest      | -- 0.960000 -- | -- 0.000000 -- | -- 0.960000 -- | -- 0.998667 -- | -- 0.000000 -- | -- 0.998667 -- |
================================================================================================================================
```

Seems that scikit-learn models are better than `carefree-learn` models! This is not surprising because neural networks often require more data than traditional machine learning algorithms. However, we can boost `carefree-learn` models with AutoML, as shown in the next section.

## AutoML on Iris

As mentioned in the [`Introduction`](../), `carefree-learn` is actually a minimal Automatic Machine Learning (AutoML) solution for tabular datasets. Up till now we haven't mentioned any AutoML stuffs yet, so in this section we'll illustrate how to perform AutoML on Iris dataset, as well as how to pack the AutoML results into production.

Since `carefree-learn` has provided the [`cflearn.Auto`](../user-guides/auto-ml) API for out-of-the-box usages, AutoML in `carefree-learn` could be achieved in two lines of code:

```python
auto = cflearn.Auto("clf", models="fcnn")
auto.fit(tr_x, tr_y, cv_x, cv_y)
```

We can make predictions directly with `auto.predict`:

```python
predictions = auto.predict(cv_x)
print("accuracy:", (predictions == cv_y).mean())  # ~0.97
```

And of course, we can compare it with other models:

```python
all_patterns = sk_patterns.copy()
all_patterns["auto"] = auto.pattern
cflearn.evaluate(cv_x, cv_y, pipelines=pipelines, other_patterns=all_patterns)
```

Which yields

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          auto          | -- 0.986667 -- | -- 0.000000 -- | -- 0.986667 -- | -- 0.998933 -- | -- 0.000000 -- | -- 0.998933 -- |
--------------------------------------------------------------------------------------------------------------------------------
|     decision_tree      |    0.960000    | -- 0.000000 -- |    0.960000    |    0.998667    | -- 0.000000 -- |    0.998667    |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.960000    | -- 0.000000 -- |    0.960000    |    0.994133    | -- 0.000000 -- |    0.994133    |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.466667    | -- 0.000000 -- |    0.466667    |    0.725600    | -- 0.000000 -- |    0.725600    |
--------------------------------------------------------------------------------------------------------------------------------
|     random_forest      |    0.960000    | -- 0.000000 -- |    0.960000    |    0.998667    | -- 0.000000 -- |    0.998667    |
================================================================================================================================
```

Bravo! Our AutoML model beats the scikit-learn models 🥳

If we are satisfied with the results, we can pack the models up into a zip file

```python
# This will generate a pack.zip file.
auto.pack("pack")
```

which could be used on our production environments / machines easily:

```python
unpacked = cflearn.Auto.unpack("pack")
predictions = unpacked.pattern.predict(cv_x)
```


## Conclusion

Contained in this article is just a subset of the features that `carefree-learn` offers, but we've already walked through many basic & common steps we'll encounter in real life machine learning tasks.
