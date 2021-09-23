---
id: quick-start
title: Quick Start
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

In `carefree-learn`, it's easy to train and serialize a model **for all tasks**.


## Training

### Machine Learning 📈

<Tabs
  defaultValue="numpy"
  values={[
    {label: 'With NumPy', value: 'numpy'},
    {label: 'With File', value: 'file'},
  ]
}>
<TabItem value="numpy">

```python
import cflearn
from cfdata.tabular import TabularDataset

x, y = TabularDataset.iris().xy
m = cflearn.api.fit_ml(x, y, carefree=True)
# Predict logits
inference_data = cflearn.ml.MLInferenceData(x, y)
m.predict(inference_data)[cflearn.PREDICTIONS_KEY]
# Evaluate performance
cflearn.ml.evaluate(inference_data, pipelines=m, metrics=["acc", "auc"])
```

This yields:

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.980000    |    0.000000    |    0.980000    |    0.998933    |    0.000000    |    0.998933    |
================================================================================================================================
```

</TabItem>
<TabItem value="file">

`carefree-learn` can also easily fit / predict / evaluate directly on files (**file-in, file-out**). Suppose we have an `xor.txt` file with following contents:

```text
0,0,0
0,1,1
1,0,1
1,1,0
```

Then `carefree-learn` can be utilized with only few lines of code:

```python
import cflearn

y_train = None
x_train = x_valid = "xor.txt"
args = x_train, y_train, x_valid
data_config = {"read_config": dict(delim=",", has_column_names=False)}
m = cflearn.api.fit_ml(*args, carefree=True, data_config=data_config)
# `contains_labels` is set to True because we're evaluating on training set
inference_data = cflearn.ml.MLInferenceData("xor.txt")
cflearn.ml.evaluate(inference_data, contains_labels=True, pipelines=m, metrics=["acc", "auc"])
```

:::info
+ `delim` refers to '**delimiter**', and `has_column_names` refers to whether the file has column names (or, header) or not.
+ Please refer to [carefree-data](https://github.com/carefree0910/carefree-data/blob/dev/README.md) if you're interested in more details.
:::

This yields:

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    1.000000    |    0.000000    |    1.000000    |    1.000000    |    0.000000    |    1.000000    |
================================================================================================================================
```

When we fit from files, we can predict on either files or lists:

```python
key = cflearn.PREDICTIONS_KEY
data_base = cflearn.ml.MLInferenceData
print(m.predict(data_base([[0, 0]]))[key].argmax(1))   # [0]
print(m.predict(data_base([[0, 1]]))[key].argmax(1))   # [1]
print(m.predict(data_base("xor.txt"), contains_labels=True)[key].argmax(1))  # [0 1 1 0]
```

</TabItem>
</Tabs>


### Computer Vision 🖼️

<Tabs
  defaultValue="preset"
  values={[
    {label: 'Preset (torchvision) Dataset', value: 'preset'},
    {label: 'Custom Image Folder Dataset', value: 'custom'},
  ]
}>
<TabItem value="preset">

```python
import cflearn

data = cflearn.cv.MNISTData(transform="to_tensor")
m = cflearn.api.resnet18_gray(10)
# m.fit(data, cuda=0)  # If CUDA available
# m.fit(data)          # If not
```

</TabItem>
<TabItem value="custom">

> This is a WIP section :D

</TabItem>
</Tabs>


## Serializing

### Saving

`carefree-learn` models can be saved easily, into a zip file (for both ml & cv tasks) !

```python
m.save("model")  # a `model.zip` file will be created
```

It's worth mentioning that `carefree-learn` supports a two-stage style serializing:
1. A `_logs` folder (with timestamps as its subfolders) will be created after training.

```text
--- _logs
 |-- 2021-08-08_16-00-24-175005
  |-- checkpoints
  |-- configs.json
  |-- metrics.txt
  ...
 |-- 2021-08-08_17-25-21-803661
  |-- checkpoints
  |-- configs.json
  |-- metrics.txt
  ...
```

2. `carefree-learn` could therefore 'pack' the corresponding (timestamp) folder into a zip file.

```python
cflearn.api.pack("_logs/2021-08-08_17-25-21-803661")
```

:::note
This `pack` API is a '**unified**' API, which means you can use it to serialize either Machine Learning models or Computer Vision models!
:::

### Loading

Of course, loading `carefree-learn` pipelines are easy as well!

```python
m = cflearn.api.load("/path/to/your/zip/file")
```

:::note
+ This is also a '**unified**' API.
+ zip file exported from either `save` API or `pack` API can be loaded in this way.
+ Please refer to the [Production](../user-guides/production) section for production usages.
:::
