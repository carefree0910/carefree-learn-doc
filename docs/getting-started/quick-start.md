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
m = cflearn.ml.CarefreePipeline().fit(x, y)
# Predict logits
m.predict(x)[cflearn.PREDICTIONS_KEY]
# Evaluate performance
cflearn.ml.evaluate(x, y, pipelines=m, metrics=["acc", "auc"])
```

This yields:

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.973333    |    0.000000    |    0.973333    |    0.999067    |    0.000000    |    0.999067    |
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
read_config = dict(delim=",", has_column_names=False)
m = cflearn.ml.CarefreePipeline(read_config=read_config)
m.fit(x_train, y_train, x_valid)
# `contains_labels` is set to True because we're evaluating on training set
cflearn.ml.evaluate("xor.txt", contains_labels=True, pipelines=m, metrics=["acc", "auc"])
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
print(m.predict([[0, 0]])[key].argmax(1))   # [0]
print(m.predict([[0, 1]])[key].argmax(1))   # [1]
print(m.predict("xor.txt", contains_labels=True)[key].argmax(1))  # [0 1 1 0]
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
# MNIST classification task with resnet18

import os
import cflearn

train_loader, valid_loader = cflearn.cv.get_mnist(transform="to_tensor")

m = cflearn.cv.CarefreePipeline(
    "clf",
    {
        "in_channels": 1,
        "num_classes": 10,
        "latent_dim": 512,
        "encoder1d": "backbone",
        "encoder1d_configs": {"name": "resnet18"},
    },
    loss_name="cross_entropy",
    metric_names="acc",
    fixed_epoch=1,  # For demo purpose
)
m.fit(train_loader, valid_loader)
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

In most cases, `carefree-learn` also supports a two-stage style serializing:
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

2. `carefree-learn` Pipelines could therefore 'pack' the corresponding (timestamp) folder into a zip file.

```python
# Notice that we should use the same `*Pipeline` as we use at training stage
base = cflearn.cv.CarefreePipeline
# A `packed.zip` file will be created under `_logs/2021-08-08_17-25-21-803661`
base.pack("_logs/2021-08-08_17-25-21-803661")
```

:::note
+ The **pack** procedure could be done 'individually', which means there are no dependencies between the **pack** procedure and the **training** procedure.
+ Machine Learning Pipeline may not always be able to do the same thing. To be exact, `cflearn.ml.SimplePipeline` supports the **pack** procedure, but `cflearn.ml.CarefreePipeline` doesn't. This is because `cflearn.ml.CarefreePipeline` contains some extra data structure (the `carefree-data` stuffs) which is not recorded in the `_logs` folder. In this case, we should use the `m.save` API to save all the necessary information.
:::

### Loading

Of course, loading `carefree-learn` models are easy too!

<Tabs
  defaultValue="ml"
  values={[
    {label: 'Machine Learning 📈', value: 'ml'},
    {label: 'Computer Vision 🖼️', value: 'cv'},
  ]
}>
<TabItem value="ml">

```python
m = cflearn.ml.CarefreePipeline.load("/path/to/zip_file")
```

</TabItem>
<TabItem value="cv">

```python
m = cflearn.cv.CarefreePipeline.load("/path/to/zip_file")
```

</TabItem>
</Tabs>

:::note
+ zip file from either `save` API or `pack` API can be loaded in this way.
+ Please refer to the [Production](../user-guides/production) section for production usages.
:::
