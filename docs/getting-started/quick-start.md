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

For custom image folder dataset, `carefree-learn` provides a `Preparation`-style API for you to prepare your data. In this demo, we'll show you how to use it for image classification tasks. Suppose we have the following file structure:

```text
|--- data
   |--- labels.csv
   |--- images
      |-- 0.png
      |-- 1.png
      |-- 2.png
      |-- 3.png
```

Where `labels.csv` file contains the labels of each image:

```text
0.png,positive
1.png,negative
2.png,positive
3.png,negative
```

Then we should define a `Preparation`, which tells `carefree-learn` how to interpret the data:

```python
import os
import cflearn

class DemoPreparation(cflearn.DefaultPreparation):
    def __init__(self):
        self.labels = {}
        with open("data/labels.csv", "r") as f:
            for line in f:
                k, v = line.strip().split(",")
                self.labels[k] = v

    def get_label(self, hierarchy):
        """
        `hierarchy` is a list of string, representing the file hierarchy.
        For instance, the `hierarchy` of 0.png will be ["data", "images", "0.png"]
        """
        return self.labels[hierarchy[-1]]


rs = cflearn.prepare_image_folder_data(
    "data",
    "data/gathered",  # This is where you want to put the prepared dataset
    to_index=True,    # We should turn the original labels ('positive', 'negative') to integer values
    batch_size=1,
    preparation=DemoPreparation(),
    transform="to_tensor",
)
```

It's worth mentioning that `carefree-learn` will automatically achieve many common practices for you, such as:
- Split out validation set properly.
- Save the mappings between indices and original labels to some `json` files.

:::note
In addition, for classification tasks, `carefree-learn` will ensure that:
- The class distribution of validation dataset is the same as the one of training dataset.
- Validation dataset has at least one sample per class.
:::

The '**prepared**' file structure will be as follows:

```text
|--- data
   |--- gathered
      |--- train
         |-- 0.png 
         |-- 3.png
         |-- labels.json
         ...
      |--- valid
         |-- 1.png 
         |-- 2.png
         |-- labels.json
         ...
      |-- idx2labels.json
      |-- labels2idx.json
   ...
```

Where

```json
{"0": "negative", "1": "positive"}  // idx2labels.json
{"negative": 0, "positive": 1}      // labels2idx.json

{"/absolute/path/to/0.png": 1, "/absolute/path/to/3.png": 0}  // ./train/labels.json
{"/absolute/path/to/2.png": 1, "/absolute/path/to/1.png": 0}  // ./valid/labels.json
```

After the data is prepared, we can define a model to fit it, which is fairly easy for `carefree-learn`:

```python
m = cflearn.api.resnet18(
    2,              # We have two classes
    fixed_steps=1,  # For demo purpose, we only train the model for one step
)
m.fit(rs.data)
```

</TabItem>

</Tabs>


## Serializing

### Saving

`carefree-learn` pipelines can be saved easily, into a zip file (for both ml & cv tasks) !

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
This `pack` API is a '**unified**' API, which means you can use it to serialize either Machine Learning pipelines or Computer Vision pipelines!
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
