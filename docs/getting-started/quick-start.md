---
id: quick-start
title: Quick Start
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

In `carefree-learn`, it's both easy to train and serialize a model:

## Training

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
m = cflearn.make().fit(x, y)
# Make label predictions
m.predict(x)
# Make probability predictions
m.predict_prob(x)
# Estimate performance
cflearn.estimate(x, y, pipelines=m)
```

Then you will see something like this:

```text
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.946667    |    0.000000    |    0.946667    |    0.993200    |    0.000000    |    0.993200    |
================================================================================================================================
```

</TabItem>
<TabItem value="file">

`carefree-learn` can also easily fit / predict / estimate directly on files (**file-in, file-out**). Suppose we have an `xor.txt` file with following contents:

```text
0,0,0
0,1,1
1,0,1
1,1,0
```

Then `carefree-learn` can be utilized with only few lines of code:

> `delim` refers to 'delimiter', and `has_column_names` refers to whether the file has column names (or, header) or not.
> 
> Please refer to [carefree-data](https://github.com/carefree0910/carefree-data/blob/dev/README.md) if you're interested in more details.

```python
import cflearn
m = cflearn.make(delim=",", has_column_names=False).fit("xor.txt", x_cv="xor.txt")
# `contains_labels` is set to True because we're evaluating on training set
cflearn.estimate("xor.txt", pipelines=m, contains_labels=True)
```

After which you will see something like this:

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
print(m.predict([[0, 0]]))   # [[0]]
print(m.predict([[0, 1]]))   # [[1]]
print(m.predict("xor.txt", contains_labels=True))  # [ [0] [1] [1] [0] ]
```

</TabItem>
</Tabs>

## Serializing

It is also worth mentioning that `carefree-learn` models can be saved easily, into a zip file!

For example, a `cflearn^_^fcnn.zip` file will be created with one line of code:

```python
cflearn.save(m)
```

Of course, loading `carefree-learn` models are easy too!

```python
m = cflearn.load()
print(m)  # {'fcnn': FCNN()}
```
