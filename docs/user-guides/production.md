---
id: production
title: Production
---

`carefree-learn` supports `onnx` export, but we need much more than one single model (`predictor`) in production environment:

![Pack](../../static/img/pack.png)

Fortunately, `carefree-learn` also supports exporting every part of this pipeline into a zip file with one line of code. Let's first train a simple model on `iris` dataset:

```python
import cflearn
from cfdata.tabular import TabularDataset

x, y = TabularDataset.iris().xy
m = cflearn.make().fit(x, y)
```

After which we can pack everything up with `cflearn.Pack` API:

```python
cflearn.Pack.pack(m, "pack")
```

This will generate a `pack.zip` in the working directory with following file structure:

```text
|--- preprocessor
   |-- ...
|--- binary_config.json
|--- m.onnx
|--- output_names.json
|--- output_probabilities.txt
```

We can make inference with `pack.zip` on our production environments / machines easily:

```python
import cflearn

predictor = cflearn.Pack.get_predictor("pack")
predictions = predictor.predict(x)
```
