---
id: general
title: General
---

:::tip
+ For configurations guide, please refer to the [Configurations](/docs/getting-started/configurations) section.
+ For development guide, please refer to the [Developer Guides](/docs/developer-guides/general-customization) section.
:::


## Introduction

In this section, we will introduce how to utilize `carefree-learn` to solve deep learning tasks in general.

Recall that [`Pipeline`](/docs/design-principles#pipeline) serves as the (internal) user interface in `carefree-learn`, so one of the main effort of utilizing `carefree-learn` will be how to construct a suitable [`Pipeline`](/docs/design-principles#pipeline).

:::info
+ Please refer to the [Configurations](/docs/getting-started/configurations) section for more details on how to construct a [Pipeline](/docs/design-principles#pipeline).
+ Please refer to the [Supported Models](/docs/getting-started/configurations#supported-models) section to see currently supported models.
:::

After a [`Pipeline`](/docs/design-principles#pipeline) is constructed, another effort will be how to define our dataset. Although different tasks require different data format, `carefree-learn` introduced `DLDataModule` to unify the APIs, which means we can always utilize `carefree-learn` in this way more or less:

```python
import cflearn

m = cflearn.api.xxx(...)  # construct `Pipeline`, based on your model
data = ...                # construct `DLDataModule`, based on your dataset
m.fit(data)               # train your model on your dataset!
```

:::info
+ Please refer to [MLData](machine-learning#mldata) section to see how to construct `DLDataModule` for ML tasks.
+ Please refer to [ImageFolderData](computer-vision#imagefolderdata) section to see how to construct `DLDataModule` for CV tasks.
:::

As shown above, `Pipeline` implements `fit` method to train models on datasets, which is similar to `scikit-learn`. Details of the high level APIs will be described in the following sections.

### `fit`

```python
def fit(
    self,
    data: DLDataModule,
    *,
    sample_weights: sample_weights_type = None,
    cuda: Optional[Union[int, str]] = None,
) -> DLPipeline:
```

+ **`data`**
    + `DLDataModule` constructed by our dataset.
+ **`sample_weights`** [default = `None`]
    + This is an experimental feature and is not fully supported.
+ **`cuda`**  [default = `None`]
    + Specify which `cuda` device we would like to train our models on.
    + If not provided, no `cuda` will be used and `cpu` will be used.

#### Example

```python
m.fit(data)
```

### `predict`

```python
def predict(
    self,
    data: DLDataModule,
    *,
    batch_size: int = 128,
    make_loader_kwargs: Optional[Dict[str, Any]] = None,
    **predict_kwargs: Any,
) -> np_dict_type:
```

+ **`data`**
    + `DLDataModule` constructed by our (new) dataset.
+ **`batch_size`** [default = `128`]
    + Specify the size of each batch we would like to use.
+ **`make_loader_kwargs`** [default = `{}`]
    + Specify some extra configurations we would like to use when constructing the `DataLoader`.
+ **`predict_kwargs`** [default = `{}`]
    + Specify some extra parameters we would like to use when running the forward pass.

#### Example

```python
# `predictions` will be a dictionary of np.ndarray
predictions = m.predict(data)
```

### `save`

```python
def save(
    self,
    export_folder: str,
    *,
    compress: bool = True,
    remove_original: bool = True,
) -> DLPipeline:
```

+ **`export_folder`**
    + Specify the export folder.
+ **`compress`** [default = `True`]
    + Specify whether should we compress the exported stuffs to a `.zip` file or not.
+ **`remove_original`** [default = `True`]
    + Specify whether should we remove the original folder after compressing or not.

#### Example

```python
# a `model.zip` file will be generated in the current working directory
m.save("model")
```

### `load`

```python
@staticmethod
def load(
    export_folder: str,
    *,
    cuda: Optional[Union[int, str]] = None,
    compress: bool = True,
    states_callback: states_callback_type = None,
    pre_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    post_callback: Optional[Callable[[DLPipeline, Dict[str, Any]], None]] = None,
) -> DLPipeline:
```

+ **`export_folder`**
    + Specify the export folder.
+ **`cuda`** [default = `None`]
    + Specify which `cuda` device we would like to load the models on.
    + If not provided, we will load the models to `cpu`
+ **`compress`** [default = `True`]
    + Specify whether the saved stuffs are compressed or not.
+ **`states_callback`** [default = `None`]
    + Specify the callback we would like to apply to the saved parameters.
+ **`pre_callback`** [default = `None`]
    + Specify the callback we would like to apply to the saved configurations.
+ **`post_callback`** [default = `None`]
    + Specify the callback we would like to apply after the `Pipeline` is loaded.

### `to_onnx`

```python
def to_onnx(
    self,
    export_folder: str,
    dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
    *,
    onnx_file: str = "model.onnx",
    opset: int = 11,
    simplify: bool = True,
    onnx_only: bool = False,
    forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    output_names: Optional[List[str]] = None,
    input_sample: Optional[tensor_dict_type] = None,
    num_samples: Optional[int] = None,
    compress: Optional[bool] = None,
    remove_original: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> DLPipeline:
```

+ **`export_folder`**
    + Specify the export folder.
+ **`dynamic_axes`** [default = `None`]
    + Specify the dynamic axes.
    + Notice that the first axis, which usually represents the batch size, should not be included here.
+ **`onnx_file`** [default = `"model.onnx"`]
    + Specify the name of the saved onnx file.
+ **`opset`** [default = `11`]
    + Specify the target opset version.
+ **`simplify`** [default = `True`]
    + Specify whether should we simplify the exported onnx file using [onnx-simplifier](https://github.com/daquexian/onnx-simplifier).
+ **`onnx_only`** [default = `False`]
    + Specify whether should we save all the information or not.
+ **`forward_fn`** [default = `None`]
    + If provided, we will replace the original forward pass with it when exporting to onnx.
+ **`output_names`** [default = `None`]
    + Specify the names of the outputs.
    + If not provided, `carefree-learn` will infer the correct ones automatically.
+ **`input_sample`** [default = `None`]
    + Specify the input sample for the forward pass.
    + If not provided, `carefree-learn` will utilize the training `DataLoader` to generate it.
        + However sometimes the training `DataLoader` may not exist, in which case we should provide the `input_sample` manually.
+ **`num_samples`** [default = `None`]
    + Specify whether should we use dynamic batch size or not.
    + If not provided, the exported onnx file will have dynamic batch size.
    + If provided, the exported onnx file will have fixed batch size (equals to `num_samples`).
        + In most cases, `num_samples` will be either `None` or `1`.
+ **`compress`** [default = `True`]
    + Specify whether should we compress the exported stuffs to a `.zip` file or not.
+ **`remove_original`** [default = `True`]
    + Specify whether should we remove the original folder after compressing or not.
+ **`verbose`** [default = `True`]
    + Specify if we want to log some messages to the console.
+ **`kwargs`** [default = `{}`]
    + Specify other keyword arguments we want to use in `torch.onnx.export`.
