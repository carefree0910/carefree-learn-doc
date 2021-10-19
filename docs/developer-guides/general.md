---
id: general-customization
title: General
---

In general, in order to solve our own tasks with our own models in `carefree-learn`, we need to concern:
+ How to define a new model & How to use it for training.
+ How to customize pre-processings of the dataset.
+ How to control some fine-grained behaviours of the training loop.

In this section, we will focus on the general customizations.

:::tip
+ See [here](computer-vision-customization) for customizations of Computer Vision 🖼️.
+ See [here](machine-learning-customization) for customizations of Machine Learning 📈.
:::


## Customize Models

In `carefree-learn`, a `Model` should implement the core algorithms. It's basically an `nn.Module`, with some extra useful functions:

```python
class ModelProtocol(nn.Module, WithRegister, metaclass=ABCMeta):
    d = model_dict

    @property
    def device(self) -> torch.device:
        return list(self.parameters())[0].device

    def onnx_forward(self, batch: tensor_dict_type) -> Any:
        return self.forward(0, batch)

    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
        self.forward(batch_idx, batch)
    
    def _init_with_trainer(self, trainer: Any) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass
```

As shown above, there are two special `forward` methods defined in a `Model`, which allows us to customize `onnx` export procedure and `summary` procedure respectively.

If we want to define our own models, we will need to override the `forward` method (required) and the `_init_with_trainer` method (optional).

### `forward`

```python
def forward(
    self,
    batch_idx: int,
    batch: tensor_dict_type,
    state: Optional["TrainerState"] = None,
    **kwargs: Any,
) -> tensor_dict_type:
    pass
```

+ **`batch_idx`**
    + Indicates the batch index of current batch.
+ **`batch`**
    + Input batch. It will be a dictionary (`Dict[str, torch.Tensor]`) returned by `DataLoader`.
    + In general, it will:
        + always contain an `"input"` key, which represents the input data.
        + usually contain a `"labels"` key, which represents the target labels.
      
      Other constants could be found [here](https://github.com/carefree0910/carefree-learn/blob/99c946ffa1df2b821161d52aae19f67e91abf46e/cflearn/constants.py).
+ **`state`** [default = `None`]
    + The [`TrainerState`](../getting-started/configurations#trainerstate) instance.
+ **`kwargs`**
    + Other keyword arguments.

### `_init_with_trainer`

This is an optional method, which is useful when we need to initialize our models with the prepared `Trainer` instance.

:::tip
Since the prepared `Trainer` instance will contain the dataset information, this method will be very useful if our models depend on the information.
:::

### Register & Apply

After defining the `forward` (and probably the `_init_with_trainer`) method, we need to [register](../design-principles#register-mechanism) our model to apply it in `carefree-learn`:

```python
@ModelProtocol.register("my_fancy_model")
class MyFancyModel(ModelProtocol):
    def __init__(self, foo):
        super().__init__()
        self.foo = foo

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        ...
```

After which we can:
+ set the `model_name` in [`Pipeline`](../getting-started/configurations#dlsimplepipeline) to the corresponding name to apply it.
+ set the `model_config` in [`Pipeline`](../getting-started/configurations#dlsimplepipeline) to the corresponding configurations.

```python
m = cflearn.cv.CarefreePipeline("my_fancy_model", {"foo": "bar"})
m.build({})
print(m.model.foo)  # bar
```

:::note
For Machine Learning tasks, the APIs will remain the same but the internal design will be a little different. Please refer to the [`MLModel`](machine-learning-customization#mlmodel) section for more details.
:::


## Customize Training Loop

:::caution
To be continued...
:::
