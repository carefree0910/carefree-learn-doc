---
id: design-principles
title: Design Principles
---

`carefree-learn` was designed to support most commonly used methods with 'carefree' APIs. Moreover, `carefree-learn` was also designed with interface which is general enough, so that more sophisticated functionality can also be easily integrated in the future. This brings a tension in how to create abstractions in code, which is a challenge for us:

+ On the one hand, it requires a reasonably high-level abstraction so that users can easily work around with it in a standard way, without having to worry too much about the details.
+ On the other hand, it also needs to have a very thin abstraction to allow users to do (many) other things in new ways. Breaking existing abstractions and replacing them with new ones should be fairly easy.

In `carefree-learn`, there are five main design principles that address this tension together:

+ Share configurations with the help of `Environment` (see [`Configurations`](getting-started/configurations#environment)).
+ Build some common blocks which shall be leveraged across different models (see [`Common Blocks`](#common-blocks)).
+ Divide `carefree-learn` into three parts: [`Model`](#model), [`Trainer`](#trainer) and [`Pipeline`](#pipeline), each focuses on certain roles.
+ Divide `Model` into three parts: [`transform`](#transform), [`extract`](#extract) and [`head`](#head), each focuses on one part of a [`pipe`](#pipe)
+ Implemente functions (`cflearn.register_*` to be exact) to ensure flexibility and control on different modules and stuffs (see [`Registration`](#registration)).

We will introduce the details in the following subsections.

### Common Blocks

> Source code: [blocks.py](https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/modules/blocks.py).

Commonality is important for abstractions. When it comes to deep learning, it is not difficult to figure out the very common structure across all models: the `Mapping` Layers which is responsible for mapping data distrubution from one dimensional to another.

Although some efforts have been made to replace the `Mapping` Layers (e.g. DNDF[^1]), it is undeniable that the `Mapping` Layers should be extracted as a stand-alone module before any other structures. But in `carefree-learn`, we should do more than that.

Recall that `carefree-learn` focuses on tabular datasets, which means `carefree-learn` will use `Mapping` Layers in most cases (Unlike CNN or RNN which has Convolutional Blocks and RNNCell Blocks respectively). In this case, it is necessary to wrap multiple `Mapping`s into one single Module - also well known as `MLP` - in `carefree-learn`. So, in CNN we have `Conv2D`, in RNN we have `LSTM`, and in `carefree-learn` we have `MLP`.

### Model

> Source code: [`ModelBase`](https://github.com/carefree0910/carefree-learn/blob/ecdae9702456910b5075d1972de66a4f64ea733a/cflearn/models/base.py#L87).

In `carefree-learn`, a `Model` should implement the core algorithms.

+ It assumes that the input data in training process is already 'batched, processed, nice and clean', but not yet 'encoded'.
    + Fortunately, `carefree-learn` pre-defined some useful methods which can encode categorical columns easily and **EFFICIENTLY** (see [`Optimizations`](optimizations))
+ It does not care about how to train a model, it only focuses on how to make predictions with input, and how to calculate losses with them.

:::note
`Model` is likely to define `MLP` blocks frequently, as explained in the [`Common Blocks`](#common-blocks) section.
:::

One thing we would like to proudly announce is that `carefree-learn` has made an elegant abstraction, namely `pipe`, which is suitable for most of the models which aim at solving tabular tasks:

#### Pipe

Unlike unstructured datasets (CV, NLP, etc), it's hard to inject our prior knowledge into structured datasets because in most cases we simply use `MLP` to solve the problem. Researchers therefore mainly focused on how the improve the 'inputs' and the 'connections' of the traditional fully-connected ones. Some famous models, such as Wide-and-Deep[^2], Deep-and-Cross[^3], DeepFM[^4], share this common pattern. `carefree-learn` therefore defined `pipe`, which corresponds to one of those 'branches' which takes in all / part of the inputs, apply some `transform`s, `extract` some features, and then feed the final network (`head`) with these features. Here's an example:

![Pipe](../static/img/pipe.png)

In this model, we have $3$ `pipe`s. The first `pipe` takes in $x_1, x_2$, the second one takes in $x_3$, and the third one takes in $x_3, x_4$.



### Trainer

:::info
+ Source codes path: [core.py](https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/trainer/core.py) -> `class Trainer`.
:::

In `carefree-learn`, a `Trainer` should implement the high-level parts, as listed below:

+ It assumes that the input data is already 'processed, nice and clean', but it should take care of getting input data into batches, because in real applications batching is essential for performance.
+ It should take care of the training loop, which includes updating parameters with an optimizer, verbosing metrics, checkpointing, early stopping, logging, etc.

### Pipeline

:::info
+ Source codes path: [core.py](https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/pipeline/core.py) -> `class Pipeline`.
:::

In `carefree-learn`, a `Pipeline` should implement the preparation and API part.

+ It should not make any assumptions to the input data, it could already be 'nice and clean', but it could also be 'dirty and messy'. Therefore, it needs to transform the original data into 'nice and clean' data and then feed it to `Trainer`. The data transformations include (this part is mainly handled by [`carefree-data`](https://github.com/carefree0910/carefree-data), though):
    + Imputation of missing values.
    + Transforming string columns into categorical columns.
    + Processing numerical columns.
    + Processing label column (if needed).
+ It should implement some algorithm-agnostic functions (e.g. `predict`, `save`, `load`, etc.).

### Registration

Registration in `carefree-learn` means registering user-defined modules to `carefree-learn`, so `carefree-learn` can leverage these modules to resolve more specific tasks. In most cases, the registration stuffs are done by simply defining and updating many global `:::python dict`s.

For example, `carefree-learn` defined some useful parameter initializations in `cflearn.misc.toolkit.Initializer`. If we want to use our own initialization methods, simply register it and then everything will be fine:

```python
import torch
import cflearn

from torch.nn import Parameter

initializer = cflearn.Initializer({})

@cflearn.register_initializer("all_one")
def all_one(initializer_, parameter):
    parameter.fill_(1.)

param = Parameter(torch.zeros(3))
with torch.no_grad():
    initializer.initialize(param, "all_one")
print(param)  # tensor([1., 1., 1.], requires_grad=True)
```

:::info
Currently we mainly have 5 registrations in use: `register_metric`, `register_optimizer`, `register_scheduler`, `register_initializer` and `register_processor`
:::


[^1]: Kontschieder P, Fiterau M, Criminisi A, et al. Deep neural decision forests[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1467-1475. 
[^2]: Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st workshop on deep learning for recommender systems. 2016: 7-10. 
[^3]: Wang, Ruoxi, et al. “Deep & cross network for ad click predictions.” Proceedings of the ADKDD’17. 2017. 1-7. 
[^4]: Guo, Huifeng, et al. “Deepfm: An end-to-end wide & deep learning framework for CTR prediction.” 
