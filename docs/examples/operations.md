---
id: Operations
title: Operations
---

| Python source code | Jupyter Notebook |
|:---:|:---:|
| [op.py](https://github.com/carefree0910/carefree-learn/blob/dev/examples/operations/op.py) | [op.ipynb](https://nbviewer.jupyter.org/github/carefree0910/carefree-learn/blob/dev/examples/operations/op.ipynb) |

`Operations` are toy datasets for us to illustrate how to build your own models in `carefree-learn`. We will generate some artificial datasets based on basic *operations*, namely `sum`, `prod` and their mixture, to deonstrate the validity of our customized model.

Here are the formula of the definitions of the datasets:

$$
\begin{aligned}
\mathcal{D}_{\text {sum}} &=\{(\mathbf x,\sum_{i=1}^d x_i)|\mathbf x\in\mathbb{R}^d\} \\
\mathcal{D}_{\text {prod}}&=\{(\mathbf x,\prod_{i=1}^d x_i)|\mathbf x\in\mathbb{R}^d\} \\
\mathcal{D}_{\text {mix}} &=\{(\mathbf x,[y_{\text{sum}},y_{\text{prod}}])|\mathbf x\in\mathbb{R}^d\}
\end{aligned}
$$

In short, the `sum` dataset simply sums up the features, the `prod` dataset simply multiplies all the features, and the `mix` dataset is a mixture of `add` and `prod`. Here are the codes to generate them:

```python
import numpy as np

# prepare
dim = 5
num_data = 10000

x = np.random.random([num_data, dim]) * 2.0
y_add = np.sum(x, axis=1, keepdims=True)
y_prod = np.prod(x, axis=1, keepdims=True)
y_mix = np.hstack([y_add, y_prod])
```

Since we want to hold the datasets' property, we should not apply any pre-processing strategies to these datasets. Fortunately, `carefree-learn` has provided a simple configuration for us to do so:

```python
# `reg` represents a regression task
# `use_simplify_data` indicates that `carefree-learn` will do nothing to the input data
kwargs = {"task_type": "reg", "use_simplify_data": True}
```


## The `add` Dataset

It's pretty clear that the `add` dataset could be solved easily with a `linear` model

$$
\hat y = wx + b,\quad w\in\mathbb{R}^{1\times d},b\in\mathbb{R}^{1\times 1}
$$

because the *ground truth* of `add` dataset could be represented as `linear` model, where

$$
w=[1,1,...,1],b=[0]
$$

Although this is a simple task, using Neural Networks to solve it might actually fail because it is likely to overfit the training set with some strange representation. We can demonstrate it by lifting a simple, quick experiment with the help of `carefree-learn`:

```python
import cflearn

linear = cflearn.make("linear", **kwargs).fit(x, y_add)
fcnn = cflearn.make("fcnn", **kwargs).fit(x, y_add)
cflearn.evaluate(x, y_add, pipelines=[linear, fcnn])
```

Which yields

```text
================================================================================================================================
|        metrics         |                       mae                        |                       mse                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.346006    | -- 0.000000 -- |    -0.34600    |    0.154638    | -- 0.000000 -- |    -0.15463    |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         | -- 0.000418 -- | -- 0.000000 -- | -- -0.00041 -- | -- 0.000000 -- | -- 0.000000 -- | -- -0.00000 -- |
================================================================================================================================
```

As we expected, the `fcnn` (Fully Connected Neural Network) model actually fails to reach a satisfying result, while the `linear` model approaches to the ground truth easily.

We can also check whether the model has *actually* learned the ground truth by checking its parameters ($w$ and $b$):

```python
linear_core = linear.model.heads["linear"].linear
print(f"w: {linear_core.weight.data}")
print(f"b: {linear_core.bias.data}")
```

Which yields

```text
w: tensor([[0.9998, 0.9999, 1.0002, 0.9997, 1.0000]], device='cuda:0')
b: tensor([-0.0001], device='cuda:0')
```

It's not perfect, but we are happy enough😆


## The `prod` Dataset

However, when it comes to the `prod` dataset, the `linear` model is likely to face the *underfitting* issue because theoratically it cannot represent such formulation:

$$
y=\prod_{i=1}^{d}x_i
$$

Neural Networks, on the other side, are able to represent **ANY** functions ([Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)). In this case, the `fcnn` model should be able to outperform the `linear` model:

```python
linear = cflearn.make("linear", **kwargs).fit(x, y_prod)
fcnn = cflearn.make("fcnn", **kwargs).fit(x, y_prod)
cflearn.evaluate(x, y_prod, pipelines=[linear, fcnn])
```

Which yields

```text
================================================================================================================================
|        metrics         |                       mae                        |                       mse                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          | -- 0.167043 -- | -- 0.000000 -- | -- -0.16704 -- | -- 0.152063 -- | -- 0.000000 -- | -- -0.15206 -- |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.955200    | -- 0.000000 -- |    -0.95520    |    2.798824    | -- 0.000000 -- |    -2.79882    |
================================================================================================================================
```

Although `fcnn` outperforms `linear`, it is still not as satisfied as the results that we've got in `add` dataset. That's because although `fcnn` has strong approximation power, its representations are basically based on the `add` operations between features, and the non-linearities come from an activation function applied to **EACH** neuron. Which means, `fcnn` can hardly learn `prod` operation **ACROSS** features.

A trivial thought is to manually extract the `prod` features $\tilde x$ from the input data with a new `extractor`:

$$
\tilde x\triangleq \prod_{i=1}^d x_i
$$

After which a `linear` model should solve the problem, because the *ground truth* here is simply

$$
w=[1], b=[0]
$$

But how could we apply this prior knowledge to our model? Thanks to `carefree-learn`, this is actually quite simple with only a few lines of codes:

```python
import torch

# register an `extractor` which represents the `prod` operation
@cflearn.register_extractor("prod_extractor")
class ProdExtractor(cflearn.ExtractorBase):
    @property
    def out_dim(self) -> int:
        return 1

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return net.prod(dim=1, keepdim=True)


# define the `Config` for this `extractor`
# since `ProdExtractor` don't need any configurations, we can simply return an empty dict here
cflearn.register_config("prod_extractor", "default", config={})
```

:::tip
If you are interested in how does `extractor` actually work in `carefree-learn`, please refer to [pipe](../design-principles#pipe) and [extractor](../design-principles#extractor) for more information.
:::

After defining the `extractor`, we need to define a model that leverages it:

```python
# we call this new model `prod`
# we use our new `extractor` followed by traditional `linear` model
cflearn.register_model(
    "prod",
    pipes=[cflearn.PipeInfo("linear", extractor="prod_extractor")],
)
```

And that's it! We can now train our new model and evaluate it:

```python
prod = cflearn.make("prod", **kwargs).fit(x, y_prod)
cflearn.evaluate(x, y_prod, pipelines=[linear, fcnn, prod])
```

Which yields

```text
================================================================================================================================
|        metrics         |                       mae                        |                       mse                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.167043    | -- 0.000000 -- |    -0.16704    |    0.152063    | -- 0.000000 -- |    -0.15206    |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.955200    | -- 0.000000 -- |    -0.95520    |    2.798824    | -- 0.000000 -- |    -2.79882    |
--------------------------------------------------------------------------------------------------------------------------------
|          prod          | -- 0.000143 -- | -- 0.000000 -- | -- -0.00014 -- | -- 0.000000 -- | -- 0.000000 -- | -- -0.00000 -- |
================================================================================================================================
```

As we expected, the `prod` model approaches to the ground truth easily.

We can also check whether the model has actually learned the ground truth by checking its parameters ($w$ and $b$):

```python
prod_linear = prod.model.heads["linear"].linear
print(f"w: {prod_linear.weight.item():8.6f}, b: {prod_linear.bias.item():8.6f}")
```

Which yields

```text
w: 1.000136, b: -0.000113
``` 

It's not perfect, but we are happy enough😆


## The `mix` Dataset

Now comes to the fun part: what if we mix up `add` and `prod` dataset? Since `linear` is professional in `add`, `prod` is professional in `prod`, and `fcnn` is **QUITE** professional in **ALL** datasets (🤣), it is hard to tell which one will outshine in the `mix` dataset. So let's do an experiment to obtain an empirical conclusion:

```python
linear = cflearn.make("linear", **kwargs).fit(x, y_mix)
fcnn = cflearn.make("fcnn", **kwargs).fit(x, y_mix)
prod = cflearn.make("prod", **kwargs).fit(x, y_mix)
cflearn.evaluate(x, y_mix, pipelines=[linear, fcnn, prod])
```

Which yields

```text
================================================================================================================================
|        metrics         |                       mae                        |                       mse                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          | -- 0.158854 -- | -- 0.000000 -- | -- -0.15885 -- | -- 0.094512 -- | -- 0.000000 -- | -- -0.09451 -- |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.341526    | -- 0.000000 -- |    -0.34152    |    1.104427    | -- 0.000000 -- |    -1.10442    |
--------------------------------------------------------------------------------------------------------------------------------
|          prod          |    0.341261    | -- 0.000000 -- |    -0.34126    |    0.435841    | -- 0.000000 -- |    -0.43584    |
================================================================================================================================
```

Seems that the non-expert in both domain (`fcnn`) outperforms the domain experts (`linear`, `prod`)! But again, this is far from satisfying because theoratically we can combine the domain experts to build an expert in `mix` dataset.

Thanks to `carefree-learn`, we again can actually do so, but this time we'll need some more coding. Recall that we build an expert in `prod` dataset by defining a novel `extractor`, because we needed to pre-process the input data. However in `mix`, what we actually need is to combine `linear` and `prod`, which means we need to define a novel `head` this time.

:::tip
If you are interested in how does `head` actually work in `carefree-learn`, please refer to [pipe](../design-principles#pipe) and [head](../design-principles#head) for more information.
:::

Concretely, suppose we already have two models, $f_1$ and $f_2$, that are experts in `add` dataset and `prod` dataset respectively. What we need to do is to combine the first dimension of $f_1(\mathbf x)$ and the second dimension of $f_2(\mathbf x)$ to construct our final outputs:

$$
\begin{aligned}
f_1(\mathbf x) \triangleq [\hat y_{11}, \hat y_{12}] \\
f_2(\mathbf x) \triangleq [\hat y_{21}, \hat y_{22}] \\
\Rightarrow \tilde f(\mathbf x) \triangleq [\hat y_{11}, \hat y_{22}]
\end{aligned}
$$

Since $\hat y_{11}$ can fit `add` dataset perfectly, $\hat y_{22}$ can fit `prod` dataset perfectly, $\tilde f(\mathbf x)$ should be able to fit `mix` dataset perfectly. Let's implement this model to demonstrate it with experiment:

```python
from cflearn.modules.blocks import Linear

@cflearn.register_head("mixture")
class MixtureHead(cflearn.HeadBase):
    def __init__(self, in_dim: int, out_dim: int, target_dim: int):
        super().__init__(in_dim, out_dim)
        # when `target_dim == 0`, it represents an `add` head (y_11)
        # when `target_dim == 1`, it represents a `prod` head (y_22)
        self.dim = target_dim
        self.linear = Linear(in_dim, 1)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        target = self.linear(net)
        zeros = torch.zeros_like(target)
        tensors = [target, zeros] if self.dim == 0 else [zeros, target]
        return torch.cat(tensors, dim=1)

# we need to define two configurations for `add` and `prod` respectively    
cflearn.register_head_config("mixture", "add", head_configs={"target_dim": 0})
cflearn.register_head_config("mixture", "prod", head_configs={"target_dim": 1})

# we use our new `head` to define the new model
# note that we need two `pipe`s, one for `add` and the other for `prod`
cflearn.register_model(
    "mixture",
    pipes=[
        cflearn.PipeInfo(
            "add",
            extractor="identity",
            head="mixture",
            head_config="add",
        ),
        cflearn.PipeInfo(
            "prod",
            extractor="prod_extractor",
            head="mixture",
            head_config="prod",
        )
    ]
)

mixture = cflearn.make("mixture", **kwargs).fit(x, y_mix)
cflearn.evaluate(x, y_mix, pipelines=[linear, fcnn, prod, mixture])
```

Which yields

```text
================================================================================================================================
|        metrics         |                       mae                        |                       mse                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.158854    | -- 0.000000 -- |    -0.15885    |    0.094512    | -- 0.000000 -- |    -0.09451    |
--------------------------------------------------------------------------------------------------------------------------------
|         linear         |    0.341526    | -- 0.000000 -- |    -0.34152    |    1.104427    | -- 0.000000 -- |    -1.10442    |
--------------------------------------------------------------------------------------------------------------------------------
|        mixture         | -- 0.000219 -- | -- 0.000000 -- | -- -0.00021 -- | -- 0.000000 -- | -- 0.000000 -- | -- -0.00000 -- |
--------------------------------------------------------------------------------------------------------------------------------
|          prod          |    0.341261    | -- 0.000000 -- |    -0.34126    |    0.435841    | -- 0.000000 -- |    -0.43584    |
================================================================================================================================
```

As we expected, the `mixture` model approaches to the ground truth easily.

We can also check whether the model has actually learned the ground truth by checking its parameters ($w$ and $b$):

```python
add_linear = mixture.model.heads["add"].linear
prod_linear = mixture.model.heads["prod"].linear
print(f"add  w: {add_linear.weight.data}")
print(f"add  b: {add_linear.bias.data}")
print(f"prod w: {prod_linear.weight.data}")
print(f"prod b: {prod_linear.bias.data}")
```

Which yields

```text
add  w: tensor([[1.0002, 0.9999, 1.0000, 1.0000, 0.9999]], device='cuda:0')
add  b: tensor([-0.0002], device='cuda:0')
prod w: tensor([[1.0001]], device='cuda:0')
prod b: tensor([-0.0002], device='cuda:0')
```

It's not perfect, but we are happy enough🥳


## Conclusions

`Operations` are just artificial toy datasets, but quite handy for us to illustrate some basic concepts in `carefre-learn`. We hope that this small example can help you quickly walk through some development guides in `carefre-learn`, and help you leverage `carefree-learn` in your own tasks!
