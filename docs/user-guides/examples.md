---
id: examples
title: Examples
---

In this page we'll introduce how to utilize `carefree-learn` in some real life tasks.

:::caution
This page is a work in progress.
:::


## Titanic

| Python source code | Jupyter Notebook |
|:---:|:---:|
| [titanic.py](https://github.com/carefree0910/carefree-learn/blob/dev/examples/titanic/titanic.py) | [titanic.ipynb](https://github.com/carefree0910/carefree-learn/blob/dev/examples/titanic/titanic.ipynb) |

`Titanic` is a famous playground competition hosted by Kaggle ([here](https://www.kaggle.com/c/titanic)), so I'll simply copy-paste its brief description here:

> This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works.
> 
> The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

Here are the frist few rows of the `train.csv` of `Titanic`:

```csv
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
```

And the first few rows of the `test.csv`:

```csv
PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
892,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q
893,3,"Wilkes, Mrs. James (Ellen Needs)",female,47,1,0,363272,7,,S
894,2,"Myles, Mr. Thomas Francis",male,62,0,0,240276,9.6875,,Q
```

What we need to do is to predict the `Survived` column in `test.csv`.

### Configurations

Since the target column is not the last column (which is the default setting of `carefree-learn`), we need to manually configure it:

```python
data_config = {"label_name": "Survived"}
```

And you're all set! Notice that only the `label_name` needs to be provided, and `carefree-learn` will find out the corresponding target column for you😉

### Build Your Model

For instance, we'll use the famous `Wide & Deep` model:

```python
import cflearn

m = cflearn.make("wnd", data_config=data_config)
```

Unlike other libraries, `carefree-learn` supports *file-in*:

```python
m.fit("train.csv")
```

### Evaluate Your Model

After building the model, we can directly evaluate our model with a file (*file-out*):

```python
# `contains_labels` is set to True because we're evaluating on training set
cflearn.evaluate("train.csv", pipelines=m, contains_labels=True)
```

Then you will see something like this:

```text
~~  [ info ] Results
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          wnd           |    0.860831    |    0.000000    |    0.860831    |    0.915396    |    0.000000    |    0.915396    |
================================================================================================================================
```

Our model achieved an accuracy of `0.860831`, not bad!

:::info
Note that this performance may vary and is not exactly the *training* performance, because `carefree-learn` will automatically split out the cross validation dataset for you. Please refer to [cv_split](../getting-started/configurations#cv_split) for more details.
:::

### Making Predictions

Again, we can directly make predictions with a file (*file-out*):

```python
# `contains_labels` is set to False because `test.csv` does not contain labels
# It is OK to simply call `m.predict("test.csv")` because `contains_labels` is False by default
predictions = m.predict("test.csv", contains_labels=False)
```

### Submit Your Results

If you reached here, we have actually already completed this `Titanic` task! All we need to do is to convert the `predictions` into a submission file:

```python
def write_submissions(name, predictions_) -> None:
    with open("test.csv", "r") as f:
        f.readline()
        id_list = [line.strip().split(",")[0] for line in f]
    with open(name, "w") as f:
        f.write("PassengerId,Survived\n")
        for test_id, prediction in zip(id_list, predictions_.ravel()):
            f.write(f"{test_id},{prediction}\n")

write_submissions("submissions.csv", predictions)
```

After running these codes, a `submissions.csv` will be generated and you can submit it to Kaggle directly! In my personal experience, it could achieve from 0.71 to 0.76.

### Improve Your Results

Although the whole process is *carefree* enough, the final score is not yet satisfied. One way to improve the result is to try different models:

```python
m = cflearn.make("tree_linear", data_config=data_config).fit("train.csv")
predictions = m.predict("test.csv", contains_labels=False)
write_submissions("submissions_tree_linear.csv", predictions)
```

After submitting `submissions_tree_linear.csv`, we could achieve ~0.775 (and even up to 0.79) now, cool!

### Conclusions

Since `Titanic` is just a small toy dataset, using Neural Network to solve it might actually 'over-killed' (or, overfit) it, and that's why we decided to conclude here instead of introducing more fancy techniques (e.g. ensemble, AutoML, etc.). We hope that this small example can help you quickly walk through some basic concepts in `carefre-learn`, as well as help you leverage `carefree-learn` in your own tasks!


## Operations

| Python source code | Jupyter Notebook |
|:---:|:---:|
| [op.py](https://github.com/carefree0910/carefree-learn/blob/dev/examples/operations/op.py) | [op.ipynb](https://github.com/carefree0910/carefree-learn/blob/dev/examples/operations/op.ipynb) |

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

### The `add` Dataset

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

### The `prod` Dataset

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

A trivial thought is to manually `extract` the `prod` features $\tilde x$ from the input data:

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

from typing import Any
from typing import Dict

# register an `extract` which represents the `prod` operation
@cflearn.register_extractor("prod_extractor")
class ProdExtractor(cflearn.ExtractorBase):
    @property
    def out_dim(self) -> int:
        return 1

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return net.prod(dim=1, keepdim=True)


# define the `Config` for this `extract`
# since `ProdExtractor` don't need any configurations, we can simply return an empty dict here
@cflearn.register_config("prod_extractor", "default")
class ProdExtractorConfig(cflearn.Configs):
    def get_default(self) -> Dict[str, Any]:
        return {}
```

:::tip
If you are interested in how does `extract` actually work in `carefree-learn`, please refer to [pipe](../design-principles#pipe) and [extract](../design-principles#extract) for more information.
:::

After defining the `extract`, we need to define a model that leverages it:

```python
# we call this new model `prod`
@cflearn.register_model("prod")
# we use our new `extract` followed by traditional `linear` model
@cflearn.register_pipe("linear", extractor="prod_extractor")
class ProdNetwork(cflearn.ModelBase):
    pass
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

As we expected, the `prod` approaches to the ground truth easily.

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

### The `mix` Dataset

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

Thanks to `carefree-learn`, we again can actually do so, but this time we'll need some more coding. Recall that we build an expert in `prod` dataset by defining a novel `extract`, because we needed to pre-process the input data. However in `mix`, what we actually need is to combine `linear` and `prod`, which means we need to define a novel `head` this time.

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

@cflearn.register_head("mixture_head")
class MixtureHead(cflearn.HeadBase):
    def __init__(self, in_dim: int, out_dim: int, target_dim: int):
        super().__init__()
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

@cflearn.register_head_config("mixture_head", "add")
class MixtureHeadAddConfig(cflearn.HeadConfigs):
    def get_default(self) -> Dict[str, Any]:
        return {"target_dim": 0}


@cflearn.register_head_config("mixture_head", "prod")
class MixtureHeadProdConfig(cflearn.HeadConfigs):
    def get_default(self) -> Dict[str, Any]:
        return {"target_dim": 1}
    
# we use our new `head` to define the new model
# note that we need two `pipe`s, one for `add` and the other for `prod`
@cflearn.register_model("mixture")
@cflearn.register_pipe(
    "add",
    extractor="identity",
    head="mixture_head",
    head_config="add",
)
@cflearn.register_pipe(
    "prod",
    extractor="prod_extractor",
    head="mixture_head",
    head_config="prod",
)
class MixtureNetwork(cflearn.ModelBase):
    pass

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

As we expected, the `mixture` approaches to the ground truth easily.

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

### Conclusions

`Operations` are just artificial toy datasets, but quite handy for us to illustrate some basic concepts in `carefre-learn`. We hope that this small example can help you quickly walk through some development guides in `carefre-learn`, and help you leverage `carefree-learn` in your own tasks!
