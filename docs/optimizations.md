---
id: optimizations
title: Optimizations
---

`carefree-learn` not only provides *carefree* APIs for easier usages, but also did quite a few optimizations to make training on tabular datasets faster than other similar libraries. In this page we'll introduce some techniques `carefree-learn` adopted under the hood, and will show how much performance boost we've obtained with them.


## Categorical Encodings

Encoding categorical features is one of the most important pre-processing we need to perform on tabular datasets. As mentioned in [`Design Principles`](./design-principles#transform), `carefree-learn` will stick to [`one_hot`](#one-hot-encoding) encoding and [`embedding`](#embedding) encoding, which will be introduced in turn in the following sections.

### One Hot Encoding

A `one_hot` encoding basically encodes categorical features as a one-hot numeric array, as defined in [`sklearn`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). Suppose we have 5 classes in total, then:

$$
\text{OneHot}(0) = [1,0,0,0,0] \\
\text{OneHot}(3) = [0,0,0,1,0]
$$

We can figure out that this kind of encoding is **static**, which means it will not change during the training process. In this case, we can cache down all the encodings and access them through indexing. This will speed up the encoding process for ~66x:

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder

num_data = 10000
num_classes = 10
batch_size = 128

x = np.random.randint(0, num_classes, num_data).reshape([-1, 1])
enc = OneHotEncoder(sparse=False).fit(x)
x_one_hot = enc.transform(x)

target_indices = np.random.permutation(num_data)[:batch_size]
x_target = x[target_indices]

assert np.allclose(x_one_hot[target_indices], enc.transform(x_target))
%timeit x_one_hot[target_indices]  # 2.97 µs ± 22.4 ns
%timeit enc.transform(x_target)    # 207 µs ± 4.05 µs
```

The gap will decrease a little when we increase `num_classes` to a larger number, let's say 100:

```python
%timeit x_one_hot[target_indices]  # 6.26 µs ± 81 ns
%timeit enc.transform(x_target)    # 201 µs ± 1.98 µs
```

However there are still ~33x boost.

:::caution
Although caching can boost performance, it is at the cost of consuming much more memories. A better solution should be caching sparse tensors instead of dense ones, but `PyTorch` has not supported sparsity good enough. See [Sparsity](#sparsity) section for more details.
:::

### Embedding

An `embedding` encoding actually borrows from **N**atual **L**anguage **P**rocessing (**NLP**) where they converted (sparse) input words into dense embeddings with embedding look up. It is quite trivial to turn categorical features into embeddings with the same look up techniques, but tabular datasets hold a different property compared with **NLP**: tabular datasets will maintain many embedding tables because they have different categorical features with different number of values, while in **NLP** it only need to maintain one embedding table in most cases.

Since `embedding` is a **dynamic** encoding which contains trainable parameters, we cannot cache them beforehand like we did to `one_hot`. However, we can still optimize it with *fast embedding*. A *fast embedding* basically unifies the embedding dimension of different categorical features, so one unified embedding table is sufficient for the whole `embedding` process.

There's one more thing we need to take care of when applying *fast embedding*: we need to *increment* the values of each categorical features. Here's a minimal example to illustrate this. Suppose we have two categorical features ($x_1, x_2$) with 2 and 3 classes respectively, then our embedding table will contain 5 rows:

$$
\begin{bmatrix}
    \text{---} \text{---} \ v_1 \ \text{---} \text{---} \\
    \text{---} \text{---} \ v_2 \ \text{---} \text{---} \\
    \text{---} \text{---} \ v_3 \ \text{---} \text{---} \\
    \text{---} \text{---} \ v_4 \ \text{---} \text{---} \\
    \text{---} \text{---} \ v_5 \ \text{---} \text{---}
\end{bmatrix}
$$

In this table, the first two rows belong to $x_1$, while the last three rows belong to $x_2$. However, as we defined above, $x_1\in\{0,1\}$ and $x_2\in\{0,1,2\}$. In order to assign $v_3,v_4,v_5$ to $x_2$, we need to *increment* $x_2$ by $2$ (which is the number of choices $x_1$ could have). After *increment*, we have $x_2\in\{2,3,4\}$ so it can successfully look up $v_3,v_4,v_5$.

Note that the *incremented* indices are **static**, so `carefree-learn` will cache these indices to avoid duplicate calculations when *fast embedding* is applied.

Since the embedding dimensions are unified, *fast embedding* actually reduces the flexibility a little bit, but it can speed up the encoding process for ~17.5x:


```python
import math
import torch
import numpy as np
from torch.nn import Embedding
from cflearn.misc.toolkit import to_torch

dim = 20
batch_size = 256

features = []
embeddings = []
for i in range(dim):
    # 5, 10, 15, 20
    num_classes = math.ceil((i + 1) / 5) * 5
    x = np.random.randint(0, num_classes, batch_size).reshape([-1, 1])
    embedding = Embedding(num_classes, 1)
    embeddings.append(embedding)
    features.append(x)

fast_embedding = Embedding(250, 1)
tensor = to_torch(np.hstack(features)).to(torch.long)

def f1():
    return fast_embedding(tensor)

def f2():
    embedded = []
    for i, embedding in enumerate(embeddings):
        embedded.append(embedding(tensor[..., i:i+1]))
    return torch.cat(embedded, dim=1)

assert f1().shape == f2().shape
%timeit f1()  # 33.6 µs ± 506 ns
%timeit f2()  # 587 µs ± 3.02 µs
```

:::note
Theoratically, `embedding` encoding is nothing more than a `one_hot` encoding followed by a linear projection, so it should be fast enough if we apply sparse matrix multiplications between `one_hot` encodings and a block diagnal `embedding` look up table. However as mentioned in [One Hot Encoding](#one-hot-encoding) section, `PyTorch` has not supported sparsity good enough. See [Sparsity](#sparsity) section for more details.
:::

### Sparsity

It is quite trivial that the `one_hot` encoding actually outputs a sparse matrix with sparsity equals to:

$$
1-\frac{1}{\text{num\_classes}}
$$

So the sparsity could easily exceed 90%, when `num_classes` only needs to be greater than 10, therefore it is quite natural to think of leveraging sparse data structures to cache these `one_hot` encodings. What's better is that the `embedding` encoding could be represented as sparse matrix multiplications between `one_hot` encodings and a block diagnal `embedding` look up table, so **THEORATICALLY** (🤣) we could reuse the `one_hot` encodings to get the `embedding` encodings efficiently.

Unfortunately, although [`scipy`](https://docs.scipy.org/doc/scipy/reference/sparse.html) supports sparse matrices pretty well, `PyTorch` has not yet supported them good enough. So we'll stick to the dense solutions mentioned above, but will switch to the sparse ones iff `PyTorch` releases some fancy sparsity supports!
