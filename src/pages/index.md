import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

![carefree-learn][socialify-image]

Deep Learning with [PyTorch](https://pytorch.org/) made easy 🚀 !


## Carefree?

`carefree-learn` aims to provide **CAREFREE** usages for both users and developers. It also provides a [corresponding repo](https://github.com/carefree0910/carefree-learn-deploy) for production.

### Machine Learning 📈

<Tabs
  defaultValue="users"
  values={[
    {label: 'Users', value: 'users'},
    {label: 'Developers', value: 'developers'},
  ]
}>

<TabItem value="users">

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
m = cflearn.api.fit_ml(x, y, carefree=True)
```

</TabItem>

<TabItem value="developers">

> This is a WIP section :D

</TabItem>

</Tabs>

### Computer Vision 🖼️

<Tabs
  defaultValue="users"
  values={[
    {label: 'Users', value: 'users'},
    {label: 'Developers', value: 'developers'},
  ]
}>

<TabItem value="users">

```python
import cflearn

data = cflearn.cv.MNISTData(batch_size=16, transform="to_tensor")
m = cflearn.api.resnet18_gray(10).fit(data)
```

</TabItem>

<TabItem value="developers">

> This is a WIP section :D

</TabItem>

</Tabs>

:::info
Please refer to [Quick Start](docs/getting-started/quick-start) and [Developer Guides](docs/developer-guides/general) for detailed information.
:::


## Why carefree-learn?

`carefree-learn` is a general Deep Learning framework based on PyTorch. Since `v0.2.x`, `carefree-learn` has extended its usage from **tabular dataset** to (almost) **all kinds of dataset**. In the mean time, the APIs remain (almost) the same as `v0.1.x`: still simple, powerful and easy to use!

Here are some main advantages that `carefree-learn` holds:

### Machine Learning 📈

+ Provides a [scikit-learn](https://scikit-learn.org/stable/)-like interface with much more 'carefree' usages, including:
    + Automatically deals with data pre-processing.
    + Automatically handles datasets saved in files (.txt, .csv).
    + Supports [Distributed Training](docs/user-guides/distributed#distributed-training), which means hyper-parameter tuning can be very efficient in `carefree-learn`.
+ Includes some brand new techniques which may boost vanilla Neural Network (NN) performances on tabular datasets, including:
    + [`TreeDNN` with `Dynamic Soft Pruning`](https://arxiv.org/pdf/1911.05443.pdf), which makes NN less sensitive to hyper-parameters. 
    + [`Deep Distribution Regression (DDR)`](https://arxiv.org/pdf/1911.05441.pdf), which is capable of modeling the entire conditional distribution with one single NN model.
+ Supports many convenient functionality in deep learning, including:
    + Early stopping.
    + Model persistence.
    + Learning rate schedulers.
    + And more...
+ Full utilization of the WIP ecosystem `cf*`, such as:
    + [`carefree-toolkit`](https://github.com/carefree0910/carefree-toolkit): provides a lot of utility classes & functions which are 'stand alone' and can be leveraged in your own projects.
    + [`carefree-data`](https://github.com/carefree0910/carefree-data): a lightweight tool to read -> convert -> process **ANY** tabular datasets. It also utilizes [cython](https://cython.org/) to accelerate critical procedures.

From the above, it comes out that `carefree-learn` could be treated as a minimal **Auto**matic **M**achine **L**earning (AutoML) solution for tabular datasets when it is fully utilized. However, this is not built on the sacrifice of flexibility. In fact, the functionality we've mentioned are all wrapped into individual modules in `carefree-learn` and allow users to customize them easily.

### Computer Vision 🖼️

+ Also provides a [scikit-learn](https://scikit-learn.org/stable/)-like interface with much more 'carefree' usages.
+ Provides many out-of-the-box pre-trained models and well hand-crafted training defaults for reproduction & finetuning.
+ Seamlessly supports efficient `ddp` (simply call `cflearn.api.run_ddp("run.py")`, where `run.py` is your normal training script).
+ Bunch of utility functions for research and production.


## Citation

If you use `carefree-learn` in your research, we would greatly appreciate if you cite this library using this Bibtex:

```
@misc{carefree-learn,
  year={2020},
  author={Yujian He},
  title={carefree-learn, Deep Learning with PyTorch made easy},
  howpublished={\url{https://https://github.com/carefree0910/carefree-learn/}},
}
```


## License

`carefree-learn` is MIT licensed, as found in the [`LICENSE`](docs/about/license) file.


[socialify-image]: https://socialify.git.ci/carefree0910/carefree-learn/image?description=1&descriptionEditable=Tabular%20Datasets%20%E2%9D%A4%EF%B8%8F%C2%A0PyTorch&font=Inter&forks=1&issues=1&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fcarefree0910%2Fcarefree-learn-doc%2Fmaster%2Fstatic%2Fimg%2Flogo.min.svg&pattern=Floating%20Cogs&stargazers=1&theme=Light
