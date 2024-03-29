---
id: installation
title: Installation
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

`carefree-learn` requires Python 3.6 or higher.

### Pre-Installing PyTorch

`carefree-learn` requires `pytorch>=1.9.0`. Please refer to [PyTorch](https://pytorch.org/get-started/locally/), and it is highly recommended to pre-install PyTorch with conda.

### pip installation

After installing PyTorch, installation of `carefree-learn` would be rather easy:

:::tip
If you pre-installed PyTorch with conda, remember to activate the corresponding environment!
:::

<Tabs
  defaultValue="pypi"
  values={[
    {label: 'PyPI', value: 'pypi'},
    {label: 'GitHub', value: 'github'},
  ]
}>

<TabItem value="pypi">

```bash
pip install carefree-learn
```

</TabItem>

<TabItem value="github">

```bash
git clone https://github.com/carefree0910/carefree-learn.git
cd carefree-learn
pip install -e .
```

</TabItem>

</Tabs>


## Docker

### Prepare

`carefree-learn` has already been published on DockerHub, so it can be pulled directly:

```bash
docker pull carefree0910/carefree-learn:dev
```

or can be built locally:

```bash
docker build -t carefree0910/carefree-learn:dev .
```

### Run

```bash
docker run --rm -it --gpus all carefree0910/carefree-learn:dev
```
