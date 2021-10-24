---
id: contributing
title: Contributing
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Thank you for your interest in contributing to `carefree-learn`! Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it.
    - Post about your intended feature in an [issue](https://github.com/carefree0910/carefree-learn/issues), and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    - Search for your issue in the [`carefree-learn` issue list](https://github.com/carefree0910/carefree-learn/issues).
    - Pick an issue and comment that you'd like to work on the feature or bug-fix.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you implement and test your feature or bug-fix, please include some unittests and submit a Pull Request to https://github.com/carefree0910/carefree-learn.


## Developing

To develop `carefree-learn` on your machine, here are some tips:

1. Uninstall all existing `carefree-learn` installs:
```bash
conda uninstall carefree-learn
pip uninstall carefree-learn
```

2. Follow [Installation Guide](/docs/getting-started/installation) to install `carefree-learn`. Remember to choose the `GitHub` tab in the [pip installation](/docs/getting-started/installation#pip-installation) section.

3. Follow [Style Guide](#style-guide) and happy coding!


## Style Guide

`carefree-learn` adopted [`black`](https://github.com/psf/black) and [`mypy`](https://github.com/python/mypy) to stylize its codes, so you may need to check the format, coding style and type hint with them before your codes could actually be merged.

Besides, there are a few more principles that I'm using for sorting imports:
+ From short to long (for both naming and path).
+ From *a* to *z* (alphabetically).
+ Divided into four sections:
  1. `import ...`
  2. `import ... as ...`
  3. `from ... import ...`
  4. relative imports
+ From general to specific (a `*` will always appear at the top of each section)

Here's an example to illustrate these ([source code](https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/api/auto.py)):

```python
import os
import json
import torch
import optuna

import numpy as np
import optuna.visualization as vis

from typing import *
from functools import partial
from tqdm.autonotebook import tqdm
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.ml.utils import scoring_fn_type
from cfdata.tabular import task_type_type
from cfdata.tabular import parse_task_type
from cfdata.tabular import TabularData
from optuna.trial import TrialState
from optuna.trial import FrozenTrial
from optuna.importance import BaseImportanceEvaluator
from plotly.graph_objects import Figure

from .basic import *
from .ensemble import *
from .hpo import optuna_tune
from .hpo import default_scoring
from .hpo import optuna_params_type
from .hpo import OptunaPresetParams
from .production import Pack
from .production import Predictor
from ..types import data_type
```

But after all, this is not a strict constraint so everything will be fine as long as it 'looks good'🤣


## Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind.

### Title

The title of your pull request should

+ briefly describe and reflect the changes
+ wrap any code with backticks

### Description

The description of your pull request should

- describe the motivation
- describe the changes
- if still work-in-progress, describe remaining tasks
