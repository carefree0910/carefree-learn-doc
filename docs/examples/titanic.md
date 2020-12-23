---
id: Titanic
title: Titanic
---

| Python source code | Jupyter Notebook |
|:---:|:---:|
| [titanic.py](https://github.com/carefree0910/carefree-learn/blob/dev/examples/titanic/titanic.py) | [titanic.ipynb](https://nbviewer.jupyter.org/github/carefree0910/carefree-learn/blob/dev/examples/titanic/titanic.ipynb) |

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


## Configurations

Since the target column is not the last column (which is the default setting of `carefree-learn`), we need to manually configure it:

```python
data_config = {"label_name": "Survived"}
```

And you're all set! Notice that only the `label_name` needs to be provided, and `carefree-learn` will find out the corresponding target column for you😉


## Build Your Model

For instance, we'll use the famous `Wide & Deep` model:

```python
import cflearn

m = cflearn.make("wnd", data_config=data_config)
```

Unlike other libraries, `carefree-learn` supports *file-in*:

```python
m.fit("train.csv")
```


## Evaluate Your Model

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


## Making Predictions

Again, we can directly make predictions with a file (*file-out*):

```python
# `contains_labels` is set to False because `test.csv` does not contain labels
# It is OK to simply call `m.predict("test.csv")` because `contains_labels` is False by default
predictions = m.predict("test.csv", contains_labels=False)
```


## Submit Your Results

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


## Improve Your Results

Although the whole process is *carefree* enough, the final score is not yet satisfied. One way to improve the result is to try different models:

```python
m = cflearn.make("tree_linear", data_config=data_config).fit("train.csv")
predictions = m.predict("test.csv", contains_labels=False)
write_submissions("submissions_tree_linear.csv", predictions)
```

After submitting `submissions_tree_linear.csv`, we could achieve ~0.775 (and even up to 0.79) now, cool!


## Conclusions

Since `Titanic` is just a small toy dataset, using Neural Network to solve it might actually 'over-killed' (or, overfit) it, and that's why we decided to conclude here instead of introducing more fancy techniques (e.g. ensemble, AutoML, etc.). We hope that this small example can help you quickly walk through some basic concepts in `carefre-learn`, as well as help you leverage `carefree-learn` in your own tasks!
