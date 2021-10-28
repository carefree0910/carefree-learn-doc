(window.webpackJsonp=window.webpackJsonp||[]).push([[9],{78:function(e,a,n){"use strict";n.r(a),n.d(a,"frontMatter",(function(){return c})),n.d(a,"metadata",(function(){return l})),n.d(a,"rightToc",(function(){return s})),n.d(a,"default",(function(){return d}));var t=n(3),r=n(7),i=(n(0),n(97)),c={id:"Iris",title:"Iris"},l={unversionedId:"examples/Iris",id:"examples/Iris",isDocsHomePage:!1,title:"Iris",description:"| Python source code | Jupyter Notebook | Task |",source:"@site/docs/examples/iris.md",slug:"/examples/Iris",permalink:"/carefree-learn-doc/docs/examples/Iris",version:"current",lastUpdatedAt:1634995906,sidebar:"docs",previous:{title:"Configurations",permalink:"/carefree-learn-doc/docs/getting-started/configurations"},next:{title:"Titanic",permalink:"/carefree-learn-doc/docs/examples/Titanic"}},s=[{value:"Basic Usages",id:"basic-usages",children:[]},{value:"Benchmarking",id:"benchmarking",children:[]},{value:"Advanced Benchmarking",id:"advanced-benchmarking",children:[]},{value:"Conclusion",id:"conclusion",children:[]}],o={rightToc:s};function d(e){var a=e.components,n=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(t.a)({},o,n,{components:a,mdxType:"MDXLayout"}),Object(i.b)("table",null,Object(i.b)("thead",{parentName:"table"},Object(i.b)("tr",{parentName:"thead"},Object(i.b)("th",Object(t.a)({parentName:"tr"},{align:"center"}),"Python source code"),Object(i.b)("th",Object(t.a)({parentName:"tr"},{align:"center"}),"Jupyter Notebook"),Object(i.b)("th",Object(t.a)({parentName:"tr"},{align:"center"}),"Task"))),Object(i.b)("tbody",{parentName:"table"},Object(i.b)("tr",{parentName:"tbody"},Object(i.b)("td",Object(t.a)({parentName:"tr"},{align:"center"}),Object(i.b)("a",Object(t.a)({parentName:"td"},{href:"https://github.com/carefree0910/carefree-learn/blob/dev/examples/ml/iris/run_iris.py"}),"iris.py")),Object(i.b)("td",Object(t.a)({parentName:"tr"},{align:"center"}),Object(i.b)("a",Object(t.a)({parentName:"td"},{href:"https://nbviewer.org/github/carefree0910/carefree-learn/blob/dev/examples/ml/iris/iris.ipynb"}),"iris.ipynb")),Object(i.b)("td",Object(t.a)({parentName:"tr"},{align:"center"}),"Machine Learning \ud83d\udcc8")))),Object(i.b)("p",null,"Here are some of the information provided by the official website:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-text"}),"This is perhaps the best known database to be found in the pattern recognition literature.\nThe data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.\nPredicted attribute: class of iris plant.\n")),Object(i.b)("p",null,"And here's the pandas-view of the raw data:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-text"}),"      f0   f1   f2   f3           label\n0    5.1  3.5  1.4  0.2     Iris-setosa\n1    4.9  3.0  1.4  0.2     Iris-setosa\n2    4.7  3.2  1.3  0.2     Iris-setosa\n3    4.6  3.1  1.5  0.2     Iris-setosa\n4    5.0  3.6  1.4  0.2     Iris-setosa\n..   ...  ...  ...  ...             ...\n145  6.7  3.0  5.2  2.3  Iris-virginica\n146  6.3  2.5  5.0  1.9  Iris-virginica\n147  6.5  3.0  5.2  2.0  Iris-virginica\n148  6.2  3.4  5.4  2.3  Iris-virginica\n149  5.9  3.0  5.1  1.8  Iris-virginica\n\n[150 rows x 5 columns]\n")),Object(i.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(i.b)("div",Object(t.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(t.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(t.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(i.b)("path",Object(t.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(i.b)("div",Object(t.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("ul",{parentName:"div"},Object(i.b)("li",{parentName:"ul"},"You can download the raw data (",Object(i.b)("inlineCode",{parentName:"li"},"iris.data"),") with ",Object(i.b)("a",Object(t.a)({parentName:"li"},{href:"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"}),"this link"),"."),Object(i.b)("li",{parentName:"ul"},"We didn't use pandas in our code, but it is convenient to visualize some data with it though \ud83e\udd23")))),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),"# preparations\n\nimport os\nimport torch\nimport pickle\nimport cflearn\nimport numpy as np\n\nnp.random.seed(142857)\ntorch.manual_seed(142857)\n")),Object(i.b)("h2",{id:"basic-usages"},"Basic Usages"),Object(i.b)("p",null,"Traditionally, we need to process the raw data before we feed them into our machine learning models (e.g. encode the label column, which is a string column, into an ordinal column). In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", however, we can train neural networks directly on files without worrying about the rest:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),'m = cflearn.api.fit_ml("iris.data", carefree=True)\n')),Object(i.b)("p",null,"What's going under the hood is that ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," will try to parse the ",Object(i.b)("inlineCode",{parentName:"p"},"iris.data")," automatically (with the help of ",Object(i.b)("a",Object(t.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-data"}),"carefree-data"),"), split the data into training set and validation set, with which we'll train a fully connected neural network (fcnn)."),Object(i.b)("p",null,"We can further inspect the processed data if we want to know how ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," actually parsed the input data:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),"print(m.cf_data.raw.x[0])\nprint(m.cf_data.raw.y[0])\nprint(m.cf_data.processed.x[0])\nprint(m.cf_data.processed.y[0])\n")),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-text"}),"[5.1 3.5 1.4 0.2]\n['Iris-setosa']\n[-0.9006812  1.0320569 -1.3412726 -1.3129768]\n[2]\n")),Object(i.b)("p",null,"It shows that the raw data is carefully normalized into numerical data that neural networks can accept. What's more, by saying ",Object(i.b)("em",{parentName:"p"},"normalized"),", it means that the input features will be automatically normalized to ",Object(i.b)("inlineCode",{parentName:"p"},"mean=0.0")," and ",Object(i.b)("inlineCode",{parentName:"p"},"std=1.0"),":"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),"ata = m.data\nx_train, y_train = data.train_cf_data.processed.xy\nx_valid, y_valid = data.valid_cf_data.processed.xy\nstacked = np.vstack([x_train, x_valid])\nprint(stacked.mean(0))\nprint(stacked.std(0))\n")),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-text"}),"[-5.5631002e-09 -3.2588841e-07 -1.5576680e-07 -5.7220458e-08]\n[0.9999999 1.0000001 1.0000002 0.9999999]\n")),Object(i.b)("div",{className:"admonition admonition-info alert alert--info"},Object(i.b)("div",Object(t.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(t.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(t.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(t.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(i.b)("div",Object(t.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"The results shown above means we first normalized the data before we actually split it into train & validation set."))),Object(i.b)("p",null,"After training on files, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," can predict & evaluate on files directly as well. We'll handle the data parsing and normalization for you automatically:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),'# instantiate an `MLInferenceData` instance\nidata = cflearn.MLInferenceData("iris.data")\n# `contains_labels` is set to True because `iris.data` itself contains labels\npredictions = m.predict(idata, contains_labels=True)\n# It is OK to simply call `m.predict("iris.data")` because `contains_labels` is True by default\npredictions = m.predict(idata)\n# evaluations could be achieved easily with cflearn.evaluate\ncflearn.ml.evaluate(idata, metrics=["acc", "auc"], pipelines=m)\n')),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          |    0.926667    |    0.000000    |    0.926667    |    0.994800    |    0.000000    |    0.994800    |\n================================================================================================================================\n")),Object(i.b)("h2",{id:"benchmarking"},"Benchmarking"),Object(i.b)("p",null,"As we know, neural networks are trained with ",Object(i.b)("strong",{parentName:"p"},Object(i.b)("em",{parentName:"strong"},"stochastic"))," gradient descent (and its variants), which will introduce some randomness to the final result, even if we are training on the same dataset. In this case, we need to repeat the same task several times in order to obtain the bias & variance of our neural networks."),Object(i.b)("p",null,"Fortunately, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," introduced ",Object(i.b)("inlineCode",{parentName:"p"},"repeat_ml")," API, which can achieve this goal easily with only a few lines of code:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),'# With num_repeat=3 specified, we\'ll train 3 models on `iris.data`.\nresult = cflearn.api.repeat_ml("iris.data", carefree=True, num_repeat=3)\ncflearn.ml.evaluate(idata, metrics=["acc", "auc"], pipelines=result.pipelines)\n')),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          |    0.902222    |    0.019116    |    0.883106    |    0.985778    |    0.004722    |    0.981055    |\n================================================================================================================================\n")),Object(i.b)("p",null,"We can also compare the performances across different models:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),'# With models=["linear", "fcnn"], we\'ll train both linear models and fcnn models.\nmodels = ["linear", "fcnn"]\nresult = cflearn.api.repeat_ml("iris.data", carefree=True, models=models, num_repeat=3)\ncflearn.ml.evaluate(idata, metrics=["acc", "auc"], pipelines=result.pipelines)\n')),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          | -- 0.915556 -- | -- 0.027933 -- | -- 0.887623 -- | -- 0.985467 -- | -- 0.004121 -- | -- 0.981345 -- |\n--------------------------------------------------------------------------------------------------------------------------------\n|         linear         |    0.620000    |    0.176970    |    0.443030    |    0.733778    |    0.148427    |    0.585351    |\n================================================================================================================================\n")),Object(i.b)("p",null,"It is worth mentioning that ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," supports Distributed Training, which means when we need to perform large scale benchmarking (e.g. train 100 models), we could accelerate the process through multiprocessing:"),Object(i.b)("div",{className:"admonition admonition-info alert alert--info"},Object(i.b)("div",Object(t.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(t.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(t.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(t.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(i.b)("div",Object(t.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", Distributed Training in Machine Learning tasks sometimes doesn't mean training your model on multiple GPUs or multiple machines. Instead, it may mean training multiple models at the same time."))),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),'# With num_jobs=2, we will launch 2 processes to run the tasks in a distributed way.\nresult = cflearn.api.repeat_ml("iris.data", carefree=True, num_repeat=10, num_jobs=2)\n')),Object(i.b)("div",{className:"admonition admonition-caution alert alert--warning"},Object(i.b)("div",Object(t.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(t.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(t.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"16",height:"16",viewBox:"0 0 16 16"}),Object(i.b)("path",Object(t.a)({parentName:"svg"},{fillRule:"evenodd",d:"M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"})))),"caution")),Object(i.b)("div",Object(t.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"On iris dataset, however, launching distributed training will actually hurt the speed because iris dataset only contains 150 samples, so the relative overhead brought by distributed training will be too large."))),Object(i.b)("h2",{id:"advanced-benchmarking"},"Advanced Benchmarking"),Object(i.b)("p",null,"But this is not enough, because we want to know whether other models (e.g. scikit-learn models) could achieve a better performance than ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," models. In this case, we can perform an advanced benchmarking with the ",Object(i.b)("inlineCode",{parentName:"p"},"Experiment")," helper class."),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),'experiment = cflearn.dist.ml.Experiment()\ndata_folder = experiment.dump_data_bundle(x_train, y_train, x_valid, y_valid)\n\n# Add carefree-learn tasks\nexperiment.add_task(model="fcnn", data_folder=data_folder)\nexperiment.add_task(model="linear", data_folder=data_folder)\n# Add scikit-learn tasks\nrun_command = f"python run_sklearn.py"\ncommon_kwargs = {"run_command": run_command, "data_folder": data_folder}\nexperiment.add_task(model="decision_tree", **common_kwargs)\nexperiment.add_task(model="random_forest", **common_kwargs)\n')),Object(i.b)("p",null,"Notice that we specified ",Object(i.b)("inlineCode",{parentName:"p"},'run_command="python run_sklearn.py"')," for scikit-learn tasks, which means ",Object(i.b)("inlineCode",{parentName:"p"},"Experiment")," will try to execute this command in the current working directory for training scikit-learn models. The good news is that we do not need to speciy any command line arguments, because ",Object(i.b)("inlineCode",{parentName:"p"},"Experiment")," will handle those for us."),Object(i.b)("p",null,"Here is basically what a ",Object(i.b)("inlineCode",{parentName:"p"},"run_sklearn.py")," should look like (",Object(i.b)("a",Object(t.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/dev/examples/ml/iris/run_sklearn.py"}),"source code"),"):"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),'import os\nimport pickle\n\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom cflearn.dist.runs._utils import get_info\n\nif __name__ == "__main__":\n    info = get_info()\n    kwargs = info.kwargs\n    # data\n    data_list = info.data_list\n    x, y = data_list[:2]\n    # model\n    model = kwargs["model"]\n    if model == "decision_tree":\n        base = DecisionTreeClassifier\n    elif model == "random_forest":\n        base = RandomForestClassifier\n    else:\n        raise NotImplementedError\n    sk_model = base()\n    # train & save\n    sk_model.fit(x, y.ravel())\n    with open(os.path.join(info.workplace, "sk_model.pkl"), "wb") as f:\n        pickle.dump(sk_model, f)\n')),Object(i.b)("p",null,"With ",Object(i.b)("inlineCode",{parentName:"p"},"run_sklearn.py")," defined, we could run those tasks with one line of code:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),"results = experiment.run_tasks()\n")),Object(i.b)("p",null,"After finished running with this, we should be able to see the following file structure in the current working directory:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-text"}),"|--- __experiment__\n   |--- __data__\n      |-- x.npy\n      |-- y.npy\n      |-- x_cv.npy\n      |-- y_cv.npy\n   |--- fcnn/0\n      |-- _logs\n      |-- __meta__.json\n      |-- cflearn^_^fcnn^_^0000.zip\n   |--- linear/0\n      |-- ...\n   |--- decision_tree/0\n      |-- __meta__.json\n      |-- sk_model.pkl\n   |--- random_forest/0\n      |-- ...\n")),Object(i.b)("p",null,"As we expected, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," models are saved into ",Object(i.b)("inlineCode",{parentName:"p"},".zip")," files, while scikit-learn models are saved into ",Object(i.b)("inlineCode",{parentName:"p"},"sk_model.pkl")," files. Since these models are not yet loaded, we should manually load them into our environment:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),'pipelines = {}\nsk_patterns = {}\nfor workplace, workplace_key in zip(results.workplaces, results.workplace_keys):\n    model = workplace_key[0]\n    if model not in ["decision_tree", "random_forest"]:\n        pipelines[model] = cflearn.ml.task_loader(workplace)\n    else:\n        model_file = os.path.join(workplace, "sk_model.pkl")\n        with open(model_file, "rb") as f:\n            sk_model = pickle.load(f)\n            # In `carefree-learn`, we treat labels as column vectors.\n            # So we need to reshape the outputs from the scikit-learn models.\n            sk_predict = lambda d: sk_model.predict(d.x_train).reshape([-1, 1])\n            sk_predict_prob = lambda d: sk_model.predict_proba(d.x_train)\n            sk_pattern = cflearn.ml.ModelPattern(\n                predict_method=sk_predict,\n                predict_prob_method=sk_predict_prob,\n            )\n            sk_patterns[model] = sk_pattern\n')),Object(i.b)("p",null,"After which we can finally perform benchmarking on these models:"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-python"}),'idata = cflearn.MLInferenceData(x_valid, y_valid)\ncflearn.ml.evaluate(idata, metrics=["acc", "auc"], pipelines=pipelines, other_patterns=sk_patterns)\n')),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(t.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|     decision_tree      |    0.920000    | -- 0.000000 -- |    0.920000    |    0.995199    | -- 0.000000 -- |    0.995199    |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          | -- 0.959999 -- | -- 0.000000 -- | -- 0.959999 -- | -- 0.997866 -- | -- 0.000000 -- | -- 0.997866 -- |\n--------------------------------------------------------------------------------------------------------------------------------\n|         linear         |    0.693333    | -- 0.000000 -- |    0.693333    |    0.940533    | -- 0.000000 -- |    0.940533    |\n--------------------------------------------------------------------------------------------------------------------------------\n|     random_forest      |    0.920000    | -- 0.000000 -- |    0.920000    |    0.995199    | -- 0.000000 -- |    0.995199    |\n================================================================================================================================\n")),Object(i.b)("h2",{id:"conclusion"},"Conclusion"),Object(i.b)("p",null,"Contained in this article is just a subset of the features that ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," offers, but we've already walked through many basic & common steps we'll encounter in real life machine learning tasks."))}d.isMDXComponent=!0},97:function(e,a,n){"use strict";n.d(a,"a",(function(){return p})),n.d(a,"b",(function(){return u}));var t=n(0),r=n.n(t);function i(e,a,n){return a in e?Object.defineProperty(e,a,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[a]=n,e}function c(e,a){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);a&&(t=t.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),n.push.apply(n,t)}return n}function l(e){for(var a=1;a<arguments.length;a++){var n=null!=arguments[a]?arguments[a]:{};a%2?c(Object(n),!0).forEach((function(a){i(e,a,n[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):c(Object(n)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(n,a))}))}return e}function s(e,a){if(null==e)return{};var n,t,r=function(e,a){if(null==e)return{};var n,t,r={},i=Object.keys(e);for(t=0;t<i.length;t++)n=i[t],a.indexOf(n)>=0||(r[n]=e[n]);return r}(e,a);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(t=0;t<i.length;t++)n=i[t],a.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var o=r.a.createContext({}),d=function(e){var a=r.a.useContext(o),n=a;return e&&(n="function"==typeof e?e(a):l(l({},a),e)),n},p=function(e){var a=d(e.components);return r.a.createElement(o.Provider,{value:a},e.children)},b={inlineCode:"code",wrapper:function(e){var a=e.children;return r.a.createElement(r.a.Fragment,{},a)}},m=r.a.forwardRef((function(e,a){var n=e.components,t=e.mdxType,i=e.originalType,c=e.parentName,o=s(e,["components","mdxType","originalType","parentName"]),p=d(n),m=t,u=p["".concat(c,".").concat(m)]||p[m]||b[m]||i;return n?r.a.createElement(u,l(l({ref:a},o),{},{components:n})):r.a.createElement(u,l({ref:a},o))}));function u(e,a){var n=arguments,t=a&&a.mdxType;if("string"==typeof e||t){var i=n.length,c=new Array(i);c[0]=m;var l={};for(var s in a)hasOwnProperty.call(a,s)&&(l[s]=a[s]);l.originalType=e,l.mdxType="string"==typeof e?e:t,c[1]=l;for(var o=2;o<i;o++)c[o]=n[o];return r.a.createElement.apply(null,c)}return r.a.createElement.apply(null,n)}m.displayName="MDXCreateElement"}}]);