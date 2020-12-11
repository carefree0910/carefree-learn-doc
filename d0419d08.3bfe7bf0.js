(window.webpackJsonp=window.webpackJsonp||[]).push([[26],{106:function(e,n,t){"use strict";t.d(n,"a",(function(){return p})),t.d(n,"b",(function(){return u}));var a=t(0),r=t.n(a);function i(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function l(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?l(Object(t),!0).forEach((function(n){i(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):l(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function c(e,n){if(null==e)return{};var t,a,r=function(e,n){if(null==e)return{};var t,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var s=r.a.createContext({}),d=function(e){var n=r.a.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},p=function(e){var n=d(e.components);return r.a.createElement(s.Provider,{value:n},e.children)},m={inlineCode:"code",wrapper:function(e){var n=e.children;return r.a.createElement(r.a.Fragment,{},n)}},b=r.a.forwardRef((function(e,n){var t=e.components,a=e.mdxType,i=e.originalType,l=e.parentName,s=c(e,["components","mdxType","originalType","parentName"]),p=d(t),b=a,u=p["".concat(l,".").concat(b)]||p[b]||m[b]||i;return t?r.a.createElement(u,o(o({ref:n},s),{},{components:t})):r.a.createElement(u,o({ref:n},s))}));function u(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var i=t.length,l=new Array(i);l[0]=b;var o={};for(var c in n)hasOwnProperty.call(n,c)&&(o[c]=n[c]);o.originalType=e,o.mdxType="string"==typeof e?e:a,l[1]=o;for(var s=2;s<i;s++)l[s]=t[s];return r.a.createElement.apply(null,l)}return r.a.createElement.apply(null,t)}b.displayName="MDXCreateElement"},96:function(e,n,t){"use strict";t.r(n),t.d(n,"frontMatter",(function(){return l})),t.d(n,"metadata",(function(){return o})),t.d(n,"rightToc",(function(){return c})),t.d(n,"default",(function(){return d}));var a=t(3),r=t(7),i=(t(0),t(106)),l={id:"distributed",title:"Distributed"},o={unversionedId:"user-guides/distributed",id:"user-guides/distributed",isDocsHomePage:!1,title:"Distributed",description:"Distributed Training",source:"@site/docs/user-guides/distributed.md",slug:"/user-guides/distributed",permalink:"/carefree-learn-doc/docs/user-guides/distributed",version:"current",lastUpdatedAt:1607710335,sidebar:"docs",previous:{title:"AutoML",permalink:"/carefree-learn-doc/docs/user-guides/auto-ml"},next:{title:"Production",permalink:"/carefree-learn-doc/docs/user-guides/production"}},c=[{value:"Distributed Training",id:"distributed-training",children:[{value:"<code>repeat_with</code>",id:"repeat_with",children:[]},{value:"<code>Experiment</code>",id:"experiment",children:[]},{value:"Conclusions",id:"conclusions",children:[]}]},{value:"Benchmarking",id:"benchmarking",children:[{value:"Advanced Benchmarking",id:"advanced-benchmarking",children:[]}]},{value:"Hyper Parameter Optimization (HPO)",id:"hyper-parameter-optimization-hpo",children:[]}],s={rightToc:c};function d(e){var n=e.components,t=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(a.a)({},s,t,{components:n,mdxType:"MDXLayout"}),Object(i.b)("h2",{id:"distributed-training"},"Distributed Training"),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", ",Object(i.b)("strong",{parentName:"p"},"Distributed Training")," doesn't mean training your model on multiple GPUs or multiple machines, because ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," focuses on tabular datasets (or, structured datasets) which are often not as large as unstructured datasets. Instead, ",Object(i.b)("strong",{parentName:"p"},"Distributed Training")," in ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," means ",Object(i.b)("strong",{parentName:"p"},"training multiple models")," at the same time. This is important because:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"Deep Learning models suffer from randomness, so we need to train multiple models with the same algorithm and calculate the mean / std of the performances to evaluate the algorithm's capacity and stability."),Object(i.b)("li",{parentName:"ul"},"Ensemble these models (which are trained with the same algorithm) can boost the algorithm's performance without making any changes to the algorithm itself."),Object(i.b)("li",{parentName:"ul"},"Parameter searching will be easier & faster.")),Object(i.b)("p",null,"There are two ways to perform distributed training in ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),": through high-level API ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"apis#repeat_with"}),Object(i.b)("inlineCode",{parentName:"a"},"cflearn.repeat_with"))," or through helper class ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"#experiment"}),Object(i.b)("inlineCode",{parentName:"a"},"Experiment")),". We'll introduce their usages in the following sections."),Object(i.b)("h3",{id:"repeat_with"},Object(i.b)("inlineCode",{parentName:"h3"},"repeat_with")),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"repeat_with")," is the general method for training multiple neural networks on fixed datasets. It can be used in either ",Object(i.b)("em",{parentName:"p"},"sequential")," mode or ",Object(i.b)("em",{parentName:"p"},"distributed")," mode. If ",Object(i.b)("em",{parentName:"p"},"distributed")," mode is enabled, it will leverage the helper class ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"#experiment"}),Object(i.b)("inlineCode",{parentName:"a"},"Experiment"))," internally (here are the pseudo codes):"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"experiment = Experiment()\nfor model in models:\n    for _ in range(num_repeat):\n        experiment.add_task(\n            model=model,\n            config=fetch_config(model),\n            data_folder=data_folder,\n        )\n")),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"repeat_with")," is very useful when we want to quickly inspect some statistics of our model (e.g. bias and variance), because you can distributedly perform the same algorithm over the same datasets, and then ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"apis#evaluate"}),Object(i.b)("inlineCode",{parentName:"a"},"cflearn.evaluate"))," will handle the statistics for you:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"results = cflearn.repeat_with(x, y, num_repeat=..., num_jobs=...)\ncflearn.evaluate(x, y, metrics=...)\n")),Object(i.b)("div",{className:"admonition admonition-info alert alert--info"},Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("ul",{parentName:"div"},Object(i.b)("li",{parentName:"ul"},"See ",Object(i.b)("a",Object(a.a)({parentName:"li"},{href:"#benchmarking"}),"Benchmarking")," section for more details."),Object(i.b)("li",{parentName:"ul"},"See ",Object(i.b)("a",Object(a.a)({parentName:"li"},{href:"apis#repeat_with"}),"here")," for the detailed API documentation.")))),Object(i.b)("h3",{id:"experiment"},Object(i.b)("inlineCode",{parentName:"h3"},"Experiment")),Object(i.b)("p",null,"If we want to customize the distributed training process (instead of simply replicating), ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," also provides an ",Object(i.b)("inlineCode",{parentName:"p"},"Experiment")," class for us to control every experiment setting, including:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"Which model should we use for a specific task."),Object(i.b)("li",{parentName:"ul"},"Which dataset should we use for a specific task."),Object(i.b)("li",{parentName:"ul"},"Which configuration should we use for a specific task."),Object(i.b)("li",{parentName:"ul"},"And everything else...")),Object(i.b)("p",null,"Here are two examples that may frequently appear in real scenarios:"),Object(i.b)("h4",{id:"training-multiple-models-on-same-dataset"},"Training Multiple Models on Same Dataset"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\nimport numpy as np\n\nx = np.random.random([1000, 10])\ny = np.random.random([1000, 10])\n\nexperiment = cflearn.Experiment()\n# Since we will train every model on x & y, we should dump them to a `data_folder` first.\n# After that, every model can access this dataset by reading `data_folder`.\ndata_folder = experiment.dump_data_bundle(x, y)\n# We can add task which will train a model on the dataset.\nfor model in ["linear", "fcnn", "tree_dnn"]:\n    # Don\'t forget to specify the `data_folder`!\n    experiment.add_task(model=model, data_folder=data_folder)\n# After adding the tasks, we can run our tasks easily.\n# Remember to specify the `task_loader` if we want to fetch the `pipeline_dict`.\nresults = experiment.run_tasks(task_loader=cflearn.task_loader)\nprint(results.pipelines)  # [FCNN(), LinearModel(), TreeDNN()]\n')),Object(i.b)("h4",{id:"training-same-model-on-different-datasets"},"Training Same Model on Different Datasets"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"import cflearn\nimport numpy as np\n\nx1 = np.random.random([1000, 10])\ny1 = np.random.random([1000, 10])\nx2 = np.random.random([1000, 10])\ny2 = np.random.random([1000, 10])\n\nexperiment = cflearn.Experiment()\n# What's going under the hood here is that `carefree-learn` will \n#  call `dump_data_bundle` internally to manage the datasets\nexperiment.add_task(x1, y1)\nexperiment.add_task(x2, y2)\nresults = experiment.run_tasks(task_loader=cflearn.task_loader)\nprint(results.pipelines)  # [FCNN(), FCNN()]\n")),Object(i.b)("h3",{id:"conclusions"},"Conclusions"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"If we want to train same model on same dataset multiple times, use ",Object(i.b)("inlineCode",{parentName:"li"},"repeat_with"),"."),Object(i.b)("li",{parentName:"ul"},"Otherwise, use ",Object(i.b)("inlineCode",{parentName:"li"},"Experiment"),", and keep in mind that:",Object(i.b)("ul",{parentName:"li"},Object(i.b)("li",{parentName:"ul"},"If we need to share data, use ",Object(i.b)("inlineCode",{parentName:"li"},"dump_data_bundle")," to dump the shared data to a ",Object(i.b)("inlineCode",{parentName:"li"},"data_folder"),", then specify this ",Object(i.b)("inlineCode",{parentName:"li"},"data_folder")," when we call ",Object(i.b)("inlineCode",{parentName:"li"},"add_task"),"."),Object(i.b)("li",{parentName:"ul"},"If we want to add a rather 'isolated' task, simply call ",Object(i.b)("inlineCode",{parentName:"li"},"add_task")," with the corresponding dataset will be fine."),Object(i.b)("li",{parentName:"ul"},"Specify ",Object(i.b)("inlineCode",{parentName:"li"},"task_loader=cflearn.task_loader")," if we want to fetch the ",Object(i.b)("inlineCode",{parentName:"li"},"pipeline_dict"),".")))),Object(i.b)("div",{className:"admonition admonition-info alert alert--info"},Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},Object(i.b)("inlineCode",{parentName:"p"},"Experiment")," supports much more customizations (e.g. customize configurations) than those mentioned above. Please refer to ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"#advanced-benchmarking"}),"Advanced Benchmarking")," for more details."))),Object(i.b)("h2",{id:"benchmarking"},"Benchmarking"),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," has a related repository (namely ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn-benchmark"}),Object(i.b)("inlineCode",{parentName:"a"},"carefree-learn-benchmark")),") which implemented some sophisticated benchmarking functionalities. However, for many common use cases, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," provides ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"apis#repeat_with"}),Object(i.b)("inlineCode",{parentName:"a"},"cflearn.repeat_with"))," and ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"apis#evaluate"}),Object(i.b)("inlineCode",{parentName:"a"},"cflearn.evaluate"))," for quick benchmarking. For example, if we want to compare the ",Object(i.b)("inlineCode",{parentName:"p"},"linear")," model and the ",Object(i.b)("inlineCode",{parentName:"p"},"fcnn")," model by running them ",Object(i.b)("inlineCode",{parentName:"p"},"3")," times, we can simply:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\nimport numpy as np\n\nx = np.random.random([1000, 10])\ny = np.random.random([1000, 1])\n\nif __name__ == "__main__":\n    # Notice that there will always be 2 models training simultaneously with `num_jobs=2`\n    result = cflearn.repeat_with(x, y, models=["linear", "fcnn"], num_repeat=3, num_jobs=2)\n    cflearn.evaluate(x, y, pipelines=result.pipelines)\n')),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       mae                        |                       mse                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          | -- 0.251717 -- | -- 0.002158 -- | -- -0.25387 -- | -- 0.086110 -- | -- 0.002165 -- | -- -0.08827 -- |\n--------------------------------------------------------------------------------------------------------------------------------\n|         linear         |    0.283154    |    0.015341    |    -0.29849    |    0.118122    |    0.016185    |    -0.13430    |\n================================================================================================================================\n")),Object(i.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("ul",{parentName:"div"},Object(i.b)("li",{parentName:"ul"},"It is necessary to wrap codes under ",Object(i.b)("inlineCode",{parentName:"li"},"__main__")," on WINDOWS when running distributed codes."),Object(i.b)("li",{parentName:"ul"},"You might notice that the best results of each column is highlighted with a pair of '--'.")))),Object(i.b)("div",{className:"admonition admonition-caution alert alert--warning"},Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"16",height:"16",viewBox:"0 0 16 16"}),Object(i.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"})))),"caution")),Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"It is not recommended to enable distributed training unless:"),Object(i.b)("ul",{parentName:"div"},Object(i.b)("li",{parentName:"ul"},"There are plenty of tasks that we need to run. "),Object(i.b)("li",{parentName:"ul"},"Running each task is quite costly in time."),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"num_jobs")," could be set to a relatively high value (e.g., ",Object(i.b)("inlineCode",{parentName:"li"},"8"),").")),Object(i.b)("p",{parentName:"div"},"Otherwise the overhead brought by launching distributed training might actually hurt the overall performance."),Object(i.b)("p",{parentName:"div"},"However, there are no 'golden rules' of whether we should use distributed training or not for us to follow, so the best practice is to actually try it out in a smaller scale \ud83e\udd23"))),Object(i.b)("h3",{id:"advanced-benchmarking"},"Advanced Benchmarking"),Object(i.b)("p",null,"In order to serve as a carefree tool, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," is able to perform advanced benchmarking (e.g. compare with scikit-learn models) in a few lines of code (in a distributed mode, if needed)."),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\nimport numpy as np\n\nx = np.random.random([1000, 10])\ny = np.random.random([1000, 1])\n\nexperiment = cflearn.Experiment()\ndata_folder = experiment.dump_data_bundle(x, y)\n\n# Add carefree-learn tasks\nfor model in ["linear", "fcnn", "tree_dnn"]:\n    experiment.add_task(model=model, data_folder=data_folder)\n# Add scikit-learn tasks\nrun_command = "python run_sklearn.py"\nexperiment.add_task(model="svr", run_command=run_command, data_folder=data_folder)\nexperiment.add_task(model="linear_svr", run_command=run_command, data_folder=data_folder)\n')),Object(i.b)("p",null,"Notice that we specified ",Object(i.b)("inlineCode",{parentName:"p"},'run_command="python run_sklearn.py"')," for scikit-learn tasks, which means ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"#experiment"}),Object(i.b)("inlineCode",{parentName:"a"},"Experiment"))," will try to execute this command in the current working directory for training scikit-learn models. The good news is that we do not need to speciy any command line arguments, because ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"#experiment"}),Object(i.b)("inlineCode",{parentName:"a"},"Experiment"))," will handle those for us."),Object(i.b)("p",null,"Here is basically what a ",Object(i.b)("inlineCode",{parentName:"p"},"run_sklearn.py")," should look like:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'import os\nimport pickle\n\nfrom sklearn.svm import SVR\nfrom sklearn.svm import LinearSVR\nfrom cflearn.dist.runs._utils import get_info\n\nif __name__ == "__main__":\n    info = get_info()\n    kwargs = info.kwargs\n    # data\n    data_list = info.data_list\n    x, y = data_list[:2]\n    # model\n    model = kwargs["model"]\n    sk_model = (SVR if model == "svr" else LinearSVR)()\n    # train & save\n    sk_model.fit(x, y.ravel())\n    with open(os.path.join(info.workplace, "sk_model.pkl"), "wb") as f:\n        pickle.dump(sk_model, f)\n')),Object(i.b)("p",null,"With ",Object(i.b)("inlineCode",{parentName:"p"},"run_sklearn.py")," defined, we should run those tasks without ",Object(i.b)("inlineCode",{parentName:"p"},"task_loader")," (because ",Object(i.b)("inlineCode",{parentName:"p"},"sk_model")," cannot be loaded by ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," internally):"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"results = experiment.run_tasks()\n")),Object(i.b)("p",null,"After finished running, we should be able to see this file structure in the current working directory:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-text"}),"|--- __experiment__\n   |--- __data__\n      |-- x.npy\n      |-- y.npy\n   |--- fcnn/0\n      |-- _logs\n      |-- __meta__.json\n      |-- cflearn^_^fcnn^_^0000.zip\n   |--- linear/0\n      |-- ...\n   |--- tree_dnn/0\n      |-- ...\n   |--- linear_svr/0\n      |-- __meta__.json\n      |-- sk_model.pkl\n   |--- svr/0\n      |-- ...\n")),Object(i.b)("p",null,"As we expected, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," models are saved into zip files, while scikit-learn models are saved into ",Object(i.b)("inlineCode",{parentName:"p"},"sk_model.pkl"),". We can further inspect these models with ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.evaluate"),":"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'import os\nimport pickle\n\npipelines = {}\nscikit_patterns = {}\nfor workplace, workplace_key in zip(results.workplaces, results.workplace_keys):\n    model = workplace_key[0]\n    if model not in ["svr", "linear_svr"]:\n        pipelines[model] = cflearn.task_loader(workplace)\n    else:\n        model_file = os.path.join(workplace, "sk_model.pkl")\n        with open(model_file, "rb") as f:\n            sk_model = pickle.load(f)\n            # In `carefree-learn`, we treat labels as column vectors.\n            # So we need to reshape the outputs from the scikit-learn models.\n            sk_predict = lambda x: sk_model.predict(x).reshape([-1, 1])\n            sk_pattern = cflearn.ModelPattern(predict_method=sk_predict)\n            scikit_patterns[model] = sk_pattern\n\ncflearn.evaluate(x, y, pipelines=pipelines, other_patterns=scikit_patterns)\n')),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-text"}),"~~~  [ info ] Results\n================================================================================================================================\n|        metrics         |                       mae                        |                       mse                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          |    0.246332    | -- 0.000000 -- |    -0.24633    |    0.082304    | -- 0.000000 -- |    -0.08230    |\n--------------------------------------------------------------------------------------------------------------------------------\n|         linear         |    0.251605    | -- 0.000000 -- |    -0.25160    |    0.087469    | -- 0.000000 -- |    -0.08746    |\n--------------------------------------------------------------------------------------------------------------------------------\n|       linear_svr       | -- 0.168027 -- | -- 0.000000 -- | -- -0.16802 -- | -- 0.043490 -- | -- 0.000000 -- | -- -0.04349 -- |\n--------------------------------------------------------------------------------------------------------------------------------\n|          svr           | -- 0.168027 -- | -- 0.000000 -- | -- -0.16802 -- | -- 0.043490 -- | -- 0.000000 -- | -- -0.04349 -- |\n--------------------------------------------------------------------------------------------------------------------------------\n|        tree_dnn        |    0.246306    | -- 0.000000 -- |    -0.24630    |    0.082190    | -- 0.000000 -- |    -0.08219    |\n================================================================================================================================\n")),Object(i.b)("h2",{id:"hyper-parameter-optimization-hpo"},"Hyper Parameter Optimization (HPO)"),Object(i.b)("p",null,"Although ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," has already provided an ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"auto-ml"}),Object(i.b)("inlineCode",{parentName:"a"},"AutoML"))," API, we can still play with the ",Object(i.b)("strong",{parentName:"p"},"HPO")," APIs manually:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"import cflearn\nfrom cfdata.tabular import TabularDataset\n \nif __name__ == '__main__':\n    x, y = TabularDataset.iris().xy\n    # Bayesian Optimization (BO) will be used as default\n    hpo = cflearn.tune_with(\n        x, y,\n        task_type=\"clf\",\n        num_repeat=2, num_parallel=0, num_search=10\n    )\n    # We can further train our model with the best hyper-parameters we've obtained:\n    m = cflearn.make(**hpo.best_param).fit(x, y)\n    cflearn.evaluate(x, y, pipelines=m)\n")),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-text"}),"~~~  [ info ] Results\n================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|        0659e09f        |    0.943333    |    0.016667    |    0.926667    |    0.995500    |    0.001967    |    0.993533    |\n--------------------------------------------------------------------------------------------------------------------------------\n|        08a0a030        |    0.796667    |    0.130000    |    0.666667    |    0.969333    |    0.012000    |    0.957333    |\n--------------------------------------------------------------------------------------------------------------------------------\n|        1962285c        |    0.950000    |    0.003333    |    0.946667    |    0.997467    |    0.000533    |    0.996933    |\n--------------------------------------------------------------------------------------------------------------------------------\n|        1eb7f2a0        |    0.933333    |    0.020000    |    0.913333    |    0.994833    |    0.003033    |    0.991800    |\n--------------------------------------------------------------------------------------------------------------------------------\n|        4ed5bb3b        |    0.973333    |    0.013333    |    0.960000    |    0.998733    |    0.000467    |    0.998267    |\n--------------------------------------------------------------------------------------------------------------------------------\n|        5a652f3c        |    0.953333    | -- 0.000000 -- |    0.953333    |    0.997400    |    0.000133    |    0.997267    |\n--------------------------------------------------------------------------------------------------------------------------------\n|        82c35e77        |    0.940000    |    0.020000    |    0.920000    |    0.995467    |    0.002133    |    0.993333    |\n--------------------------------------------------------------------------------------------------------------------------------\n|        a9ef52d0        | -- 0.986667 -- |    0.006667    | -- 0.980000 -- | -- 0.999200 -- | -- 0.000000 -- | -- 0.999200 -- |\n--------------------------------------------------------------------------------------------------------------------------------\n|        ba2e179a        |    0.946667    |    0.026667    |    0.920000    |    0.995633    |    0.001900    |    0.993733    |\n--------------------------------------------------------------------------------------------------------------------------------\n|        ec8c0837        |    0.973333    | -- 0.000000 -- |    0.973333    |    0.998867    |    0.000067    |    0.998800    |\n================================================================================================================================\n\n~~~  [ info ] Best Parameters\n----------------------------------------------------------------------------------------------------\nacc  (a9ef52d0) (0.986667 \xb1 0.006667)\n----------------------------------------------------------------------------------------------------\n{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.005810863965757382}}\n----------------------------------------------------------------------------------------------------\nauc  (a9ef52d0) (0.999200 \xb1 0.000000)\n----------------------------------------------------------------------------------------------------\n{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.005810863965757382}}\n----------------------------------------------------------------------------------------------------\nbest (a9ef52d0)\n----------------------------------------------------------------------------------------------------\n{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.005810863965757382}}\n----------------------------------------------------------------------------------------------------\n\n~~  [ info ] Results\n================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          |    0.980000    |    0.000000    |    0.980000    |    0.998867    |    0.000000    |    0.998867    |\n================================================================================================================================\n")),Object(i.b)("p",null,"You might notice that:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"The final results obtained by ",Object(i.b)("strong",{parentName:"li"},"HPO")," is even better than the stacking ensemble results mentioned above."),Object(i.b)("li",{parentName:"ul"},"We search for ",Object(i.b)("inlineCode",{parentName:"li"},"optimizer")," and ",Object(i.b)("inlineCode",{parentName:"li"},"lr")," as default. In fact, we can manually passed ",Object(i.b)("inlineCode",{parentName:"li"},"params")," into ",Object(i.b)("inlineCode",{parentName:"li"},"cflearn.tune_with"),". If not, then ",Object(i.b)("inlineCode",{parentName:"li"},"carefree-learn")," will execute following codes:")),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'from cftool.ml.param_utils import *\n\nparams = {\n    "optimizer": String(Choice(values=["sgd", "rmsprop", "adam"])),\n    "optimizer_config": {\n        "lr": Float(Exponential(1e-5, 0.1))\n    }\n}\n')))}d.isMDXComponent=!0}}]);