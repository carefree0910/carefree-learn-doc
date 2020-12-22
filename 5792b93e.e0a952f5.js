(window.webpackJsonp=window.webpackJsonp||[]).push([[16],{109:function(e,t,a){"use strict";a.d(t,"a",(function(){return b})),a.d(t,"b",(function(){return m}));var n=a(0),r=a.n(n);function i(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function o(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function c(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?o(Object(a),!0).forEach((function(t){i(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):o(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function l(e,t){if(null==e)return{};var a,n,r=function(e,t){if(null==e)return{};var a,n,r={},i=Object.keys(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||(r[a]=e[a]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(r[a]=e[a])}return r}var s=r.a.createContext({}),p=function(e){var t=r.a.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):c(c({},t),e)),a},b=function(e){var t=p(e.components);return r.a.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},u=r.a.forwardRef((function(e,t){var a=e.components,n=e.mdxType,i=e.originalType,o=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),b=p(a),u=n,m=b["".concat(o,".").concat(u)]||b[u]||d[u]||i;return a?r.a.createElement(m,c(c({ref:t},s),{},{components:a})):r.a.createElement(m,c({ref:t},s))}));function m(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var i=a.length,o=new Array(i);o[0]=u;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:n,o[1]=c;for(var s=2;s<i;s++)o[s]=a[s];return r.a.createElement.apply(null,o)}return r.a.createElement.apply(null,a)}u.displayName="MDXCreateElement"},85:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return o})),a.d(t,"metadata",(function(){return c})),a.d(t,"rightToc",(function(){return l})),a.d(t,"default",(function(){return p}));var n=a(3),r=a(7),i=(a(0),a(109)),o={id:"auto-ml",title:"AutoML"},c={unversionedId:"user-guides/auto-ml",id:"user-guides/auto-ml",isDocsHomePage:!1,title:"AutoML",description:"carefree-learn provides cflearn.Auto API for out-of-the-box usages.",source:"@site/docs/user-guides/auto-ml.md",slug:"/user-guides/auto-ml",permalink:"/carefree-learn-doc/docs/user-guides/auto-ml",version:"current",lastUpdatedAt:1608608613,sidebar:"docs",previous:{title:"APIs",permalink:"/carefree-learn-doc/docs/user-guides/apis"},next:{title:"Distributed",permalink:"/carefree-learn-doc/docs/user-guides/distributed"}},l=[{value:"Explained",id:"explained",children:[]},{value:"Configurations",id:"configurations",children:[{value:"Define Model Space",id:"define-model-space",children:[]},{value:"Define Extra Configurations",id:"define-extra-configurations",children:[]}]},{value:"Production",id:"production",children:[]}],s={rightToc:l};function p(e){var t=e.components,a=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(n.a)({},s,a,{components:t,mdxType:"MDXLayout"}),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," provides ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Auto")," API for out-of-the-box usages."),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\n\nfrom cfdata.tabular import *\n\n# prepare iris dataset\niris = TabularDataset.iris()\niris = TabularData.from_dataset(iris)\n# split 10% of the data as validation data\nsplit = iris.split(0.1)\ntrain, valid = split.remained, split.split\nx_tr, y_tr = train.processed.xy\nx_cv, y_cv = valid.processed.xy\ndata = x_tr, y_tr, x_cv, y_cv\n\n\nif __name__ == \'__main__\':\n    # standard usage\n    fcnn = cflearn.make().fit(*data)\n\n    # \'overfit\' validation set\n    # * `clf` indicates this is a classification task\n    # * for regression tasks, use `reg` instead\n    auto = cflearn.Auto("clf").fit(*data, num_jobs=2)\n\n    # evaluate manually\n    predictions = auto.predict(x_cv)\n    print("accuracy:", (y_cv == predictions).mean())\n\n    # evaluate with `cflearn`\n    cflearn.evaluate(\n        x_cv,\n        y_cv,\n        pipelines=fcnn,\n        other_patterns={"auto": auto.pattern},\n    )\n')),Object(i.b)("p",null,"Which yields"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          auto          | -- 1.000000 -- | -- 0.000000 -- | -- 1.000000 -- | -- 1.000000 -- | -- 0.000000 -- | -- 1.000000 -- |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          |    0.933333    | -- 0.000000 -- |    0.933333    |    0.993333    | -- 0.000000 -- |    0.993333    |\n================================================================================================================================\n")),Object(i.b)("h2",{id:"explained"},"Explained"),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Auto.fit")," will run through the following steps:"),Object(i.b)("ol",null,Object(i.b)("li",{parentName:"ol"},"define the model space automatically (or manually; see ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"#define-model-space"}),"Define Model Space")," for more details)."),Object(i.b)("li",{parentName:"ol"},"fetch pre-defined hyper-parameters search space of each model from ",Object(i.b)("inlineCode",{parentName:"li"},"OptunaPresetParams")," (and inject manual configurations, if provided; see ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"#define-extra-configurations"}),"Define Extra Configurations")," for more details)."),Object(i.b)("li",{parentName:"ol"},"leverage ",Object(i.b)("inlineCode",{parentName:"li"},"optuna")," with ",Object(i.b)("inlineCode",{parentName:"li"},"cflearn.optuna_tune")," to perform hyper-parameters optimization."),Object(i.b)("li",{parentName:"ol"},"use searched hyper-parameters to train each model multiple times (separately)."),Object(i.b)("li",{parentName:"ol"},"ensemble all trained models (with ",Object(i.b)("inlineCode",{parentName:"li"},"cflearn.Ensemble.stacking"),")."),Object(i.b)("li",{parentName:"ol"},"record all these results to corresponding attributes.")),Object(i.b)("p",null,"So after ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Auto.fit"),", we can perform visualizations provided by ",Object(i.b)("inlineCode",{parentName:"p"},"optuna")," easily:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'export_folder = "iris_vis"\nauto.plot_param_importances("fcnn", export_folder=export_folder)\nauto.plot_intermediate_values("fcnn", export_folder=export_folder)\n')),Object(i.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"It is also worth mentioning that we can pass file datasets into ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Auto")," as well. See ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/3fb03dbfc3e2b494f2ab03b9d8f07683fe30e7ef/tests/usages/test_basic.py#L221"}),"test_auto_file")," for more details."))),Object(i.b)("h2",{id:"configurations"},"Configurations"),Object(i.b)("p",null,"Although ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Auto")," could achieve acceptable performances, we can manually adjust its behaviour for even better ones as well."),Object(i.b)("h3",{id:"define-model-space"},"Define Model Space"),Object(i.b)("p",null,"Model space could be defined by specifying the ",Object(i.b)("inlineCode",{parentName:"p"},"models"),":"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'auto = cflearn.Auto(..., models="fcnn")\n')),Object(i.b)("p",null,"or"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'auto = cflearn.Auto(..., models=["linear", "fcnn"])\n')),Object(i.b)("div",{className:"admonition admonition-info alert alert--info"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"By default, ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Auto")," will use a large model space and hope for the best:"),Object(i.b)("pre",{parentName:"div"},Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'if models == "auto":\n    models = ["linear", "fcnn", "tree_dnn"]\n    parsed_task_type = parse_task_type(task_type)\n    # time series tasks\n    if parsed_task_type.is_ts:\n        models += ["rnn", "transformer"]\n    # classification tasks\n    elif parsed_task_type.is_clf:\n        models += ["nnb", "ndt"]\n    # regression tasks\n    else:\n        models.append("ddr")\n')),Object(i.b)("p",{parentName:"div"},"We recommend to use ",Object(i.b)("inlineCode",{parentName:"p"},'models="fcnn"')," before actually dive into this bunch of models \ud83e\udd23"))),Object(i.b)("h3",{id:"define-extra-configurations"},"Define Extra Configurations"),Object(i.b)("p",null,"If we want to change some default behaviours of ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Auto"),", we can specify the ",Object(i.b)("inlineCode",{parentName:"p"},"extra_configs"),":"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),"auto.fit(..., extra_config={...})\n")),Object(i.b)("p",null,"And the usage of ",Object(i.b)("inlineCode",{parentName:"p"},"extra_config")," should be equivalent to the usage of ",Object(i.b)("inlineCode",{parentName:"p"},"config")," in ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"../getting-started/configurations#make"}),Object(i.b)("inlineCode",{parentName:"a"},"make"))," API."),Object(i.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},Object(i.b)("inlineCode",{parentName:"p"},"extra_config")," is not able to overwrite the hyperparameters generated by the search space, so in fact the options we can play with it are limited \ud83e\udd23"))),Object(i.b)("h2",{id:"production"},"Production"),Object(i.b)("p",null,"What's facinating is that we can pack the models trained by ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Auto")," into a zip file for production:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'auto.pack("pack")\n')),Object(i.b)("p",null,"Please refer to ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"production#automl-in-production"}),"AutoML in Production")," for more details."))}p.isMDXComponent=!0}}]);