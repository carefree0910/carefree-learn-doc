(window.webpackJsonp=window.webpackJsonp||[]).push([[8],{105:function(e,t,a){"use strict";a.d(t,"a",(function(){return d})),a.d(t,"b",(function(){return u}));var n=a(0),i=a.n(n);function r(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function o(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function c(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?o(Object(a),!0).forEach((function(t){r(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):o(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function s(e,t){if(null==e)return{};var a,n,i=function(e,t){if(null==e)return{};var a,n,i={},r=Object.keys(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||(i[a]=e[a]);return i}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(i[a]=e[a])}return i}var l=i.a.createContext({}),b=function(e){var t=i.a.useContext(l),a=t;return e&&(a="function"==typeof e?e(t):c(c({},t),e)),a},d=function(e){var t=b(e.components);return i.a.createElement(l.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return i.a.createElement(i.a.Fragment,{},t)}},m=i.a.forwardRef((function(e,t){var a=e.components,n=e.mdxType,r=e.originalType,o=e.parentName,l=s(e,["components","mdxType","originalType","parentName"]),d=b(a),m=n,u=d["".concat(o,".").concat(m)]||d[m]||p[m]||r;return a?i.a.createElement(u,c(c({ref:t},l),{},{components:a})):i.a.createElement(u,c({ref:t},l))}));function u(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var r=a.length,o=new Array(r);o[0]=m;var c={};for(var s in t)hasOwnProperty.call(t,s)&&(c[s]=t[s]);c.originalType=e,c.mdxType="string"==typeof e?e:n,o[1]=c;for(var l=2;l<r;l++)o[l]=a[l];return i.a.createElement.apply(null,o)}return i.a.createElement.apply(null,a)}m.displayName="MDXCreateElement"},73:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return o})),a.d(t,"metadata",(function(){return c})),a.d(t,"rightToc",(function(){return s})),a.d(t,"default",(function(){return b}));var n=a(3),i=a(7),r=(a(0),a(105)),o={id:"introduction",title:"Introduction",slug:"/"},c={unversionedId:"introduction",id:"introduction",isDocsHomePage:!1,title:"Introduction",description:"Advantages",source:"@site/docs\\introduction.md",slug:"/",permalink:"/carefree-learn-doc/docs/",version:"current",lastUpdatedAt:1605851411,sidebar:"docs",next:{title:"Design Principles",permalink:"/carefree-learn-doc/docs/design-principles"}},s=[{value:"Advantages",id:"advantages",children:[]},{value:"Configurations",id:"configurations",children:[]},{value:"Data Loading Strategy",id:"data-loading-strategy",children:[]},{value:"Terminologies",id:"terminologies",children:[{value:"step",id:"step",children:[]},{value:"epoch",id:"epoch",children:[]},{value:"batch_size",id:"batch_size",children:[]},{value:"config",id:"config",children:[]},{value:"increment_config",id:"increment_config",children:[]},{value:"forward",id:"forward",children:[]},{value:"task_type",id:"task_type",children:[]},{value:"tr, cv &amp; te",id:"tr-cv--te",children:[]},{value:"metrics",id:"metrics",children:[]},{value:"optimizers",id:"optimizers",children:[]}]}],l={rightToc:s};function b(e){var t=e.components,a=Object(i.a)(e,["components"]);return Object(r.b)("wrapper",Object(n.a)({},l,a,{components:t,mdxType:"MDXLayout"}),Object(r.b)("h2",{id:"advantages"},"Advantages"),Object(r.b)("p",null,"Like many similar projects, ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn")," can be treated as a high-level library to help with training neural networks in PyTorch. However, ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn")," does less and more than that."),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"carefree-learn")," focuses on tabular (structured) datasets, instead of unstructured datasets (e.g. NLP datasets or CV datasets)."),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"carefree-learn")," provides an end-to-end pipeline on tabular datasets, including ",Object(r.b)("strong",{parentName:"li"},"AUTOMATICALLY")," deal with (this part is mainly handled by ",Object(r.b)("a",Object(n.a)({parentName:"li"},{href:"https://github.com/carefree0910/carefree-data"}),Object(r.b)("inlineCode",{parentName:"a"},"carefree-data")),", though):",Object(r.b)("ul",{parentName:"li"},Object(r.b)("li",{parentName:"ul"},"Detection of redundant feature columns which can be excluded (all SAME, all DIFFERENT, etc)."),Object(r.b)("li",{parentName:"ul"},"Detection of feature columns types (whether a feature column is string column / numerical column / categorical column)."),Object(r.b)("li",{parentName:"ul"},"Imputation of missing values."),Object(r.b)("li",{parentName:"ul"},"Encoding of string columns and categorical columns (Embedding or One Hot Encoding)."),Object(r.b)("li",{parentName:"ul"},"Pre-processing of numerical columns (Normalize, Min Max, etc.)."),Object(r.b)("li",{parentName:"ul"},"And much more..."))),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"carefree-learn")," can help you deal with almost ",Object(r.b)("strong",{parentName:"li"},"ANY")," kind of tabular datasets, no matter how ",Object(r.b)("em",{parentName:"li"},"dirty")," and ",Object(r.b)("em",{parentName:"li"},"messy")," it is. It can be either trained directly with some numpy arrays, or trained indirectly with some files locate on your machine. This makes ",Object(r.b)("inlineCode",{parentName:"li"},"carefree-learn")," stand out from similar projects."),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"carefree-learn")," is ",Object(r.b)("strong",{parentName:"li"},"highly customizable")," for developers. We have already wrapped (almost) every single functionality / process into a single module (a Python class), and they can be replaced or enhanced either directly from source codes or from local codes with the help of some pre-defined functions provided by ",Object(r.b)("inlineCode",{parentName:"li"},"carefree-learn")," (see ",Object(r.b)("a",Object(n.a)({parentName:"li"},{href:"design-principles#registration"}),Object(r.b)("inlineCode",{parentName:"a"},"Registration")),")."),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"carefree-learn")," supports easy-to-use saving and loading. By default, everything will be wrapped into a zip file!"),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"carefree-learn")," supports ",Object(r.b)("a",Object(n.a)({parentName:"li"},{href:"user-guides/distributed#distributed-training"}),Object(r.b)("inlineCode",{parentName:"a"},"Distributed Training")),".")),Object(r.b)("div",{className:"admonition admonition-info alert alert--info"},Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(r.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"From the discriptions above, you might notice that ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn")," is more of a minimal ",Object(r.b)("strong",{parentName:"p"},"Automatic Machine Learning")," (AutoML) solution than a pure Machine Learning package."))),Object(r.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(r.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"When we say ",Object(r.b)("strong",{parentName:"p"},"ANY"),", it means that ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn")," can even train on one single sample."),Object(r.b)("details",null,Object(r.b)("summary",null,Object(r.b)("b",null,"For example")),Object(r.b)("p",null,Object(r.b)("pre",{parentName:"div"},Object(r.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\n\ntoy = cflearn.make_toy_model()\ndata = toy.tr_data.converted\nprint(f"x={data.x}, y={data.y}")  # x=[[0.]], y=[[1.]]\n')))),Object(r.b)("br",null),Object(r.b)("p",{parentName:"div"},"This is especially useful when we need to do unittests or to verify whether our custom modules (e.g. custom pre-processes) are correctly integrated into ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn"),"."),Object(r.b)("details",null,Object(r.b)("summary",null,Object(r.b)("b",null,"For example")),Object(r.b)("p",null,Object(r.b)("pre",{parentName:"div"},Object(r.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\nimport numpy as np\n\n# here we implement a custom processor\n@cflearn.register_processor("plus_one")\nclass PlusOne(cflearn.Processor):\n    @property\n    def input_dim(self) -> int:\n        return 1\n\n    @property\n    def output_dim(self) -> int:\n        return 1\n\n    def fit(self, columns: np.ndarray) -> cflearn.Processor:\n        return self\n\n    def _process(self, columns: np.ndarray) -> np.ndarray:\n        return columns + 1\n\n    def _recover(self, processed_columns: np.ndarray) -> np.ndarray:\n        return processed_columns - 1\n\n# we need to specify that we use the custom process method to process our labels\nconfig = {"data_config": {"label_process_method": "plus_one"}}\ntoy = cflearn.make_toy_model(config=config)\ny = toy.tr_data.converted.y\nprocessed_y = toy.tr_data.processed.y\nprint(f"y={y}, new_y={processed_y}")  # y=[[1.]], new_y=[[2.]]\n')))))),Object(r.b)("p",null,"There is one more thing we'd like to mention: ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn")," is ",Object(r.b)("em",{parentName:"p"},Object(r.b)("a",Object(n.a)({parentName:"em"},{href:"https://pandas.pydata.org/"}),"Pandas"),"-free"),". The reasons why we excluded ",Object(r.b)("a",Object(n.a)({parentName:"p"},{href:"https://pandas.pydata.org/"}),"Pandas")," are listed in ",Object(r.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-data"}),Object(r.b)("inlineCode",{parentName:"a"},"carefree-data")),"."),Object(r.b)("h2",{id:"configurations"},"Configurations"),Object(r.b)("p",null,"In ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn"),", we have few args and kwargs in each module. Instead, we'll use one single argument ",Object(r.b)("a",Object(n.a)({parentName:"p"},{href:"getting-started/configurations#environment"}),Object(r.b)("inlineCode",{parentName:"a"},"Environment"))," to handle the configurations. That's why we can easily support JSON configuration, which is very useful when you need to share your models to others or reproduce others' works."),Object(r.b)("h2",{id:"data-loading-strategy"},"Data Loading Strategy"),Object(r.b)("p",null,"Since ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn")," focuses on tabular datasets, the data loading strategy is very different from unstructured datasets' strategy. For instance, it is quite common that a CV dataset is a bunch of pictures located in a folder, and we will either read them sequentially or read them in parallel. Nowadays, almost every famous deep learning framework has their own solution to load unstructured datasets efficiently, e.g. PyTorch officially implements ",Object(r.b)("a",Object(n.a)({parentName:"p"},{href:"https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"}),"DataLoader")," to support multi-process loading and other features."),Object(r.b)("p",null,"Although we know that RAM speed is (almost) always faster than I/O operations, we still prefer leveraging multi-process to read files than loading them all into RAM at once. This is because unstructured datasets are often too large to allocate them all to RAM. However, when it comes to tabular datasets, we prefer to load everything into RAM at the very beginning. The main reasons are listed below:"),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},"Tabular datasets are often quite small and are able to put into RAM at once."),Object(r.b)("li",{parentName:"ul"},"Network structures for tabular datasets are often much smaller, which means using multi-process loading will cause a much heavier overhead."),Object(r.b)("li",{parentName:"ul"},"We need to take ",Object(r.b)("a",Object(n.a)({parentName:"li"},{href:"user-guides/distributed#distributed-training"}),Object(r.b)("inlineCode",{parentName:"a"},"Distributed Training"))," into account. If we stick to multi-process loading, there would be too many threads in the pool which is not a good practice.")),Object(r.b)("h2",{id:"terminologies"},"Terminologies"),Object(r.b)("p",null,"In ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn"),", there are some frequently used terminologies, and we will introduce them in this section. If you are confused by some other terminologies in ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn")," when you are using it, feel free to edit this list:"),Object(r.b)("h3",{id:"step"},"step"),Object(r.b)("p",null,"One ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"step"))," in the training process means that one mini-batch passed through our model."),Object(r.b)("h3",{id:"epoch"},"epoch"),Object(r.b)("p",null,"In most deep learning processes, training is structured into epochs. An epoch is one iteration over the entire input data, which is constructed by several ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"step")),"s."),Object(r.b)("h3",{id:"batch_size"},"batch_size"),Object(r.b)("p",null,"It is a good practice to slice the data into smaller batches and iterates over these batches during training, and ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"batch_size"))," specifies the size of each batch. Be aware that the last batch may be smaller if the total number of samples is not divisible by the ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"batch_size")),"."),Object(r.b)("h3",{id:"config"},"config"),Object(r.b)("p",null,"A ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"config"))," indicates the main part (or, the shared part) of the configuration."),Object(r.b)("h3",{id:"increment_config"},"increment_config"),Object(r.b)("p",null,"An ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"increment_config"))," indicates the configurations that you want to update on ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"config")),"."),Object(r.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(r.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"This is very useful when you only want to tune a single configuration and yet you have tons of other configurations need to be fixed. In this case, you can set other configurations as ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"config")),", and adjust the target configuration in ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"increment_config")),"."))),Object(r.b)("h3",{id:"forward"},"forward"),Object(r.b)("p",null,"A ",Object(r.b)("strong",{parentName:"p"},Object(r.b)("inlineCode",{parentName:"strong"},"forward"))," method is a common method required by (almost) all PyTorch modules."),Object(r.b)("div",{className:"admonition admonition-info alert alert--info"},Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(r.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},Object(r.b)("a",Object(n.a)({parentName:"p"},{href:"https://discuss.pytorch.org/t/about-the-nn-module-forward/20858"}),"Here")," is a nice discussion."))),Object(r.b)("h3",{id:"task_type"},"task_type"),Object(r.b)("p",null,"We use ",Object(r.b)("inlineCode",{parentName:"p"},'task_type = "clf"')," to indicate a classification task, and ",Object(r.b)("inlineCode",{parentName:"p"},'task_type = "reg"')," to indicate a regression task."),Object(r.b)("div",{className:"admonition admonition-info alert alert--info"},Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(r.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"And we'll convert them into ",Object(r.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-data/blob/82f158be82ced404a1f4ac37e7a669a50470b109/cfdata/tabular/misc.py#L126"}),"cfdata.tabular.TaskTypes")," under the hood."))),Object(r.b)("h3",{id:"tr-cv--te"},"tr, cv & te"),Object(r.b)("p",null,"In most cases, we use:"),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"x_tr"),", ",Object(r.b)("inlineCode",{parentName:"li"},"x_cv")," and ",Object(r.b)("inlineCode",{parentName:"li"},"x_te")," to represent ",Object(r.b)("inlineCode",{parentName:"li"},"training"),", ",Object(r.b)("inlineCode",{parentName:"li"},"cross validation")," and ",Object(r.b)("inlineCode",{parentName:"li"},"test")," ",Object(r.b)("strong",{parentName:"li"},"features"),"."),Object(r.b)("li",{parentName:"ul"},Object(r.b)("inlineCode",{parentName:"li"},"y_tr"),", ",Object(r.b)("inlineCode",{parentName:"li"},"y_cv")," and ",Object(r.b)("inlineCode",{parentName:"li"},"y_te")," to represent ",Object(r.b)("inlineCode",{parentName:"li"},"training"),", ",Object(r.b)("inlineCode",{parentName:"li"},"cross validation")," and ",Object(r.b)("inlineCode",{parentName:"li"},"test")," ",Object(r.b)("strong",{parentName:"li"},"labels"),".")),Object(r.b)("h3",{id:"metrics"},"metrics"),Object(r.b)("p",null,"Although ",Object(r.b)("inlineCode",{parentName:"p"},"losses")," are what we optimize directly during training, ",Object(r.b)("inlineCode",{parentName:"p"},"metrics")," are what we ",Object(r.b)("em",{parentName:"p"},"actually")," want to optimize (e.g. ",Object(r.b)("inlineCode",{parentName:"p"},"acc"),", ",Object(r.b)("inlineCode",{parentName:"p"},"auc"),", ",Object(r.b)("inlineCode",{parentName:"p"},"f1-score"),", etc.). Sometimes we may want to take multiple ",Object(r.b)("inlineCode",{parentName:"p"},"metrics")," into consideration, and we may also want to eliminate the fluctuation comes with mini-batch training by applying EMA on the metrics."),Object(r.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(r.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"Please refer to ",Object(r.b)("a",Object(n.a)({parentName:"p"},{href:"getting-started/configurations#metrics"}),"metrics")," and see how to customize the behaviour of ",Object(r.b)("inlineCode",{parentName:"p"},"metrics")," in ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn"),"."))),Object(r.b)("h3",{id:"optimizers"},"optimizers"),Object(r.b)("p",null,"In PyTorch (and other deep learning framework) we have ",Object(r.b)("inlineCode",{parentName:"p"},"optimizers")," to ",Object(r.b)("em",{parentName:"p"},"optimize")," the parameters of our model. We sometimes need to divide the parameters into several groups and optimize them individually (which is quite common in GANs)."),Object(r.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(r.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(r.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"Please refer to ",Object(r.b)("a",Object(n.a)({parentName:"p"},{href:"getting-started/configurations#optimizers"}),"optimizers")," and see how to control the behaviour of ",Object(r.b)("inlineCode",{parentName:"p"},"optimizers")," in ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn"),"."))))}b.isMDXComponent=!0}}]);