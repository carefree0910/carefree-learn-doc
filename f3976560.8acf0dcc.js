(window.webpackJsonp=window.webpackJsonp||[]).push([[33],{103:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return l})),a.d(t,"metadata",(function(){return s})),a.d(t,"rightToc",(function(){return b})),a.d(t,"default",(function(){return p}));var n=a(3),r=a(7),i=(a(0),a(109)),c=a(117),o=a(118),l={},s={type:"mdx",permalink:"/carefree-learn-doc/",source:"@site/src/pages/index.md"},b=[{value:"Carefree?",id:"carefree",children:[{value:"Machine Learning \ud83d\udcc8",id:"machine-learning-",children:[]},{value:"Computer Vision \ud83d\uddbc\ufe0f",id:"computer-vision-\ufe0f",children:[]}]},{value:"Why carefree-learn?",id:"why-carefree-learn",children:[{value:"Machine Learning \ud83d\udcc8",id:"machine-learning--1",children:[]},{value:"Computer Vision \ud83d\uddbc\ufe0f",id:"computer-vision-\ufe0f-1",children:[]}]},{value:"Citation",id:"citation",children:[]},{value:"License",id:"license",children:[]}],u={rightToc:b};function p(e){var t=e.components,a=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(n.a)({},u,a,{components:t,mdxType:"MDXLayout"}),Object(i.b)("p",null,Object(i.b)("img",Object(n.a)({parentName:"p"},{src:"https://socialify.git.ci/carefree0910/carefree-learn/image?description=1&descriptionEditable=Tabular%20Datasets%20%E2%9D%A4%EF%B8%8F%C2%A0PyTorch&font=Inter&forks=1&issues=1&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fcarefree0910%2Fcarefree-learn-doc%2Fmaster%2Fstatic%2Fimg%2Flogo.min.svg&pattern=Floating%20Cogs&stargazers=1&theme=Light",alt:"carefree-learn"}))),Object(i.b)("p",null,"Deep Learning with ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://pytorch.org/"}),"PyTorch")," made easy \ud83d\ude80 !"),Object(i.b)("h2",{id:"carefree"},"Carefree?"),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," aims to provide ",Object(i.b)("strong",{parentName:"p"},"CAREFREE")," usages for both users and developers. It also provides a ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn-deploy"}),"corresponding repo")," for production."),Object(i.b)("h3",{id:"machine-learning-"},"Machine Learning \ud83d\udcc8"),Object(i.b)(c.a,{defaultValue:"users",values:[{label:"Users",value:"users"},{label:"Developers",value:"developers"}],mdxType:"Tabs"},Object(i.b)(o.a,{value:"users",mdxType:"TabItem"},Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),"import cflearn\nimport numpy as np\n\nx = np.random.random([1000, 10])\ny = np.random.random([1000, 1])\nm = cflearn.ml.CarefreePipeline().fit(x, y)\n"))),Object(i.b)(o.a,{value:"developers",mdxType:"TabItem"},Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"This is a WIP section :D")))),Object(i.b)("h3",{id:"computer-vision-\ufe0f"},"Computer Vision \ud83d\uddbc\ufe0f"),Object(i.b)(c.a,{defaultValue:"users",values:[{label:"Users",value:"users"},{label:"Developers",value:"developers"}],mdxType:"Tabs"},Object(i.b)(o.a,{value:"users",mdxType:"TabItem"},Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'# MNIST classification task with resnet18\n\nimport os\nimport cflearn\n\ntrain_loader, valid_loader = cflearn.cv.get_mnist(transform="to_tensor")\n\nm = cflearn.cv.CarefreePipeline(\n    "clf",\n    {\n        "in_channels": 1,\n        "num_classes": 10,\n        "latent_dim": 512,\n        "encoder1d": "backbone",\n        "encoder1d_configs": {"name": "resnet18"},\n    },\n    loss_name="cross_entropy",\n    metric_names="acc",\n)\nm.fit(train_loader, valid_loader)\n'))),Object(i.b)(o.a,{value:"developers",mdxType:"TabItem"},Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"This is a WIP section :D")))),Object(i.b)("div",{className:"admonition admonition-info alert alert--info"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"Please refer to ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"docs/getting-started/quick-start"}),"Quick Start")," and ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"docs/developer-guides/customization"}),"Build Your Own Models")," for detailed information."))),Object(i.b)("h2",{id:"why-carefree-learn"},"Why carefree-learn?"),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," is a general Deep Learning framework based on PyTorch. Since ",Object(i.b)("inlineCode",{parentName:"p"},"v0.2.x"),", ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," has extended its usage from ",Object(i.b)("strong",{parentName:"p"},"tabular dataset")," to (almost) ",Object(i.b)("strong",{parentName:"p"},"all kinds of dataset"),". In the mean time, the APIs remain (almost) the same as ",Object(i.b)("inlineCode",{parentName:"p"},"v0.1.x"),": still simple, powerful and easy to use!"),Object(i.b)("p",null,"Here are some main advantages that ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," holds:"),Object(i.b)("h3",{id:"machine-learning--1"},"Machine Learning \ud83d\udcc8"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"Provides a ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"https://scikit-learn.org/stable/"}),"scikit-learn"),"-like interface with much more 'carefree' usages, including:",Object(i.b)("ul",{parentName:"li"},Object(i.b)("li",{parentName:"ul"},"Automatically deals with data pre-processing."),Object(i.b)("li",{parentName:"ul"},"Automatically handles datasets saved in files (.txt, .csv)."),Object(i.b)("li",{parentName:"ul"},"Supports ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"docs/user-guides/distributed#distributed-training"}),"Distributed Training"),", which means hyper-parameter tuning can be very efficient in ",Object(i.b)("inlineCode",{parentName:"li"},"carefree-learn"),"."))),Object(i.b)("li",{parentName:"ul"},"Includes some brand new techniques which may boost vanilla Neural Network (NN) performances on tabular datasets, including:",Object(i.b)("ul",{parentName:"li"},Object(i.b)("li",{parentName:"ul"},Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"https://arxiv.org/pdf/1911.05443.pdf"}),Object(i.b)("inlineCode",{parentName:"a"},"TreeDNN")," with ",Object(i.b)("inlineCode",{parentName:"a"},"Dynamic Soft Pruning")),", which makes NN less sensitive to hyper-parameters. "),Object(i.b)("li",{parentName:"ul"},Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"https://arxiv.org/pdf/1911.05441.pdf"}),Object(i.b)("inlineCode",{parentName:"a"},"Deep Distribution Regression (DDR)")),", which is capable of modeling the entire conditional distribution with one single NN model."))),Object(i.b)("li",{parentName:"ul"},"Supports many convenient functionality in deep learning, including:",Object(i.b)("ul",{parentName:"li"},Object(i.b)("li",{parentName:"ul"},"Early stopping."),Object(i.b)("li",{parentName:"ul"},"Model persistence."),Object(i.b)("li",{parentName:"ul"},"Learning rate schedulers."),Object(i.b)("li",{parentName:"ul"},"And more..."))),Object(i.b)("li",{parentName:"ul"},"Full utilization of the WIP ecosystem ",Object(i.b)("inlineCode",{parentName:"li"},"cf*"),", such as:",Object(i.b)("ul",{parentName:"li"},Object(i.b)("li",{parentName:"ul"},Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"https://github.com/carefree0910/carefree-toolkit"}),Object(i.b)("inlineCode",{parentName:"a"},"carefree-toolkit")),": provides a lot of utility classes & functions which are 'stand alone' and can be leveraged in your own projects."),Object(i.b)("li",{parentName:"ul"},Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"https://github.com/carefree0910/carefree-data"}),Object(i.b)("inlineCode",{parentName:"a"},"carefree-data")),": a lightweight tool to read -> convert -> process ",Object(i.b)("strong",{parentName:"li"},"ANY")," tabular datasets. It also utilizes ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"https://cython.org/"}),"cython")," to accelerate critical procedures.")))),Object(i.b)("p",null,"From the above, it comes out that ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," could be treated as a minimal ",Object(i.b)("strong",{parentName:"p"},"Auto"),"matic ",Object(i.b)("strong",{parentName:"p"},"M"),"achine ",Object(i.b)("strong",{parentName:"p"},"L"),"earning (AutoML) solution for tabular datasets when it is fully utilized. However, this is not built on the sacrifice of flexibility. In fact, the functionality we've mentioned are all wrapped into individual modules in ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," and allow users to customize them easily."),Object(i.b)("h3",{id:"computer-vision-\ufe0f-1"},"Computer Vision \ud83d\uddbc\ufe0f"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"Also provides a ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"https://scikit-learn.org/stable/"}),"scikit-learn"),"-like interface with much more 'carefree' usages."),Object(i.b)("li",{parentName:"ul"},"Seamlessly supported ",Object(i.b)("inlineCode",{parentName:"li"},"ddp")," (simply switch ",Object(i.b)("inlineCode",{parentName:"li"},"m.fit(...)")," to ",Object(i.b)("inlineCode",{parentName:"li"},"m.ddp(...)"),")"),Object(i.b)("li",{parentName:"ul"},"Bunch of utility functions for research and production.")),Object(i.b)("h2",{id:"citation"},"Citation"),Object(i.b)("p",null,"If you use ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," in your research, we would greatly appreciate if you cite this library using this Bibtex:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{}),"@misc{carefree-learn,\n  year={2020},\n  author={Yujian He},\n  title={carefree-learn, a minimal Automatic Machine Learning (AutoML) solution for tabular datasets based on PyTorch},\n  howpublished={\\url{https://https://github.com/carefree0910/carefree-learn/}},\n}\n")),Object(i.b)("h2",{id:"license"},"License"),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," is MIT licensed, as found in the ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"docs/about/license"}),Object(i.b)("inlineCode",{parentName:"a"},"LICENSE"))," file."))}p.isMDXComponent=!0},109:function(e,t,a){"use strict";a.d(t,"a",(function(){return u})),a.d(t,"b",(function(){return d}));var n=a(0),r=a.n(n);function i(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function c(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function o(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?c(Object(a),!0).forEach((function(t){i(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):c(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function l(e,t){if(null==e)return{};var a,n,r=function(e,t){if(null==e)return{};var a,n,r={},i=Object.keys(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||(r[a]=e[a]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(r[a]=e[a])}return r}var s=r.a.createContext({}),b=function(e){var t=r.a.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):o(o({},t),e)),a},u=function(e){var t=b(e.components);return r.a.createElement(s.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},m=r.a.forwardRef((function(e,t){var a=e.components,n=e.mdxType,i=e.originalType,c=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),u=b(a),m=n,d=u["".concat(c,".").concat(m)]||u[m]||p[m]||i;return a?r.a.createElement(d,o(o({ref:t},s),{},{components:a})):r.a.createElement(d,o({ref:t},s))}));function d(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var i=a.length,c=new Array(i);c[0]=m;var o={};for(var l in t)hasOwnProperty.call(t,l)&&(o[l]=t[l]);o.originalType=e,o.mdxType="string"==typeof e?e:n,c[1]=o;for(var s=2;s<i;s++)c[s]=a[s];return r.a.createElement.apply(null,c)}return r.a.createElement.apply(null,a)}m.displayName="MDXCreateElement"},110:function(e,t,a){"use strict";function n(e){var t,a,r="";if("string"==typeof e||"number"==typeof e)r+=e;else if("object"==typeof e)if(Array.isArray(e))for(t=0;t<e.length;t++)e[t]&&(a=n(e[t]))&&(r&&(r+=" "),r+=a);else for(t in e)e[t]&&(r&&(r+=" "),r+=t);return r}t.a=function(){for(var e,t,a=0,r="";a<arguments.length;)(e=arguments[a++])&&(t=n(e))&&(r&&(r+=" "),r+=t);return r}},114:function(e,t,a){"use strict";var n=a(0),r=a(115);t.a=function(){var e=Object(n.useContext)(r.a);if(null==e)throw new Error("`useUserPreferencesContext` is used outside of `Layout` Component.");return e}},115:function(e,t,a){"use strict";var n=a(0),r=Object(n.createContext)(void 0);t.a=r},117:function(e,t,a){"use strict";var n=a(0),r=a.n(n),i=a(114),c=a(110),o=a(52),l=a.n(o),s=37,b=39;t.a=function(e){var t=e.lazy,a=e.block,o=e.children,u=e.defaultValue,p=e.values,m=e.groupId,d=e.className,f=Object(i.a)(),O=f.tabGroupChoices,h=f.setTabGroupChoices,j=Object(n.useState)(u),v=j[0],g=j[1];if(null!=m){var N=O[m];null!=N&&N!==v&&p.some((function(e){return e.value===N}))&&g(N)}var y=function(e){g(e),null!=m&&h(m,e)},w=[];return r.a.createElement("div",null,r.a.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:Object(c.a)("tabs",{"tabs--block":a},d)},p.map((function(e){var t=e.value,a=e.label;return r.a.createElement("li",{role:"tab",tabIndex:0,"aria-selected":v===t,className:Object(c.a)("tabs__item",l.a.tabItem,{"tabs__item--active":v===t}),key:t,ref:function(e){return w.push(e)},onKeyDown:function(e){!function(e,t,a){switch(a.keyCode){case b:!function(e,t){var a=e.indexOf(t)+1;e[a]?e[a].focus():e[0].focus()}(e,t);break;case s:!function(e,t){var a=e.indexOf(t)-1;e[a]?e[a].focus():e[e.length-1].focus()}(e,t)}}(w,e.target,e)},onFocus:function(){return y(t)},onClick:function(){y(t)}},a)}))),t?Object(n.cloneElement)(o.filter((function(e){return e.props.value===v}))[0],{className:"margin-vert--md"}):r.a.createElement("div",{className:"margin-vert--md"},o.map((function(e,t){return Object(n.cloneElement)(e,{key:t,hidden:e.props.value!==v})}))))}},118:function(e,t,a){"use strict";var n=a(3),r=a(0),i=a.n(r);t.a=function(e){var t=e.children,a=e.hidden,r=e.className;return i.a.createElement("div",Object(n.a)({role:"tabpanel"},{hidden:a,className:r}),t)}}}]);