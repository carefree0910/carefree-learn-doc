(window.webpackJsonp=window.webpackJsonp||[]).push([[17],{109:function(e,t,a){"use strict";a.d(t,"a",(function(){return p})),a.d(t,"b",(function(){return u}));var n=a(0),r=a.n(n);function i(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function c(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function l(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?c(Object(a),!0).forEach((function(t){i(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):c(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function o(e,t){if(null==e)return{};var a,n,r=function(e,t){if(null==e)return{};var a,n,r={},i=Object.keys(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||(r[a]=e[a]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(r[a]=e[a])}return r}var s=r.a.createContext({}),b=function(e){var t=r.a.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):l(l({},t),e)),a},p=function(e){var t=b(e.components);return r.a.createElement(s.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},d=r.a.forwardRef((function(e,t){var a=e.components,n=e.mdxType,i=e.originalType,c=e.parentName,s=o(e,["components","mdxType","originalType","parentName"]),p=b(a),d=n,u=p["".concat(c,".").concat(d)]||p[d]||m[d]||i;return a?r.a.createElement(u,l(l({ref:t},s),{},{components:a})):r.a.createElement(u,l({ref:t},s))}));function u(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var i=a.length,c=new Array(i);c[0]=d;var l={};for(var o in t)hasOwnProperty.call(t,o)&&(l[o]=t[o]);l.originalType=e,l.mdxType="string"==typeof e?e:n,c[1]=l;for(var s=2;s<i;s++)c[s]=a[s];return r.a.createElement.apply(null,c)}return r.a.createElement.apply(null,a)}d.displayName="MDXCreateElement"},110:function(e,t,a){"use strict";function n(e){var t,a,r="";if("string"==typeof e||"number"==typeof e)r+=e;else if("object"==typeof e)if(Array.isArray(e))for(t=0;t<e.length;t++)e[t]&&(a=n(e[t]))&&(r&&(r+=" "),r+=a);else for(t in e)e[t]&&(r&&(r+=" "),r+=t);return r}t.a=function(){for(var e,t,a=0,r="";a<arguments.length;)(e=arguments[a++])&&(t=n(e))&&(r&&(r+=" "),r+=t);return r}},114:function(e,t,a){"use strict";var n=a(0),r=a(115);t.a=function(){var e=Object(n.useContext)(r.a);if(null==e)throw new Error("`useUserPreferencesContext` is used outside of `Layout` Component.");return e}},115:function(e,t,a){"use strict";var n=a(0),r=Object(n.createContext)(void 0);t.a=r},117:function(e,t,a){"use strict";var n=a(0),r=a.n(n),i=a(114),c=a(110),l=a(52),o=a.n(l),s=37,b=39;t.a=function(e){var t=e.lazy,a=e.block,l=e.children,p=e.defaultValue,m=e.values,d=e.groupId,u=e.className,f=Object(i.a)(),O=f.tabGroupChoices,j=f.setTabGroupChoices,g=Object(n.useState)(p),v=g[0],h=g[1];if(null!=d){var N=O[d];null!=N&&N!==v&&m.some((function(e){return e.value===N}))&&h(N)}var y=function(e){h(e),null!=d&&j(d,e)},w=[];return r.a.createElement("div",null,r.a.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:Object(c.a)("tabs",{"tabs--block":a},u)},m.map((function(e){var t=e.value,a=e.label;return r.a.createElement("li",{role:"tab",tabIndex:0,"aria-selected":v===t,className:Object(c.a)("tabs__item",o.a.tabItem,{"tabs__item--active":v===t}),key:t,ref:function(e){return w.push(e)},onKeyDown:function(e){!function(e,t,a){switch(a.keyCode){case b:!function(e,t){var a=e.indexOf(t)+1;e[a]?e[a].focus():e[0].focus()}(e,t);break;case s:!function(e,t){var a=e.indexOf(t)-1;e[a]?e[a].focus():e[e.length-1].focus()}(e,t)}}(w,e.target,e)},onFocus:function(){return y(t)},onClick:function(){y(t)}},a)}))),t?Object(n.cloneElement)(l.filter((function(e){return e.props.value===v}))[0],{className:"margin-vert--md"}):r.a.createElement("div",{className:"margin-vert--md"},l.map((function(e,t){return Object(n.cloneElement)(e,{key:t,hidden:e.props.value!==v})}))))}},118:function(e,t,a){"use strict";var n=a(3),r=a(0),i=a.n(r);t.a=function(e){var t=e.children,a=e.hidden,r=e.className;return i.a.createElement("div",Object(n.a)({role:"tabpanel"},{hidden:a,className:r}),t)}},86:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return o})),a.d(t,"metadata",(function(){return s})),a.d(t,"rightToc",(function(){return b})),a.d(t,"default",(function(){return m}));var n=a(3),r=a(7),i=(a(0),a(109)),c=a(117),l=a(118),o={id:"quick-start",title:"Quick Start"},s={unversionedId:"getting-started/quick-start",id:"getting-started/quick-start",isDocsHomePage:!1,title:"Quick Start",description:"In carefree-learn, it's easy to train and serialize a model for all tasks.",source:"@site/docs/getting-started/quick-start.md",slug:"/getting-started/quick-start",permalink:"/carefree-learn-doc/docs/getting-started/quick-start",version:"current",lastUpdatedAt:1628423348,sidebar:"docs",previous:{title:"Installation",permalink:"/carefree-learn-doc/docs/getting-started/installation"},next:{title:"Configurations",permalink:"/carefree-learn-doc/docs/getting-started/configurations"}},b=[{value:"Training",id:"training",children:[{value:"Machine Learning \ud83d\udcc8",id:"machine-learning-",children:[]},{value:"Computer Vision \ud83d\uddbc\ufe0f",id:"computer-vision-\ufe0f",children:[]}]},{value:"Serializing",id:"serializing",children:[{value:"Saving",id:"saving",children:[]},{value:"Loading",id:"loading",children:[]}]}],p={rightToc:b};function m(e){var t=e.components,a=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(n.a)({},p,a,{components:t,mdxType:"MDXLayout"}),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", it's easy to train and serialize a model ",Object(i.b)("strong",{parentName:"p"},"for all tasks"),"."),Object(i.b)("h2",{id:"training"},"Training"),Object(i.b)("h3",{id:"machine-learning-"},"Machine Learning \ud83d\udcc8"),Object(i.b)(c.a,{defaultValue:"numpy",values:[{label:"With NumPy",value:"numpy"},{label:"With File",value:"file"}],mdxType:"Tabs"},Object(i.b)(l.a,{value:"numpy",mdxType:"TabItem"},Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\nfrom cfdata.tabular import TabularDataset\n\nx, y = TabularDataset.iris().xy\nm = cflearn.ml.CarefreePipeline().fit(x, y)\n# Predict logits\nm.predict(x)[cflearn.PREDICTIONS_KEY]\n# Evaluate performance\ncflearn.ml.evaluate(x, y, pipelines=m, metrics=["acc", "auc"])\n')),Object(i.b)("p",null,"This yields:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          |    0.973333    |    0.000000    |    0.973333    |    0.999067    |    0.000000    |    0.999067    |\n================================================================================================================================\n"))),Object(i.b)(l.a,{value:"file",mdxType:"TabItem"},Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," can also easily fit / predict / evaluate directly on files (",Object(i.b)("strong",{parentName:"p"},"file-in, file-out"),"). Suppose we have an ",Object(i.b)("inlineCode",{parentName:"p"},"xor.txt")," file with following contents:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-text"}),"0,0,0\n0,1,1\n1,0,1\n1,1,0\n")),Object(i.b)("p",null,"Then ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," can be utilized with only few lines of code:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\n\ny_train = None\nx_train = x_valid = "xor.txt"\nread_config = dict(delim=",", has_column_names=False)\nm = cflearn.ml.CarefreePipeline(read_config=read_config)\nm.fit(x_train, y_train, x_valid)\n# `contains_labels` is set to True because we\'re evaluating on training set\ncflearn.ml.evaluate("xor.txt", contains_labels=True, pipelines=m, metrics=["acc", "auc"])\n')),Object(i.b)("div",{className:"admonition admonition-info alert alert--info"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"info")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("ul",{parentName:"div"},Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"delim")," refers to '",Object(i.b)("strong",{parentName:"li"},"delimiter"),"', and ",Object(i.b)("inlineCode",{parentName:"li"},"has_column_names")," refers to whether the file has column names (or, header) or not."),Object(i.b)("li",{parentName:"ul"},"Please refer to ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"https://github.com/carefree0910/carefree-data/blob/dev/README.md"}),"carefree-data")," if you're interested in more details.")))),Object(i.b)("p",null,"This yields:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          |    1.000000    |    0.000000    |    1.000000    |    1.000000    |    0.000000    |    1.000000    |\n================================================================================================================================\n")),Object(i.b)("p",null,"When we fit from files, we can predict on either files or lists:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'key = cflearn.PREDICTIONS_KEY\nprint(m.predict([[0, 0]])[key].argmax(1))   # [0]\nprint(m.predict([[0, 1]])[key].argmax(1))   # [1]\nprint(m.predict("xor.txt", contains_labels=True)[key].argmax(1))  # [0 1 1 0]\n')))),Object(i.b)("h3",{id:"computer-vision-\ufe0f"},"Computer Vision \ud83d\uddbc\ufe0f"),Object(i.b)(c.a,{defaultValue:"preset",values:[{label:"Preset (torchvision) Dataset",value:"preset"},{label:"Custom Image Folder Dataset",value:"custom"}],mdxType:"Tabs"},Object(i.b)(l.a,{value:"preset",mdxType:"TabItem"},Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'# MNIST classification task with resnet18\n\nimport os\nimport cflearn\n\ntrain_loader, valid_loader = cflearn.cv.get_mnist(transform="to_tensor")\n\nm = cflearn.cv.CarefreePipeline(\n    "clf",\n    {\n        "in_channels": 1,\n        "num_classes": 10,\n        "latent_dim": 512,\n        "encoder1d": "backbone",\n        "encoder1d_configs": {"name": "resnet18"},\n    },\n    loss_name="cross_entropy",\n    metric_names="acc",\n    fixed_epoch=1,  # For demo purpose\n)\nm.fit(train_loader, valid_loader)\n'))),Object(i.b)(l.a,{value:"custom",mdxType:"TabItem"},Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"This is a WIP section :D")))),Object(i.b)("h2",{id:"serializing"},"Serializing"),Object(i.b)("h3",{id:"saving"},"Saving"),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," models can be saved easily, into a zip file (for both ml & cv tasks) !"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'m.save("model")  # a `model.zip` file will be created\n')),Object(i.b)("p",null,"In most cases, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," also supports a two-stage style serializing:"),Object(i.b)("ol",null,Object(i.b)("li",{parentName:"ol"},"A ",Object(i.b)("inlineCode",{parentName:"li"},"_logs")," folder (with timestamps as its subfolders) will be created after training.")),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-text"}),"--- _logs\n |-- 2021-08-08_16-00-24-175005\n  |-- checkpoints\n  |-- configs.json\n  |-- metrics.txt\n  ...\n |-- 2021-08-08_17-25-21-803661\n  |-- checkpoints\n  |-- configs.json\n  |-- metrics.txt\n  ...\n")),Object(i.b)("ol",{start:2},Object(i.b)("li",{parentName:"ol"},Object(i.b)("inlineCode",{parentName:"li"},"carefree-learn")," Pipelines could therefore 'pack' the corresponding (timestamp) folder into a zip file.")),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'# Notice that we should use the same `*Pipeline` as we use at training stage\nbase = cflearn.cv.CarefreePipeline\n# A `packed.zip` file will be created under `_logs/2021-08-08_17-25-21-803661`\nbase.pack("_logs/2021-08-08_17-25-21-803661")\n')),Object(i.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("ul",{parentName:"div"},Object(i.b)("li",{parentName:"ul"},"The ",Object(i.b)("strong",{parentName:"li"},"pack")," procedure could be done 'individually', which means there are no dependencies between the ",Object(i.b)("strong",{parentName:"li"},"pack")," procedure and the ",Object(i.b)("strong",{parentName:"li"},"training")," procedure."),Object(i.b)("li",{parentName:"ul"},"Machine Learning Pipeline may not always be able to do the same thing. To be exact, ",Object(i.b)("inlineCode",{parentName:"li"},"cflearn.ml.SimplePipeline")," supports the ",Object(i.b)("strong",{parentName:"li"},"pack")," procedure, but ",Object(i.b)("inlineCode",{parentName:"li"},"cflearn.ml.CarefreePipeline")," doesn't. This is because ",Object(i.b)("inlineCode",{parentName:"li"},"cflearn.ml.CarefreePipeline")," contains some extra data structure (the ",Object(i.b)("inlineCode",{parentName:"li"},"carefree-data")," stuffs) which is not recorded in the ",Object(i.b)("inlineCode",{parentName:"li"},"_logs")," folder. In this case, we should use the ",Object(i.b)("inlineCode",{parentName:"li"},"m.save")," API to save all the necessary information.")))),Object(i.b)("h3",{id:"loading"},"Loading"),Object(i.b)("p",null,"Of course, loading ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," models are easy too!"),Object(i.b)(c.a,{defaultValue:"ml",values:[{label:"Machine Learning \ud83d\udcc8",value:"ml"},{label:"Computer Vision \ud83d\uddbc\ufe0f",value:"cv"}],mdxType:"Tabs"},Object(i.b)(l.a,{value:"ml",mdxType:"TabItem"},Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'m = cflearn.ml.CarefreePipeline.load("/path/to/zip_file")\n'))),Object(i.b)(l.a,{value:"cv",mdxType:"TabItem"},Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'m = cflearn.cv.CarefreePipeline.load("/path/to/zip_file")\n')))),Object(i.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("ul",{parentName:"div"},Object(i.b)("li",{parentName:"ul"},"zip file from either ",Object(i.b)("inlineCode",{parentName:"li"},"save")," API or ",Object(i.b)("inlineCode",{parentName:"li"},"pack")," API can be loaded in this way."),Object(i.b)("li",{parentName:"ul"},"Please refer to the ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"../user-guides/production"}),"Production")," section for production usages.")))))}m.isMDXComponent=!0}}]);