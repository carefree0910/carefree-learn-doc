(window.webpackJsonp=window.webpackJsonp||[]).push([[32],{108:function(e,n,t){"use strict";t.d(n,"a",(function(){return u})),t.d(n,"b",(function(){return f}));var a=t(0),r=t.n(a);function i(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function c(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?c(Object(t),!0).forEach((function(n){i(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):c(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function p(e,n){if(null==e)return{};var t,a,r=function(e,n){if(null==e)return{};var t,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var l=r.a.createContext({}),s=function(e){var n=r.a.useContext(l),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},u=function(e){var n=s(e.components);return r.a.createElement(l.Provider,{value:n},e.children)},b={inlineCode:"code",wrapper:function(e){var n=e.children;return r.a.createElement(r.a.Fragment,{},n)}},d=r.a.forwardRef((function(e,n){var t=e.components,a=e.mdxType,i=e.originalType,c=e.parentName,l=p(e,["components","mdxType","originalType","parentName"]),u=s(t),d=a,f=u["".concat(c,".").concat(d)]||u[d]||b[d]||i;return t?r.a.createElement(f,o(o({ref:n},l),{},{components:t})):r.a.createElement(f,o({ref:n},l))}));function f(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var i=t.length,c=new Array(i);c[0]=d;var o={};for(var p in n)hasOwnProperty.call(n,p)&&(o[p]=n[p]);o.originalType=e,o.mdxType="string"==typeof e?e:a,c[1]=o;for(var l=2;l<i;l++)c[l]=t[l];return r.a.createElement.apply(null,c)}return r.a.createElement.apply(null,t)}d.displayName="MDXCreateElement"},157:function(e,n,t){"use strict";t.r(n),n.default=t.p+"assets/images/pack-41dffc5a04b9a5b2aa919f124e2b5d8f.png"},99:function(e,n,t){"use strict";t.r(n),t.d(n,"frontMatter",(function(){return c})),t.d(n,"metadata",(function(){return o})),t.d(n,"rightToc",(function(){return p})),t.d(n,"default",(function(){return s}));var a=t(3),r=t(7),i=(t(0),t(108)),c={id:"production",title:"Production"},o={unversionedId:"user-guides/production",id:"user-guides/production",isDocsHomePage:!1,title:"Production",description:"carefree-learn supports onnx export, but we need much more than one single model (predictor) in production environment:",source:"@site/docs/user-guides/production.md",slug:"/user-guides/production",permalink:"/carefree-learn-doc/docs/user-guides/production",version:"current",lastUpdatedAt:1606063479,sidebar:"docs",previous:{title:"Distributed",permalink:"/carefree-learn-doc/docs/user-guides/distributed"},next:{title:"Examples",permalink:"/carefree-learn-doc/docs/user-guides/examples"}},p=[{value:"Why Pack?",id:"why-pack",children:[]},{value:"AutoML in Production",id:"automl-in-production",children:[]}],l={rightToc:p};function s(e){var n=e.components,c=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(a.a)({},l,c,{components:n,mdxType:"MDXLayout"}),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," supports ",Object(i.b)("inlineCode",{parentName:"p"},"onnx")," export, but we need much more than one single model (",Object(i.b)("inlineCode",{parentName:"p"},"predictor"),") in production environment:"),Object(i.b)("p",null,Object(i.b)("img",{alt:"Pack",src:t(157).default})),Object(i.b)("p",null,"Fortunately, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," also supports exporting every part of this pipeline into a zip file with one line of code. Let's first train a simple model on ",Object(i.b)("inlineCode",{parentName:"p"},"iris")," dataset:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"import cflearn\nfrom cfdata.tabular import TabularDataset\n\nx, y = TabularDataset.iris().xy\nm = cflearn.make().fit(x, y)\n")),Object(i.b)("p",null,"After which we can pack everything up with ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Pack")," API:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'cflearn.Pack.pack(m, "pack")\n')),Object(i.b)("p",null,"This will generate a ",Object(i.b)("inlineCode",{parentName:"p"},"pack.zip")," in the working directory with following file structure:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-text"}),"|--- preprocessor\n   |-- ...\n|--- binary_config.json\n|--- m.onnx\n|--- output_names.json\n|--- output_probabilities.txt\n")),Object(i.b)("p",null,"We can make inference with this ",Object(i.b)("inlineCode",{parentName:"p"},"pack.zip")," on our production environments / machines easily:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\n\npredictor = cflearn.Pack.get_predictor("pack")\npredictions = predictor.predict(x)\n')),Object(i.b)("h2",{id:"why-pack"},"Why Pack?"),Object(i.b)("p",null,"You might notice that both ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.save")," and ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Pack.pack")," generate a zip file and both can be loaded for inference, so why should we introduce ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Pack"),"? The reason is that ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.save")," will save much more information than ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Pack"),", which is not an ideal behaviour in production. For instance, ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.save")," will save several copies of the original data, while ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn.Pack")," will simply save the core statistics used in pre-processors."),Object(i.b)("p",null,"In fact, if we execute ",Object(i.b)("inlineCode",{parentName:"p"},'cflearn.save(m, "saved")'),", which will generated a ",Object(i.b)("inlineCode",{parentName:"p"},"saved^_^fcnn.zip"),", instead of ",Object(i.b)("inlineCode",{parentName:"p"},'cflearn.Pack.pack(m, "pack")'),", we'll see that the file size of ",Object(i.b)("inlineCode",{parentName:"p"},"saved^_^fcnn.zip")," is 18k while the file size of ",Object(i.b)("inlineCode",{parentName:"p"},"pack.zip")," is only 10k. This difference will grow linearly to the dataset size, because the file size of ",Object(i.b)("inlineCode",{parentName:"p"},"pack.zip")," won't change as long as the model structure remains unchanged, while the file size of ",Object(i.b)("inlineCode",{parentName:"p"},"saved^_^fcnn.zip")," depends heavily on the dataset size."),Object(i.b)("h2",{id:"automl-in-production"},"AutoML in Production"),Object(i.b)("p",null,"As mentioned in the ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"/carefree-learn-doc/docs/user-guides/auto-ml"}),"AutoML")," section, it is possible to pack all the trained models into a single zip file:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'auto.pack("pack")\n')),Object(i.b)("p",null,"This will generate a ",Object(i.b)("inlineCode",{parentName:"p"},"pack.zip")," in the working directory with following file structure:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-text"}),"|--- __data__\n   |-- ...\n|--- fcnn\n   |-- ...\n|--- linear\n   |-- ...\n|--- ndt\n   |-- ...\n|--- nnb\n   |-- ...\n|--- tree_dnn\n   |-- ...\n|--- weights_mapping.json\n")),Object(i.b)("p",null,"We can make inference with this ",Object(i.b)("inlineCode",{parentName:"p"},"pack.zip")," on our production environments / machines easily:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\n\nunpacked = cflearn.Auto.unpack("pack")\npredictions = unpacked.pattern.predict(x)\n')))}s.isMDXComponent=!0}}]);