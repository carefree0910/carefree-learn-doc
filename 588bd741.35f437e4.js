(window.webpackJsonp=window.webpackJsonp||[]).push([[14],{105:function(e,t,n){"use strict";n.d(t,"a",(function(){return b})),n.d(t,"b",(function(){return m}));var a=n(0),r=n.n(a);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function c(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?c(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):c(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=r.a.createContext({}),u=function(e){var t=r.a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},b=function(e){var t=u(e.components);return r.a.createElement(s.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},f=r.a.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,c=e.parentName,s=o(e,["components","mdxType","originalType","parentName"]),b=u(n),f=a,m=b["".concat(c,".").concat(f)]||b[f]||p[f]||i;return n?r.a.createElement(m,l(l({ref:t},s),{},{components:n})):r.a.createElement(m,l({ref:t},s))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,c=new Array(i);c[0]=f;var l={};for(var o in t)hasOwnProperty.call(t,o)&&(l[o]=t[o]);l.originalType=e,l.mdxType="string"==typeof e?e:a,c[1]=l;for(var s=2;s<i;s++)c[s]=n[s];return r.a.createElement.apply(null,c)}return r.a.createElement.apply(null,n)}f.displayName="MDXCreateElement"},106:function(e,t,n){"use strict";function a(e){var t,n,r="";if("string"==typeof e||"number"==typeof e)r+=e;else if("object"==typeof e)if(Array.isArray(e))for(t=0;t<e.length;t++)e[t]&&(n=a(e[t]))&&(r&&(r+=" "),r+=n);else for(t in e)e[t]&&(r&&(r+=" "),r+=t);return r}t.a=function(){for(var e,t,n=0,r="";n<arguments.length;)(e=arguments[n++])&&(t=a(e))&&(r&&(r+=" "),r+=t);return r}},111:function(e,t,n){"use strict";var a=n(0),r=n(112);t.a=function(){var e=Object(a.useContext)(r.a);if(null==e)throw new Error("`useUserPreferencesContext` is used outside of `Layout` Component.");return e}},112:function(e,t,n){"use strict";var a=n(0),r=Object(a.createContext)(void 0);t.a=r},114:function(e,t,n){"use strict";var a=n(0),r=n.n(a),i=n(111),c=n(106),l=n(52),o=n.n(l),s=37,u=39;t.a=function(e){var t=e.lazy,n=e.block,l=e.children,b=e.defaultValue,p=e.values,f=e.groupId,m=e.className,d=Object(i.a)(),O=d.tabGroupChoices,j=d.setTabGroupChoices,g=Object(a.useState)(b),y=g[0],v=g[1];if(null!=f){var h=O[f];null!=h&&h!==y&&p.some((function(e){return e.value===h}))&&v(h)}var N=function(e){v(e),null!=f&&j(f,e)},x=[];return r.a.createElement("div",null,r.a.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:Object(c.a)("tabs",{"tabs--block":n},m)},p.map((function(e){var t=e.value,n=e.label;return r.a.createElement("li",{role:"tab",tabIndex:0,"aria-selected":y===t,className:Object(c.a)("tabs__item",o.a.tabItem,{"tabs__item--active":y===t}),key:t,ref:function(e){return x.push(e)},onKeyDown:function(e){!function(e,t,n){switch(n.keyCode){case u:!function(e,t){var n=e.indexOf(t)+1;e[n]?e[n].focus():e[0].focus()}(e,t);break;case s:!function(e,t){var n=e.indexOf(t)-1;e[n]?e[n].focus():e[e.length-1].focus()}(e,t)}}(x,e.target,e)},onFocus:function(){return N(t)},onClick:function(){N(t)}},n)}))),t?Object(a.cloneElement)(l.filter((function(e){return e.props.value===y}))[0],{className:"margin-vert--md"}):r.a.createElement("div",{className:"margin-vert--md"},l.map((function(e,t){return Object(a.cloneElement)(e,{key:t,hidden:e.props.value!==y})}))))}},115:function(e,t,n){"use strict";var a=n(3),r=n(0),i=n.n(r);t.a=function(e){var t=e.children,n=e.hidden,r=e.className;return i.a.createElement("div",Object(a.a)({role:"tabpanel"},{hidden:n,className:r}),t)}},82:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return o})),n.d(t,"metadata",(function(){return s})),n.d(t,"rightToc",(function(){return u})),n.d(t,"default",(function(){return p}));var a=n(3),r=n(7),i=(n(0),n(105)),c=n(114),l=n(115),o={id:"quick-start",title:"Quick Start"},s={unversionedId:"getting-started/quick-start",id:"getting-started/quick-start",isDocsHomePage:!1,title:"Quick Start",description:"In carefree-learn, it's both easy to train and serialize a model:",source:"@site/docs/getting-started/quick-start.md",slug:"/getting-started/quick-start",permalink:"/carefree-learn-doc/docs/getting-started/quick-start",version:"current",lastUpdatedAt:1605876656,sidebar:"docs",previous:{title:"Installation",permalink:"/carefree-learn-doc/docs/getting-started/installation"},next:{title:"Configurations",permalink:"/carefree-learn-doc/docs/getting-started/configurations"}},u=[{value:"Training",id:"training",children:[]},{value:"Serializing",id:"serializing",children:[]}],b={rightToc:u};function p(e){var t=e.components,n=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(a.a)({},b,n,{components:t,mdxType:"MDXLayout"}),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", it's both easy to train and serialize a model:"),Object(i.b)("h2",{id:"training"},"Training"),Object(i.b)(c.a,{defaultValue:"numpy",values:[{label:"With NumPy",value:"numpy"},{label:"With File",value:"file"}],mdxType:"Tabs"},Object(i.b)(l.a,{value:"numpy",mdxType:"TabItem"},Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"import cflearn\nfrom cfdata.tabular import TabularDataset\n\nx, y = TabularDataset.iris().xy\nm = cflearn.make().fit(x, y)\n# Make label predictions\nm.predict(x)\n# Make probability predictions\nm.predict_prob(x)\n# Evaluate performance\ncflearn.evaluate(x, y, pipelines=m)\n")),Object(i.b)("p",null,"Then you will see something like this:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          |    0.946667    |    0.000000    |    0.946667    |    0.993200    |    0.000000    |    0.993200    |\n================================================================================================================================\n"))),Object(i.b)(l.a,{value:"file",mdxType:"TabItem"},Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," can also easily fit / predict / evaluate directly on files (",Object(i.b)("strong",{parentName:"p"},"file-in, file-out"),"). Suppose we have an ",Object(i.b)("inlineCode",{parentName:"p"},"xor.txt")," file with following contents:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-text"}),"0,0,0\n0,1,1\n1,0,1\n1,1,0\n")),Object(i.b)("p",null,"Then ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," can be utilized with only few lines of code:"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},Object(i.b)("inlineCode",{parentName:"p"},"delim")," refers to 'delimiter', and ",Object(i.b)("inlineCode",{parentName:"p"},"has_column_names")," refers to whether the file has column names (or, header) or not."),Object(i.b)("p",{parentName:"blockquote"},"Please refer to ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-data/blob/dev/README.md"}),"carefree-data")," if you're interested in more details.")),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'import cflearn\nm = cflearn.make(delim=",", has_column_names=False).fit("xor.txt", x_cv="xor.txt")\n# `contains_labels` is set to True because we\'re evaluating on training set\ncflearn.evaluate("xor.txt", pipelines=m, contains_labels=True)\n')),Object(i.b)("p",null,"After which you will see something like this:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-text"}),"================================================================================================================================\n|        metrics         |                       acc                        |                       auc                        |\n--------------------------------------------------------------------------------------------------------------------------------\n|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |\n--------------------------------------------------------------------------------------------------------------------------------\n|          fcnn          |    1.000000    |    0.000000    |    1.000000    |    1.000000    |    0.000000    |    1.000000    |\n================================================================================================================================\n")),Object(i.b)("p",null,"When we fit from files, we can predict on either files or lists:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'print(m.predict([[0, 0]]))   # [[0]]\nprint(m.predict([[0, 1]]))   # [[1]]\nprint(m.predict("xor.txt", contains_labels=True))  # [ [0] [1] [1] [0] ]\n')))),Object(i.b)("h2",{id:"serializing"},"Serializing"),Object(i.b)("p",null,"It is also worth mentioning that ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," models can be saved easily, into a zip file!"),Object(i.b)("p",null,"For example, a ",Object(i.b)("inlineCode",{parentName:"p"},"cflearn^_^fcnn.zip")," file will be created with one line of code:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"cflearn.save(m)\n")),Object(i.b)("p",null,"Of course, loading ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," models are easy too!"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"m = cflearn.load()\nprint(m)  # {'fcnn': FCNN()}\n")))}p.isMDXComponent=!0}}]);