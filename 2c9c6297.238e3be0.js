(window.webpackJsonp=window.webpackJsonp||[]).push([[8],{109:function(e,t,n){"use strict";n.d(t,"a",(function(){return p})),n.d(t,"b",(function(){return u}));var a=n(0),r=n.n(a);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function c(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?c(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):c(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var b=r.a.createContext({}),s=function(e){var t=r.a.useContext(b),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},p=function(e){var t=s(e.components);return r.a.createElement(b.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},d=r.a.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,c=e.parentName,b=l(e,["components","mdxType","originalType","parentName"]),p=s(n),d=a,u=p["".concat(c,".").concat(d)]||p[d]||m[d]||i;return n?r.a.createElement(u,o(o({ref:t},b),{},{components:n})):r.a.createElement(u,o({ref:t},b))}));function u(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,c=new Array(i);c[0]=d;var o={};for(var l in t)hasOwnProperty.call(t,l)&&(o[l]=t[l]);o.originalType=e,o.mdxType="string"==typeof e?e:a,c[1]=o;for(var b=2;b<i;b++)c[b]=n[b];return r.a.createElement.apply(null,c)}return r.a.createElement.apply(null,n)}d.displayName="MDXCreateElement"},110:function(e,t,n){"use strict";function a(e){var t,n,r="";if("string"==typeof e||"number"==typeof e)r+=e;else if("object"==typeof e)if(Array.isArray(e))for(t=0;t<e.length;t++)e[t]&&(n=a(e[t]))&&(r&&(r+=" "),r+=n);else for(t in e)e[t]&&(r&&(r+=" "),r+=t);return r}t.a=function(){for(var e,t,n=0,r="";n<arguments.length;)(e=arguments[n++])&&(t=a(e))&&(r&&(r+=" "),r+=t);return r}},114:function(e,t,n){"use strict";var a=n(0),r=n(115);t.a=function(){var e=Object(a.useContext)(r.a);if(null==e)throw new Error("`useUserPreferencesContext` is used outside of `Layout` Component.");return e}},115:function(e,t,n){"use strict";var a=n(0),r=Object(a.createContext)(void 0);t.a=r},117:function(e,t,n){"use strict";var a=n(0),r=n.n(a),i=n(114),c=n(110),o=n(52),l=n.n(o),b=37,s=39;t.a=function(e){var t=e.lazy,n=e.block,o=e.children,p=e.defaultValue,m=e.values,d=e.groupId,u=e.className,f=Object(i.a)(),h=f.tabGroupChoices,O=f.setTabGroupChoices,j=Object(a.useState)(p),g=j[0],N=j[1];if(null!=d){var v=h[d];null!=v&&v!==g&&m.some((function(e){return e.value===v}))&&N(v)}var y=function(e){N(e),null!=d&&O(d,e)},w=[];return r.a.createElement("div",null,r.a.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:Object(c.a)("tabs",{"tabs--block":n},u)},m.map((function(e){var t=e.value,n=e.label;return r.a.createElement("li",{role:"tab",tabIndex:0,"aria-selected":g===t,className:Object(c.a)("tabs__item",l.a.tabItem,{"tabs__item--active":g===t}),key:t,ref:function(e){return w.push(e)},onKeyDown:function(e){!function(e,t,n){switch(n.keyCode){case s:!function(e,t){var n=e.indexOf(t)+1;e[n]?e[n].focus():e[0].focus()}(e,t);break;case b:!function(e,t){var n=e.indexOf(t)-1;e[n]?e[n].focus():e[e.length-1].focus()}(e,t)}}(w,e.target,e)},onFocus:function(){return y(t)},onClick:function(){y(t)}},n)}))),t?Object(a.cloneElement)(o.filter((function(e){return e.props.value===g}))[0],{className:"margin-vert--md"}):r.a.createElement("div",{className:"margin-vert--md"},o.map((function(e,t){return Object(a.cloneElement)(e,{key:t,hidden:e.props.value!==g})}))))}},118:function(e,t,n){"use strict";var a=n(3),r=n(0),i=n.n(r);t.a=function(e){var t=e.children,n=e.hidden,r=e.className;return i.a.createElement("div",Object(a.a)({role:"tabpanel"},{hidden:n,className:r}),t)}},77:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return c})),n.d(t,"metadata",(function(){return o})),n.d(t,"rightToc",(function(){return l})),n.d(t,"default",(function(){return s}));var a=n(3),r=n(7),i=(n(0),n(109)),c=(n(117),n(118),{id:"design-principles",title:"Design Principles"}),o={unversionedId:"design-principles",id:"design-principles",isDocsHomePage:!1,title:"Design Principles",description:"carefree-learn was designed to support most commonly used methods with carefree APIs. Moreover, carefree-learn was also designed with interface which is general enough, so that more sophisticated functionality can also be easily integrated in the future. This brings a tension in how to create abstractions in code, which is a challenge for us:",source:"@site/docs/design-principles.md",slug:"/design-principles",permalink:"/carefree-learn-doc/docs/design-principles",version:"current",lastUpdatedAt:1633856323,sidebar:"docs",previous:{title:"Introduction",permalink:"/carefree-learn-doc/docs/"},next:{title:"Optimizations",permalink:"/carefree-learn-doc/docs/optimizations"}},l=[{value:"Common Blocks",id:"common-blocks",children:[]},{value:"Configurations",id:"configurations",children:[]},{value:"Register Mechanism",id:"register-mechanism",children:[]},{value:"Model",id:"model",children:[]},{value:"Trainer",id:"trainer",children:[]},{value:"Pipeline",id:"pipeline",children:[]}],b={rightToc:l};function s(e){var t=e.components,n=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(a.a)({},b,n,{components:t,mdxType:"MDXLayout"}),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," was designed to support most commonly used methods with ",Object(i.b)("em",{parentName:"p"},"carefree")," APIs. Moreover, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," was also designed with interface which is general enough, so that more sophisticated functionality can also be easily integrated in the future. This brings a tension in how to create abstractions in code, which is a challenge for us:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"On the one hand, it requires a reasonably high-level abstraction so that users can easily work around with it in a standard way, without having to worry too much about the details."),Object(i.b)("li",{parentName:"ul"},"On the other hand, it also needs to have a very thin abstraction to allow users to do (many) other things in new ways. Breaking existing abstractions and replacing them with new ones should be fairly easy.")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", there are five main design principles that address this tension together:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"Divide ",Object(i.b)("inlineCode",{parentName:"li"},"carefree-learn")," into three parts: ",Object(i.b)("a",Object(a.a)({parentName:"li"},{href:"#model"}),Object(i.b)("inlineCode",{parentName:"a"},"Model")),", ",Object(i.b)("a",Object(a.a)({parentName:"li"},{href:"#trainer"}),Object(i.b)("inlineCode",{parentName:"a"},"Trainer"))," and ",Object(i.b)("a",Object(a.a)({parentName:"li"},{href:"#pipeline"}),Object(i.b)("inlineCode",{parentName:"a"},"Pipeline")),"."),Object(i.b)("li",{parentName:"ul"},"Build some ",Object(i.b)("a",Object(a.a)({parentName:"li"},{href:"#common-blocks"}),Object(i.b)("inlineCode",{parentName:"a"},"Common Blocks"))," which shall be leveraged across different ",Object(i.b)("a",Object(a.a)({parentName:"li"},{href:"#model"}),Object(i.b)("inlineCode",{parentName:"a"},"Model")),"s."),Object(i.b)("li",{parentName:"ul"},"Manage models / blocks with ",Object(i.b)("inlineCode",{parentName:"li"},"register")," mechanism, so they can be accessed via their names (see ",Object(i.b)("a",Object(a.a)({parentName:"li"},{href:"#registration"}),Object(i.b)("inlineCode",{parentName:"a"},"Registration")),").")),Object(i.b)("p",null,"We will introduce the details in the following subsections."),Object(i.b)("h2",{id:"common-blocks"},"Common Blocks"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/modules/blocks.py"}),"blocks.py"),".")),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," implements many basic blocks which can directly form famous models, such as ",Object(i.b)("strong",{parentName:"p"},"VAE"),", ",Object(i.b)("strong",{parentName:"p"},"AdaIN"),", ",Object(i.b)("strong",{parentName:"p"},"CycleGAN"),", ",Object(i.b)("strong",{parentName:"p"},"BERT"),", ",Object(i.b)("strong",{parentName:"p"},"ViT"),", ",Object(i.b)("strong",{parentName:"p"},"FNet"),", ",Object(i.b)("strong",{parentName:"p"},"StyleGAN"),", ",Object(i.b)("strong",{parentName:"p"},"U^2 Net")," etc. The best of ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," is that, it not only reproduces the official implementations, but also reuses everything it could. For example:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"The ",Object(i.b)("inlineCode",{parentName:"li"},"Decoder")," used in ",Object(i.b)("inlineCode",{parentName:"li"},"VAE")," and ",Object(i.b)("inlineCode",{parentName:"li"},"CycleGAN")," is the same (with different args / kwargs)."),Object(i.b)("li",{parentName:"ul"},"The ",Object(i.b)("inlineCode",{parentName:"li"},"Transformer")," used in ",Object(i.b)("inlineCode",{parentName:"li"},"BERT")," and ",Object(i.b)("inlineCode",{parentName:"li"},"Vit")," is the same (with different args / kwargs)."),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"Transformer")," and ",Object(i.b)("inlineCode",{parentName:"li"},"FNet")," shares most of the codes, except that ",Object(i.b)("inlineCode",{parentName:"li"},"Transformer")," uses ",Object(i.b)("inlineCode",{parentName:"li"},"Attention")," but ",Object(i.b)("inlineCode",{parentName:"li"},"FNet")," uses fourier transform."),Object(i.b)("li",{parentName:"ul"},"And much more...")),Object(i.b)("h2",{id:"configurations"},"Configurations"),Object(i.b)("p",null,"In general, there are three kinds of configurations in ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),":"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"Model configurations."),Object(i.b)("li",{parentName:"ul"},"Trainer configurations."),Object(i.b)("li",{parentName:"ul"},"Pipeline configurations, which is basically constructed by the above two configurations.")),Object(i.b)("p",null,"See ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"getting-started/configurations#specify-configurations"}),"specifying configurations")," for more details."),Object(i.b)("h2",{id:"register-mechanism"},"Register Mechanism"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/misc/toolkit.py#L383"}),Object(i.b)("inlineCode",{parentName:"a"},"WithRegister")),".")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", it is likely to see ",Object(i.b)("inlineCode",{parentName:"p"},"@xxx.register(...)")," all around. This is very useful when we want to provide many useful defaults for users."),Object(i.b)("p",null,"Here's a code snippet that well demonstrates how to use ",Object(i.b)("inlineCode",{parentName:"p"},"register")," mechanism:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'from cflearn.misc.toolkit import WithRegister\n\nfoo = {}\n\nclass FooBase(WithRegister):\n    d = foo\n\n@FooBase.register("bar")\nclass Bar(FooBase):\n    def __init__(self, name="foobar"):\n        self.name = name\n\nprint(foo["bar"]().name)                             # foobar\nprint(FooBase.get("bar")().name)                     # foobar\nprint(FooBase.make("bar", {"name": "barfoo"}).name)  # barfoo\n')),Object(i.b)("h2",{id:"model"},"Model"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/protocol.py#L109"}),Object(i.b)("inlineCode",{parentName:"a"},"ModelProtocol")),".")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", a ",Object(i.b)("inlineCode",{parentName:"p"},"Model")," should implement the core algorithms. It's basically an ",Object(i.b)("inlineCode",{parentName:"p"},"nn.Module"),", with some extra useful functions:"),Object(i.b)("pre",null,Object(i.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),"class ModelProtocol(nn.Module, WithRegister, metaclass=ABCMeta):\n    d = model_dict\n\n    ...\n\n    @property\n    def device(self) -> torch.device:\n        return list(self.parameters())[0].device\n\n    def onnx_forward(self, batch: tensor_dict_type) -> Any:\n        return self.forward(0, batch)\n\n    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:\n        self.forward(batch_idx, batch)\n")),Object(i.b)("p",null,"As shown above, there are two special ",Object(i.b)("inlineCode",{parentName:"p"},"forward")," methods defined in a ",Object(i.b)("inlineCode",{parentName:"p"},"Model"),", which allows us to customize ",Object(i.b)("inlineCode",{parentName:"p"},"onnx")," export procedure and ",Object(i.b)("inlineCode",{parentName:"p"},"summary")," procedure respectively."),Object(i.b)("h2",{id:"trainer"},"Trainer"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/trainer.py#L226"}),Object(i.b)("inlineCode",{parentName:"a"},"Trainer")),".")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", a ",Object(i.b)("inlineCode",{parentName:"p"},"Trainer")," should implement the training loop, which includes updating trainable parameters with an optimizer (or, some optimizers), verbosing metrics / losses, checkpointing, early stopping, logging, etc."),Object(i.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"Although we can construct a ",Object(i.b)("inlineCode",{parentName:"p"},"Trainer")," from scratch, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," provides ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/misc/internal_/trainer.py#L19"}),Object(i.b)("inlineCode",{parentName:"a"},"make_trainer"))," function, which contains many useful default ",Object(i.b)("inlineCode",{parentName:"p"},"Trainer")," values."))),Object(i.b)("h2",{id:"pipeline"},"Pipeline"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/pipeline.py#L90"}),Object(i.b)("inlineCode",{parentName:"a"},"DLPipeline")),".")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", a ",Object(i.b)("inlineCode",{parentName:"p"},"Pipeline")," should implement the high-level parts (e.g. ",Object(i.b)("inlineCode",{parentName:"p"},"fit"),", ",Object(i.b)("inlineCode",{parentName:"p"},"predict"),", ",Object(i.b)("inlineCode",{parentName:"p"},"save"),", ",Object(i.b)("inlineCode",{parentName:"p"},"load"),", etc.). It's basically a 'wrapper' which can use a ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"#trainer"}),Object(i.b)("inlineCode",{parentName:"a"},"Trainer"))," to train a ",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"#model"}),Object(i.b)("inlineCode",{parentName:"a"},"Model"))," properly, and can serialize the necessary information to disk."),Object(i.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(i.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"Although ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," focuses on Deep Learning tasks, the most general abstraction (",Object(i.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/pipeline.py#L57"}),Object(i.b)("inlineCode",{parentName:"a"},"PipelineProtocol")),") can actually utilize traditional Machine Learning models, such as ",Object(i.b)("inlineCode",{parentName:"p"},"LinearRegression")," from ",Object(i.b)("inlineCode",{parentName:"p"},"scikit-learn"),"."))))}s.isMDXComponent=!0}}]);