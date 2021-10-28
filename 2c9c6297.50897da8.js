(window.webpackJsonp=window.webpackJsonp||[]).push([[5],{74:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return c})),a.d(t,"metadata",(function(){return o})),a.d(t,"rightToc",(function(){return b})),a.d(t,"default",(function(){return s}));var n=a(3),r=a(7),i=(a(0),a(97)),c={id:"design-principles",title:"Design Principles"},o={unversionedId:"design-principles",id:"design-principles",isDocsHomePage:!1,title:"Design Principles",description:"carefree-learn was designed to support most commonly used methods with carefree APIs. Moreover, carefree-learn was also designed with interface which is general enough, so that more sophisticated functionality can also be easily integrated in the future. This brings a tension in how to create abstractions in code, which is a challenge for us:",source:"@site/docs/design-principles.md",slug:"/design-principles",permalink:"/carefree-learn-doc/docs/design-principles",version:"current",lastUpdatedAt:1635384026,sidebar:"docs",previous:{title:"Introduction",permalink:"/carefree-learn-doc/docs/"},next:{title:"Optimizations",permalink:"/carefree-learn-doc/docs/optimizations"}},b=[{value:"Common Blocks",id:"common-blocks",children:[]},{value:"Configurations",id:"configurations",children:[]},{value:"Register Mechanism",id:"register-mechanism",children:[]},{value:"Model",id:"model",children:[]},{value:"Trainer",id:"trainer",children:[]},{value:"Pipeline",id:"pipeline",children:[]}],l={rightToc:b};function s(e){var t=e.components,a=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(n.a)({},l,a,{components:t,mdxType:"MDXLayout"}),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," was designed to support most commonly used methods with ",Object(i.b)("em",{parentName:"p"},"carefree")," APIs. Moreover, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," was also designed with interface which is general enough, so that more sophisticated functionality can also be easily integrated in the future. This brings a tension in how to create abstractions in code, which is a challenge for us:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"On the one hand, it requires a reasonably high-level abstraction so that users can easily work around with it in a standard way, without having to worry too much about the details."),Object(i.b)("li",{parentName:"ul"},"On the other hand, it also needs to have a very thin abstraction to allow users to do (many) other things in new ways. Breaking existing abstractions and replacing them with new ones should be fairly easy.")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", there are five main design principles that address this tension together:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"Divide ",Object(i.b)("inlineCode",{parentName:"li"},"carefree-learn")," into three parts: ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"#model"}),Object(i.b)("inlineCode",{parentName:"a"},"Model")),", ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"#trainer"}),Object(i.b)("inlineCode",{parentName:"a"},"Trainer"))," and ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"#pipeline"}),Object(i.b)("inlineCode",{parentName:"a"},"Pipeline")),"."),Object(i.b)("li",{parentName:"ul"},"Build some ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"#common-blocks"}),Object(i.b)("inlineCode",{parentName:"a"},"Common Blocks"))," which shall be leveraged across different ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"#model"}),Object(i.b)("inlineCode",{parentName:"a"},"Model")),"s."),Object(i.b)("li",{parentName:"ul"},"Manage models / blocks with ",Object(i.b)("inlineCode",{parentName:"li"},"register")," mechanism, so they can be accessed via their names (see ",Object(i.b)("a",Object(n.a)({parentName:"li"},{href:"#register-mechanism"}),"Register Mechanism"),").")),Object(i.b)("p",null,"We will introduce the details in the following subsections."),Object(i.b)("h2",{id:"common-blocks"},"Common Blocks"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/dev/cflearn/modules/blocks.py"}),"blocks.py"),".")),Object(i.b)("p",null,Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," implements many basic blocks which can directly form famous models, such as ",Object(i.b)("strong",{parentName:"p"},"VAE"),", ",Object(i.b)("strong",{parentName:"p"},"AdaIN"),", ",Object(i.b)("strong",{parentName:"p"},"CycleGAN"),", ",Object(i.b)("strong",{parentName:"p"},"BERT"),", ",Object(i.b)("strong",{parentName:"p"},"ViT"),", ",Object(i.b)("strong",{parentName:"p"},"FNet"),", ",Object(i.b)("strong",{parentName:"p"},"StyleGAN"),", ",Object(i.b)("strong",{parentName:"p"},"U^2 Net")," etc. The best of ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," is that, it not only reproduces the official implementations, but also reuses everything it could. For example:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"The ",Object(i.b)("inlineCode",{parentName:"li"},"Decoder")," used in ",Object(i.b)("inlineCode",{parentName:"li"},"VAE")," and ",Object(i.b)("inlineCode",{parentName:"li"},"CycleGAN")," is the same (with different args / kwargs)."),Object(i.b)("li",{parentName:"ul"},"The ",Object(i.b)("inlineCode",{parentName:"li"},"Transformer")," used in ",Object(i.b)("inlineCode",{parentName:"li"},"BERT")," and ",Object(i.b)("inlineCode",{parentName:"li"},"Vit")," is the same (with different args / kwargs)."),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"Transformer")," and ",Object(i.b)("inlineCode",{parentName:"li"},"FNet")," shares most of the codes, except that ",Object(i.b)("inlineCode",{parentName:"li"},"Transformer")," uses ",Object(i.b)("inlineCode",{parentName:"li"},"Attention")," but ",Object(i.b)("inlineCode",{parentName:"li"},"FNet")," uses fourier transform."),Object(i.b)("li",{parentName:"ul"},"And much more...")),Object(i.b)("h2",{id:"configurations"},"Configurations"),Object(i.b)("p",null,"In general, there are three kinds of configurations in ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),":"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"Model configurations."),Object(i.b)("li",{parentName:"ul"},"Trainer configurations."),Object(i.b)("li",{parentName:"ul"},"Pipeline configurations, which is basically constructed by the above two configurations.")),Object(i.b)("p",null,"See ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"getting-started/configurations#specify-configurations"}),"specifying configurations")," for more details."),Object(i.b)("h2",{id:"register-mechanism"},"Register Mechanism"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/misc/toolkit.py#L383"}),Object(i.b)("inlineCode",{parentName:"a"},"WithRegister")),".")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", it is likely to see ",Object(i.b)("inlineCode",{parentName:"p"},"@xxx.register(...)")," all around. This is very useful when we want to provide many useful defaults for users."),Object(i.b)("p",null,"Here's a code snippet that well demonstrates how to use ",Object(i.b)("inlineCode",{parentName:"p"},"register")," mechanism:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'from cflearn.misc.toolkit import WithRegister\n\nfoo = {}\n\nclass FooBase(WithRegister):\n    d = foo\n\n@FooBase.register("bar")\nclass Bar(FooBase):\n    def __init__(self, name="foobar"):\n        self.name = name\n\nprint(foo["bar"]().name)                             # foobar\nprint(FooBase.get("bar")().name)                     # foobar\nprint(FooBase.make("bar", {"name": "barfoo"}).name)  # barfoo\n')),Object(i.b)("h2",{id:"model"},"Model"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/protocol.py#L109"}),Object(i.b)("inlineCode",{parentName:"a"},"ModelProtocol")),".")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", a ",Object(i.b)("inlineCode",{parentName:"p"},"Model")," should implement the core algorithms. It's basically an ",Object(i.b)("inlineCode",{parentName:"p"},"nn.Module"),", with some extra useful functions:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),"class ModelProtocol(nn.Module, WithRegister, metaclass=ABCMeta):\n    d = model_dict\n\n    ...\n\n    @property\n    def device(self) -> torch.device:\n        return list(self.parameters())[0].device\n\n    def onnx_forward(self, batch: tensor_dict_type) -> Any:\n        return self.forward(0, batch)\n\n    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:\n        self.forward(batch_idx, batch)\n")),Object(i.b)("p",null,"As shown above, there are two special ",Object(i.b)("inlineCode",{parentName:"p"},"forward")," methods defined in a ",Object(i.b)("inlineCode",{parentName:"p"},"Model"),", which allows us to customize ",Object(i.b)("inlineCode",{parentName:"p"},"onnx")," export procedure and ",Object(i.b)("inlineCode",{parentName:"p"},"summary")," procedure respectively."),Object(i.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"See ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"/docs/developer-guides/general-customization#customize-models"}),"Customize Models")," section for more details."))),Object(i.b)("h2",{id:"trainer"},"Trainer"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/trainer.py#L226"}),Object(i.b)("inlineCode",{parentName:"a"},"Trainer")),".")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", a ",Object(i.b)("inlineCode",{parentName:"p"},"Trainer")," should implement the training loop, which includes updating trainable parameters with an optimizer (or, some optimizers), verbosing metrics / losses, checkpointing, early stopping, logging, etc."),Object(i.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"Although we can construct a ",Object(i.b)("inlineCode",{parentName:"p"},"Trainer")," from scratch, ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," provides ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/misc/internal_/trainer.py#L19"}),Object(i.b)("inlineCode",{parentName:"a"},"make_trainer"))," function, which contains many useful default ",Object(i.b)("inlineCode",{parentName:"p"},"Trainer")," values."))),Object(i.b)("h2",{id:"pipeline"},"Pipeline"),Object(i.b)("blockquote",null,Object(i.b)("p",{parentName:"blockquote"},"Source code: ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/pipeline.py#L90"}),Object(i.b)("inlineCode",{parentName:"a"},"DLPipeline")),".")),Object(i.b)("p",null,"In ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn"),", a ",Object(i.b)("inlineCode",{parentName:"p"},"Pipeline")," should implement the high-level parts (e.g. ",Object(i.b)("inlineCode",{parentName:"p"},"fit"),", ",Object(i.b)("inlineCode",{parentName:"p"},"predict"),", ",Object(i.b)("inlineCode",{parentName:"p"},"save"),", ",Object(i.b)("inlineCode",{parentName:"p"},"load"),", etc.), and will be the (internal) user interface. It's basically a 'wrapper' which can use a ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"#trainer"}),Object(i.b)("inlineCode",{parentName:"a"},"Trainer"))," to train a ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"#model"}),Object(i.b)("inlineCode",{parentName:"a"},"Model"))," properly, and can serialize the necessary information to disk."),Object(i.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"Although ",Object(i.b)("inlineCode",{parentName:"p"},"carefree-learn")," focuses on Deep Learning tasks, the most general abstraction (",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/2c745bb1e998e74bbbc1c308a5716608ef1b137f/cflearn/pipeline.py#L57"}),Object(i.b)("inlineCode",{parentName:"a"},"PipelineProtocol")),") can actually utilize traditional Machine Learning models, such as ",Object(i.b)("inlineCode",{parentName:"p"},"LinearRegression")," from ",Object(i.b)("inlineCode",{parentName:"p"},"scikit-learn"),"."))),Object(i.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(i.b)("h5",{parentName:"div"},Object(i.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(i.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(i.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(i.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(i.b)("p",{parentName:"div"},"See ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"/docs/getting-started/configurations#specify-configurations"}),"Specify Configurations")," section for more details."))))}s.isMDXComponent=!0},97:function(e,t,a){"use strict";a.d(t,"a",(function(){return p})),a.d(t,"b",(function(){return u}));var n=a(0),r=a.n(n);function i(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function c(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function o(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?c(Object(a),!0).forEach((function(t){i(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):c(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function b(e,t){if(null==e)return{};var a,n,r=function(e,t){if(null==e)return{};var a,n,r={},i=Object.keys(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||(r[a]=e[a]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(r[a]=e[a])}return r}var l=r.a.createContext({}),s=function(e){var t=r.a.useContext(l),a=t;return e&&(a="function"==typeof e?e(t):o(o({},t),e)),a},p=function(e){var t=s(e.components);return r.a.createElement(l.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},d=r.a.forwardRef((function(e,t){var a=e.components,n=e.mdxType,i=e.originalType,c=e.parentName,l=b(e,["components","mdxType","originalType","parentName"]),p=s(a),d=n,u=p["".concat(c,".").concat(d)]||p[d]||m[d]||i;return a?r.a.createElement(u,o(o({ref:t},l),{},{components:a})):r.a.createElement(u,o({ref:t},l))}));function u(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var i=a.length,c=new Array(i);c[0]=d;var o={};for(var b in t)hasOwnProperty.call(t,b)&&(o[b]=t[b]);o.originalType=e,o.mdxType="string"==typeof e?e:n,c[1]=o;for(var l=2;l<i;l++)c[l]=a[l];return r.a.createElement.apply(null,c)}return r.a.createElement.apply(null,a)}d.displayName="MDXCreateElement"}}]);