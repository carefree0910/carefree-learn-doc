(window.webpackJsonp=window.webpackJsonp||[]).push([[17],{88:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return o})),n.d(t,"metadata",(function(){return c})),n.d(t,"rightToc",(function(){return l})),n.d(t,"default",(function(){return s}));var a=n(3),i=n(7),r=(n(0),n(96)),o={id:"general-customization",title:"General"},c={unversionedId:"developer-guides/general-customization",id:"developer-guides/general-customization",isDocsHomePage:!1,title:"General",description:"In general, in order to solve our own tasks with our own models in carefree-learn, we need to concern:",source:"@site/docs/developer-guides/general.md",slug:"/developer-guides/general-customization",permalink:"/carefree-learn-doc/docs/developer-guides/general-customization",version:"current",lastUpdatedAt:1635037628,sidebar:"docs",previous:{title:"Machine Learning \ud83d\udcc8",permalink:"/carefree-learn-doc/docs/user-guides/machine-learning"},next:{title:"Computer Vision \ud83d\uddbc\ufe0f",permalink:"/carefree-learn-doc/docs/developer-guides/computer-vision-customization"}},l=[{value:"Customize Models",id:"customize-models",children:[{value:"<code>forward</code>",id:"forward",children:[]},{value:"<code>_init_with_trainer</code>",id:"_init_with_trainer",children:[]},{value:"Register &amp; Apply",id:"register--apply",children:[]}]},{value:"Customize Training Loop",id:"customize-training-loop",children:[]}],b={rightToc:l};function s(e){var t=e.components,n=Object(i.a)(e,["components"]);return Object(r.b)("wrapper",Object(a.a)({},b,n,{components:t,mdxType:"MDXLayout"}),Object(r.b)("p",null,"In general, in order to solve our own tasks with our own models in ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn"),", we need to concern:"),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},"How to define a new model & How to use it for training."),Object(r.b)("li",{parentName:"ul"},"How to customize pre-processings of the dataset."),Object(r.b)("li",{parentName:"ul"},"How to control some fine-grained behaviours of the training loop.")),Object(r.b)("p",null,"In this section, we will focus on the general customizations."),Object(r.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(r.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("ul",{parentName:"div"},Object(r.b)("li",{parentName:"ul"},"See ",Object(r.b)("a",Object(a.a)({parentName:"li"},{href:"computer-vision-customization"}),"here")," for customizations of Computer Vision \ud83d\uddbc\ufe0f."),Object(r.b)("li",{parentName:"ul"},"See ",Object(r.b)("a",Object(a.a)({parentName:"li"},{href:"machine-learning-customization"}),"here")," for customizations of Machine Learning \ud83d\udcc8.")))),Object(r.b)("h2",{id:"customize-models"},"Customize Models"),Object(r.b)("p",null,"In ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn"),", a ",Object(r.b)("inlineCode",{parentName:"p"},"Model")," should implement the core algorithms. It's basically an ",Object(r.b)("inlineCode",{parentName:"p"},"nn.Module"),", with some extra useful functions:"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'class ModelProtocol(nn.Module, WithRegister, metaclass=ABCMeta):\n    d = model_dict\n\n    @property\n    def device(self) -> torch.device:\n        return list(self.parameters())[0].device\n\n    def onnx_forward(self, batch: tensor_dict_type) -> Any:\n        return self.forward(0, batch)\n\n    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:\n        self.forward(batch_idx, batch)\n    \n    def _init_with_trainer(self, trainer: Any) -> None:\n        pass\n\n    @abstractmethod\n    def forward(\n        self,\n        batch_idx: int,\n        batch: tensor_dict_type,\n        state: Optional["TrainerState"] = None,\n        **kwargs: Any,\n    ) -> tensor_dict_type:\n        pass\n')),Object(r.b)("p",null,"As shown above, there are two special ",Object(r.b)("inlineCode",{parentName:"p"},"forward")," methods defined in a ",Object(r.b)("inlineCode",{parentName:"p"},"Model"),", which allows us to customize ",Object(r.b)("inlineCode",{parentName:"p"},"onnx")," export procedure and ",Object(r.b)("inlineCode",{parentName:"p"},"summary")," procedure respectively."),Object(r.b)("p",null,"If we want to define our own models, we will need to override the ",Object(r.b)("inlineCode",{parentName:"p"},"forward")," method (required) and the ",Object(r.b)("inlineCode",{parentName:"p"},"_init_with_trainer")," method (optional)."),Object(r.b)("h3",{id:"forward"},Object(r.b)("inlineCode",{parentName:"h3"},"forward")),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'def forward(\n    self,\n    batch_idx: int,\n    batch: tensor_dict_type,\n    state: Optional["TrainerState"] = None,\n    **kwargs: Any,\n) -> tensor_dict_type:\n    pass\n')),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},Object(r.b)("strong",{parentName:"li"},Object(r.b)("inlineCode",{parentName:"strong"},"batch_idx")),Object(r.b)("ul",{parentName:"li"},Object(r.b)("li",{parentName:"ul"},"Indicates the batch index of current batch."))),Object(r.b)("li",{parentName:"ul"},Object(r.b)("strong",{parentName:"li"},Object(r.b)("inlineCode",{parentName:"strong"},"batch")),Object(r.b)("ul",{parentName:"li"},Object(r.b)("li",{parentName:"ul"},Object(r.b)("p",{parentName:"li"},"Input batch. It will be a dictionary (",Object(r.b)("inlineCode",{parentName:"p"},"Dict[str, torch.Tensor]"),") returned by ",Object(r.b)("inlineCode",{parentName:"p"},"DataLoader"),".")),Object(r.b)("li",{parentName:"ul"},Object(r.b)("p",{parentName:"li"},"In general, it will:"),Object(r.b)("ul",{parentName:"li"},Object(r.b)("li",{parentName:"ul"},"always contain an ",Object(r.b)("inlineCode",{parentName:"li"},'"input"')," key, which represents the input data."),Object(r.b)("li",{parentName:"ul"},"usually contain a ",Object(r.b)("inlineCode",{parentName:"li"},'"labels"')," key, which represents the target labels.")),Object(r.b)("p",{parentName:"li"},"Other constants could be found ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/carefree0910/carefree-learn/blob/99c946ffa1df2b821161d52aae19f67e91abf46e/cflearn/constants.py"}),"here"),".")))),Object(r.b)("li",{parentName:"ul"},Object(r.b)("strong",{parentName:"li"},Object(r.b)("inlineCode",{parentName:"strong"},"state"))," ","[default = ",Object(r.b)("inlineCode",{parentName:"li"},"None"),"]",Object(r.b)("ul",{parentName:"li"},Object(r.b)("li",{parentName:"ul"},"The ",Object(r.b)("a",Object(a.a)({parentName:"li"},{href:"/docs/getting-started/configurations#trainerstate"}),Object(r.b)("inlineCode",{parentName:"a"},"TrainerState"))," instance."))),Object(r.b)("li",{parentName:"ul"},Object(r.b)("strong",{parentName:"li"},Object(r.b)("inlineCode",{parentName:"strong"},"kwargs")),Object(r.b)("ul",{parentName:"li"},Object(r.b)("li",{parentName:"ul"},"Other keyword arguments.")))),Object(r.b)("h3",{id:"_init_with_trainer"},Object(r.b)("inlineCode",{parentName:"h3"},"_init_with_trainer")),Object(r.b)("p",null,"This is an optional method, which is useful when we need to initialize our models with the prepared ",Object(r.b)("inlineCode",{parentName:"p"},"Trainer")," instance."),Object(r.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(r.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"Since the prepared ",Object(r.b)("inlineCode",{parentName:"p"},"Trainer")," instance will contain the dataset information, this method will be very useful if our models depend on the information."))),Object(r.b)("h3",{id:"register--apply"},"Register & Apply"),Object(r.b)("p",null,"After defining the ",Object(r.b)("inlineCode",{parentName:"p"},"forward")," (and probably the ",Object(r.b)("inlineCode",{parentName:"p"},"_init_with_trainer"),") method, we need to ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:"/docs/design-principles#register-mechanism"}),"register")," our model to apply it in ",Object(r.b)("inlineCode",{parentName:"p"},"carefree-learn"),":"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'@ModelProtocol.register("my_fancy_model")\nclass MyFancyModel(ModelProtocol):\n    def __init__(self, foo):\n        super().__init__()\n        self.foo = foo\n\n    def forward(\n        self,\n        batch_idx: int,\n        batch: tensor_dict_type,\n        state: Optional["TrainerState"] = None,\n        **kwargs: Any,\n    ) -> tensor_dict_type:\n        ...\n')),Object(r.b)("p",null,"After which we can:"),Object(r.b)("ul",null,Object(r.b)("li",{parentName:"ul"},"set the ",Object(r.b)("inlineCode",{parentName:"li"},"model_name")," in ",Object(r.b)("a",Object(a.a)({parentName:"li"},{href:"/docs/getting-started/configurations#dlsimplepipeline"}),Object(r.b)("inlineCode",{parentName:"a"},"Pipeline"))," to the corresponding name to apply it."),Object(r.b)("li",{parentName:"ul"},"set the ",Object(r.b)("inlineCode",{parentName:"li"},"model_config")," in ",Object(r.b)("a",Object(a.a)({parentName:"li"},{href:"/docs/getting-started/configurations#dlsimplepipeline"}),Object(r.b)("inlineCode",{parentName:"a"},"Pipeline"))," to the corresponding configurations.")),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-python"}),'m = cflearn.cv.CarefreePipeline("my_fancy_model", {"foo": "bar"})\nm.build({})\nprint(m.model.foo)  # bar\n')),Object(r.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(r.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"For Machine Learning tasks, the APIs will remain the same but the internal design will be a little different. Please refer to the ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:"machine-learning-customization#mlmodel"}),Object(r.b)("inlineCode",{parentName:"a"},"MLModel"))," section for more details."))),Object(r.b)("h2",{id:"customize-training-loop"},"Customize Training Loop"),Object(r.b)("div",{className:"admonition admonition-caution alert alert--warning"},Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"16",height:"16",viewBox:"0 0 16 16"}),Object(r.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"})))),"caution")),Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"To be continued..."))))}s.isMDXComponent=!0},96:function(e,t,n){"use strict";n.d(t,"a",(function(){return p})),n.d(t,"b",(function(){return u}));var a=n(0),i=n.n(a);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function c(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},r=Object.keys(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var b=i.a.createContext({}),s=function(e){var t=i.a.useContext(b),n=t;return e&&(n="function"==typeof e?e(t):c(c({},t),e)),n},p=function(e){var t=s(e.components);return i.a.createElement(b.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return i.a.createElement(i.a.Fragment,{},t)}},m=i.a.forwardRef((function(e,t){var n=e.components,a=e.mdxType,r=e.originalType,o=e.parentName,b=l(e,["components","mdxType","originalType","parentName"]),p=s(n),m=a,u=p["".concat(o,".").concat(m)]||p[m]||d[m]||r;return n?i.a.createElement(u,c(c({ref:t},b),{},{components:n})):i.a.createElement(u,c({ref:t},b))}));function u(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var r=n.length,o=new Array(r);o[0]=m;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:a,o[1]=c;for(var b=2;b<r;b++)o[b]=n[b];return i.a.createElement.apply(null,o)}return i.a.createElement.apply(null,n)}m.displayName="MDXCreateElement"}}]);