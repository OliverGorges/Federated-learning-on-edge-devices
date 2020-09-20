(function(t){function e(e){for(var n,i,r=e[0],c=e[1],l=e[2],d=0,m=[];d<r.length;d++)i=r[d],Object.prototype.hasOwnProperty.call(o,i)&&o[i]&&m.push(o[i][0]),o[i]=0;for(n in c)Object.prototype.hasOwnProperty.call(c,n)&&(t[n]=c[n]);u&&u(e);while(m.length)m.shift()();return s.push.apply(s,l||[]),a()}function a(){for(var t,e=0;e<s.length;e++){for(var a=s[e],n=!0,r=1;r<a.length;r++){var c=a[r];0!==o[c]&&(n=!1)}n&&(s.splice(e--,1),t=i(i.s=a[0]))}return t}var n={},o={app:0},s=[];function i(e){if(n[e])return n[e].exports;var a=n[e]={i:e,l:!1,exports:{}};return t[e].call(a.exports,a,a.exports,i),a.l=!0,a.exports}i.m=t,i.c=n,i.d=function(t,e,a){i.o(t,e)||Object.defineProperty(t,e,{enumerable:!0,get:a})},i.r=function(t){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},i.t=function(t,e){if(1&e&&(t=i(t)),8&e)return t;if(4&e&&"object"===typeof t&&t&&t.__esModule)return t;var a=Object.create(null);if(i.r(a),Object.defineProperty(a,"default",{enumerable:!0,value:t}),2&e&&"string"!=typeof t)for(var n in t)i.d(a,n,function(e){return t[e]}.bind(null,n));return a},i.n=function(t){var e=t&&t.__esModule?function(){return t["default"]}:function(){return t};return i.d(e,"a",e),e},i.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},i.p="/";var r=window["webpackJsonp"]=window["webpackJsonp"]||[],c=r.push.bind(r);r.push=e,r=r.slice();for(var l=0;l<r.length;l++)e(r[l]);var u=c;s.push([0,"chunk-vendors"]),a()})({0:function(t,e,a){t.exports=a("56d7")},"02b8":function(t,e,a){},"1b58":function(t,e,a){},"1ed1":function(t,e,a){"use strict";var n=a("1b58"),o=a.n(n);o.a},2020:function(t,e,a){"use strict";var n=a("02b8"),o=a.n(n);o.a},"438e":function(t,e,a){"use strict";var n=a("704f"),o=a.n(n);o.a},"56d7":function(t,e,a){"use strict";a.r(e);a("e260"),a("e6cf"),a("cca6"),a("a79d");var n=a("2b0e"),o=a("8c4f"),s=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",[a("b-row",{staticClass:"header"},[a("b-col",[a("a",{staticClass:"header__logo",attrs:{href:"https://www.capgemini.com/"}},[a("img",{attrs:{src:"https://www.capgemini.com/wp-content/themes/capgemini-komposite/assets/images/logo.svg",alt:"Capgemini Worldwide"}})])]),a("b-col",[a("h3",[t._v("Federated Process")])]),a("b-col",[a("nav",[a("router-link",{staticClass:"link",attrs:{to:"/"}},[t._v("Home")]),t._v(" / "),a("router-link",{staticClass:"link",attrs:{to:"/create"}},[t._v("Create Plan")]),t._v(" / "),a("a",{staticClass:"link router-link-active",attrs:{href:"#"},on:{click:t.runDemo}},[t._v("Run Demo")])],1)])],1),a("router-view")],1)},i=[],r={name:"App",methods:{runDemo:function(){this.axios.get("http://".concat(window.location.host,"/demo"))}}},c=r,l=(a("1ed1"),a("2877")),u=Object(l["a"])(c,s,i,!1,null,"f949a368",null),d=u.exports,m=a("5f5b"),f=a("b1e0"),p=a("5681"),b=a.n(p),h=a("51a9"),g=a.n(h),v=a("bc3a"),w=a.n(v),k=a("a7fe"),x=a.n(k),_=(a("f9e3"),a("2dd8"),function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("b-container",{staticClass:" text-center small",attrs:{fluid:""}},[a("b-row",{staticClass:"mb-4 diagram-row-20"},[a("b-col",{attrs:{md:"6","offset-md":"4"}},[a("Step",{attrs:{id:4,name:"Aggregation",info:"https://blog.bindwise.com/why-online-seller-have-to-get-rid-of-manual-inventory-management/",img:"./assets/invenotry-grow.gif",step:t.globalStep,msg:t.aggregationInfo,data:t.results,keys:["metric","old","new"],buttonAction:t.aggregateButton,buttonMsg:t.aggregationButtonText}})],1)],1),a("b-row",{staticClass:"mb-3 diagram-row-30"},[a("b-col",{staticClass:"p-3 ",attrs:{md:"4"}},[a("Step",{attrs:{id:1,name:"GlobalModel",info:"https://4.bp.blogspot.com/-hLwbUkQIqpo/XO3IpFbdg4I/AAAAAAAzHoE/capc2dK0K4o1MjoJIxaQCepvL-fNaGh0ACLcBGAs/s1600/AW3884646_00.gif",img:"./assets/ai.gif",step:t.globalStep,data:t.modelData,keys:["name","mAP","ar","loss"],buttonAction:t.initButton,buttonMsg:"Load Models"}})],1),a("b-col",{staticClass:"ml-auto p-8 ",attrs:{md:"4"}},[a("Step",{attrs:{id:3,name:"Training",info:"https://albert.ai/wp-content/uploads/2016/09/blinkingBrain.gif",img:"./assets/train.gif",step:t.globalStep,data:t.meta,keys:["id","step","loss","acc"]}})],1)],1),a("b-row",{staticClass:"diagram-row-20"},[a("b-col",{attrs:{md:"7","offset-md":"4"}},[a("b-row",[a("b-col",{attrs:{md:"4"}},[a("Step",{attrs:{id:2,name:"Plan",info:"https://cdn.lowgif.com/medium/d5e6f5cb080df421-clock-sticker-for-ios-android-giphy.gif",img:"./assets/watch.gif",step:t.globalStep,data:[],buttonAction:t.createButton,buttonMsg:"Create Plan"}})],1),a("b-col",{staticClass:"ml-n1",attrs:{md:"4"}},[a("table",{staticClass:"table"},[a("tbody",t._l(t.task,(function(e,n){return a("tr",{key:n},[a("th",{attrs:{scope:"row"}},[t._v(t._s(n))]),n.includes("Time")?a("td",[t._v(t._s(t.format(e)))]):a("td",[t._v(t._s(e))])])})),0)])])],1)],1)],1)],1)}),y=[],C=(a("99af"),a("4160"),a("caad"),a("a9e3"),a("8ba4"),a("b680"),a("4fad"),a("2532"),a("159b"),function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("b-container",{attrs:{fluid:""}},[a("b-row",{staticClass:"diagram-row"},[a("b-col",{attrs:{md:"6","offset-md":"1"}},[a("b-row",{class:{colour:t.isActive}},[a("h5",[t._v("Step: "+t._s(t.id)+": "+t._s(t.name))])]),a("b-row",[a("vue-freezeframe",{ref:"freeze",staticClass:"animation",class:{inactive:!t.isActive},attrs:{src:t.img}})],1),t.buttonMsg&&t.isNext?a("b-row",[a("b-button",{on:{click:t.buttonAction}},[t._v(t._s(t.buttonMsg))])],1):t._e(),t.msg?a("b-row",[t._v(" "+t._s(t.msg)+" ")]):t._e(),t.data.length>0?a("b-row",[a("b-table",{attrs:{striped:"",hover:"",items:t.data,fields:t.keys}})],1):t._e()],1)],1)],1)}),S=[],T={name:"Step",props:{id:Number,name:String,img:String,data:Array,keys:Array,step:Number,msg:String,buttonMsg:String,buttonAction:Function},computed:{isActive:function(){return this.step==this.id},isNext:function(){return this.step+1==this.id}},mounted:function(){this.toogleAnimation()},methods:{toogleAnimation:function(){console.log("Animation ",this.id,":",this.isActive),this.isActive?this.$refs.freeze.start():this.$refs.freeze.stop()}},watch:{isActive:function(){this.$refs.freeze.toggle()}}},A=T,O=(a("438e"),Object(l["a"])(A,C,S,!1,null,"2a14133e",null)),D=O.exports,I={name:"Overview",components:{Step:D},props:{msg:String},data:function(){return{globalStep:0,task:{},uploadLink:"http://".concat(window.location.host,"/"),taskKeys:["Time","Task","ModelVersion","Data"],meta:[],modelData:[],results:[],trainInterval:null,aggregationInfo:null}},sockets:{connect:function(){console.log("socket connected")},aggregation:function(t){var e=this;console.log(t),this.aggregationInfo=t.msg,2==t.status&&this.axios.get("http://".concat(window.location.host,"/eval")).then((function(t){var a=t.data;e.results=a}))},status:function(t){var e=this;this.globalStep=t.Step;var a=["nextTime","endTime","RegisteredClients","AcceptedClients","CompletedTasks"];t.ActiveTask&&a.forEach((function(a){console.log(t[a]),0!==t[a]&&e.$set(e.task,a,t[a])}))}},mounted:function(){var t=this,e=this;e.axios.get("http://".concat(window.location.host,"/status")).then((function(e){var a;if(e.data.Step>1)for(a=1;a<=e.data.Step;a++)t.updateData(a)}))},computed:{aggregationButtonText:function(){return"Aggregate ".concat(this.task.CompletedTasks||0," clients")}},methods:{format:function(t){var e=new Date(1e3*t);return"".concat(e.getHours(),":").concat(e.getMinutes())},updateData:function(t){var e=this,a=this;switch(clearInterval(a.trainInterval),t){case 0:this.meta=[],this.modelData=[],this.results=[],this.task={};break;case 1:this.axios.get("http://".concat(window.location.host,"/models")).then((function(t){var e=t.data;a.modelData=[],e.forEach((function(t){var e={};Object.entries(t).forEach((function(t){var a=t[1];console.log(a,Number(a)&&!Number.isInteger(a)),Number(a)&&!Number.isInteger(a)&&(a=Number(a).toFixed(3)),e[t[0]]=a})),a.modelData.push(e),a.Step=1}))}));break;case 2:this.axios.get("http://".concat(window.location.host,"/task/0")).then((function(t){var a=t.data;console.log(Object.entries(a)),Object.entries(a).forEach((function(t){console.log(t[0]),e.taskKeys.includes(t[0])&&e.$set(e.task,t[0],t[1])}))}));break;case 3:a.trainInterval=setInterval((function(){a.axios.get("http://".concat(window.location.host,"/meta")).then((function(t){console.log(t.data),a.meta=[],t.data.forEach((function(t){var e={};Object.entries(t).forEach((function(t){var a=t[1];Number(a)&&!Number.isInteger(a)&&(a=Number(a).toFixed(3)),e[t[0]]=a})),a.meta.push(e)}))}))}),5e3);break}},initButton:function(){this.globalStep=1},createButton:function(){this.$router.push({path:"create"})},aggregateButton:function(){this.axios.get("http://".concat(window.location.host,"/aggregate"))}},watch:{globalStep:function(){this.updateData(this.globalStep)}}},M=I,j=(a("2020"),Object(l["a"])(M,_,y,!1,null,"46c384f9",null)),E=j.exports,P=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("b-container",[a("b-form",{attrs:{action:t.url,method:"POST"}},[a("b-form-group",{attrs:{"label-for":"task",label:"Task:"}},[a("b-form-input",{attrs:{type:"text",state:t.validateTask,id:"task",name:"Task",placeholder:"Name of the Task"},model:{value:t.taskname,callback:function(e){t.taskname=e},expression:"taskname"}}),a("b-form-invalid-feedback",{attrs:{state:t.validateTask}},[t._v(" Taskname is to short or already exists for the selected BaseModel! ")])],1),a("b-form-group",{attrs:{"label-for":"case",label:"Case"}},[a("b-form-select",{attrs:{id:"case",name:"Case",options:t.cases},model:{value:t.selectedCase,callback:function(e){t.selectedCase=e},expression:"selectedCase"}})],1),a("b-form-group",{attrs:{"label-for":"data",label:"Data"}},[a("b-form-select",{attrs:{id:"data",name:"Data",options:t.availabeData}})],1),a("b-form-group",{attrs:{"label-for":"version",label:"BaseModel"}},[a("b-form-select",{attrs:{id:"version",name:"ModelVersion",options:t.models},model:{value:t.base,callback:function(e){t.base=e},expression:"base"}})],1),a("b-form-group",{attrs:{"label-for":"maxClients",label:"MaxClients"}},[a("b-form-input",{attrs:{type:"number",min:"2",max:"100",id:"maxClients",name:"MaxClients",placeholder:"2-100"}})],1),a("b-form-group",{attrs:{"label-for":"date",label:"Date"}},[a("b-form-datepicker",{attrs:{type:"date",id:"date",value:t.date},on:{change:function(e){return t.updateTime()}},model:{value:t.date,callback:function(e){t.date=e},expression:"date"}})],1),a("b-form-group",{attrs:{"label-for":"time",label:"Time"}},[a("b-form-timepicker",{attrs:{type:"time",id:"time",value:t.time,locale:"de","now-button":""},on:{change:function(e){return t.updateTime()}},model:{value:t.time,callback:function(e){t.time=e},expression:"time"}})],1),a("b-form-group",{staticClass:"hidden",attrs:{"label-for":"utime",value:t.unixtime,label:"unixtime"}},[a("b-form-input",{attrs:{type:"text",id:"utime",readonly:!0,name:"Time"},model:{value:t.unixtime,callback:function(e){t.unixtime=e},expression:"unixtime"}})],1),a("b-button",{attrs:{type:"submit",variant:"primary"}},[t._v("Submit")])],1)],1)},B=[],N=(a("b0c0"),a("ac1f"),a("1276"),{name:"CreatePlan",data:function(){return{date:"",time:"",url:"",taskname:"",base:"",unixtime:0,selectedCase:"",data:{},cases:["Coco"],models:[]}},mounted:function(){var t=this;this.url="http://".concat(window.location.host,"/federatedPlan");var e=new Date,a=e.toISOString().substr(0,10),n=e.toISOString().substr(11,5);console.log("Date ",a),console.log("Time ",n),document.getElementById("date").value=a,document.getElementById("time").value=n,this.axios.get("http://".concat(window.location.host,"/models")).then((function(e){console.log(e.data),t.models=[],e.data.forEach((function(e){console.log(e);var a=e.name.split(".")[0];t.models.push(a)}))})),this.axios.get("http://".concat(window.location.host,"/cases")).then((function(e){console.log(e.data),t.cases=[],t.data={},e.data.forEach((function(e){t.cases.push(e.name),t.data[e.name]=e.data}))}))},methods:{updateTime:function(){console.log(this.date,this.time)}},computed:{availabeData:function(){return this.data[this.selectedCase]||["Images"]},validateTask:function(){return 0=="".concat(this.taskname).length?null:!this.models.includes("".concat(this.base,"_").concat(this.taskname))&&"".concat(this.taskname).length>1}},watch:{date:function(){console.log("Update "+this.date+"T"+this.time+":00.000Z"),this.unixtime=new Date(this.date+"T"+this.time+".000Z").getTime()/100},time:function(){console.log("Update "+this.date+"T"+this.time+":00.000Z"),this.unixtime=new Date(this.date+"T"+this.time+".000Z").getTime()/100},taskname:function(){}}}),$=N,z=(a("cd7e"),Object(l["a"])($,P,B,!1,null,"a90d2532",null)),K=z.exports,F=a("5132"),L=a.n(F);n["default"].config.productionTip=!1,n["default"].use(m["a"]),n["default"].use(f["a"]),n["default"].component("freeze",g.a),n["default"].use(b.a),n["default"].use(x.a,w.a),n["default"].use(o["a"]);var Z=new o["a"]({routes:[{path:"/",name:"home",component:E},{path:"/create",name:"create",component:K}]});n["default"].use(new L.a({debug:!0,connection:"http://".concat(window.location.host,"/"),vuex:{actionPrefix:"SOCKET_",mutationPrefix:"SOCKET_"}})),new n["default"]({render:function(t){return t(d)},router:Z}).$mount("#app")},"704f":function(t,e,a){},adb4:function(t,e,a){},cd7e:function(t,e,a){"use strict";var n=a("adb4"),o=a.n(n);o.a}});
//# sourceMappingURL=app.b7512480.js.map