import{d as r,e as t,o,c as p,b as d,_ as g}from"./app-QVsodGo1.js";const l=["href","title"],i=["src","alt"],m=r({__name:"NpmBadge",props:{package:{type:String,required:!0},distTag:{type:String,required:!1,default:"next"}},setup(a){const e=a,n=t(()=>`https://www.npmjs.com/package/${e.package}`),c=t(()=>e.distTag?`${e.package}@${e.distTag}`:e.package),s=t(()=>`https://badgen.net/npm/v/${e.package}/${e.distTag}?label=${encodeURIComponent(c.value)}`);return(u,_)=>(o(),p("a",{class:"npm-badge",href:n.value,title:a.package,target:"_blank",rel:"noopener noreferrer"},[d("img",{src:s.value,alt:a.package},null,8,i)],8,l))}}),k=g(m,[["__scopeId","data-v-c758b2a0"],["__file","NpmBadge.vue"]]);export{k as default};
