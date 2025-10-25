(async function(){
  try{
    const res=await fetch("/static/hero_questions.json");
    const list=await res.json();
    const g=document.getElementById("grid"); if(!g) return;
    g.innerHTML="";
    list.forEach(it=>{
      const d=document.createElement("div");
      d.className="card";
      d.innerHTML=`<div style="font-weight:700">${it.q}</div><div class="badge">${it.cat||""}</div>`;
      d.onclick=()=>{ const q=document.getElementById("q"); if(q){ q.value=it.q; document.getElementById("go").click(); } };
      g.appendChild(d);
    });
  }catch(e){}
})();
