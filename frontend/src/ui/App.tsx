import { useState } from "react";
async function ask(text:string, provider="stub"){
  const base = ""; // same-origin API
  const r = await fetch(`${base}/api/answer`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,provider})});
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
export default function App(){
  const [q,setQ]=useState(""); const [p,setP]=useState("stub"); const [url,setUrl]=useState(""); const [err,setErr]=useState(""); const [busy,setBusy]=useState(false);
  return (
    <div style={{maxWidth:800,margin:"40px auto",padding:"16px",fontFamily:"ui-sans-serif"}}>
      <h1>Explaina</h1>
      <textarea rows={4} value={q} onChange={e=>setQ(e.target.value)} placeholder="Ask anything…" style={{width:"100%",padding:8}}/>
      <div style={{margin:"8px 0"}}>
        <label>Renderer: </label>
        <select value={p} onChange={e=>setP(e.target.value)}>
          <option value="stub">Demo (instant)</option>
          <option value="default">Standard</option>
          <option value="sora2" disabled>Pro (Sora 2) — soon</option>
        </select>
      </div>
      <button disabled={busy} onClick={async()=>{setErr("");setUrl("");setBusy(true);try{const job=await ask(q,p); setUrl(job.video_url||""); if(!job.video_url) setErr("No video_url returned.");}catch(e:any){setErr(e.message)}finally{setBusy(false)}}}>
        {busy?"Generating…":"Generate video"}
      </button>
      {err && <div style={{color:"#f88",marginTop:8}}>{err}</div>}
      {url && <div style={{marginTop:16}}><video src={url} controls style={{width:"100%",borderRadius:12}}/></div>}
      {!url && !busy && <ul><li onClick={()=>setQ("Explain lift in aerodynamics")}>Explain lift</li></ul>}
    </div>
  );
}
