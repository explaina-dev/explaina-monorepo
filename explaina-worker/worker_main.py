# worker_main.py — Explaina Render Worker (Oct 2025)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Explaina Render Worker")

@app.get("/")
def root():
    return JSONResponse({"service": "explaina-worker", "hint": "Try /health or /api/render/start"})

@app.get("/health")
def health():
    return {"status":"ok", "service": "explaina-worker"}

@app.get("/config")
def get_config():
    """Debug endpoint to check environment configuration"""
    return {
        "USE_GCS": USE_GCS,
        "GCS_BUCKET": os.getenv("GCS_BUCKET", ""),
        "GCS_FOLDER": os.getenv("GCS_FOLDER", ""),
        "GCS_SIGN_URLS": os.getenv("GCS_SIGN_URLS", ""),
        "GCS_SIGN_EXP_SECONDS": os.getenv("GCS_SIGN_EXP_SECONDS", ""),
        "CLEAN_MEDIA_FILES": CLEAN_MEDIA_FILES,
        "storage_available": storage is not None,
        "MAX_WORKERS": MAX_WORKERS,
        "DEFAULT_PRESET": DEFAULT_PRESET
    }

def norm(p) -> str:
    """Normalize any path for ffmpeg (no backslashes)."""
    return str(p).replace("\\", "/")

# now import everything else…
import os, time, hashlib, subprocess, uuid
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from subprocess import Popen

# Optional GCS import must be guarded
USE_GCS = bool(os.getenv("GCS_BUCKET", "").strip())
try:
    if USE_GCS:
        from google.cloud import storage  # only used later
    else:
        storage = None
except Exception:
    storage = None
    USE_GCS = False

# ---------- Config ----------
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "media"))  # don't create at import time

# Defensive parsing for Cloud Run environment
def _safe_int(key: str, default: int, minimum: int = 1) -> int:
    """Parse env var to int with fallback and minimum clamping."""
    try:
        val = int(os.getenv(key, str(default)) or default)
        return max(val, minimum)
    except (ValueError, TypeError):
        return default

MAX_WORKERS = _safe_int("MAX_WORKERS", 2, minimum=1)
DEFAULT_PRESET = (os.getenv("RENDER_PRESET_DEFAULT", "mobile") or "mobile").lower()  # mobile | hd
JOB_MAX_AGE_SEC = _safe_int("JOB_MAX_AGE_SEC", 24*3600, minimum=60)
CLEAN_MEDIA_FILES = os.getenv("CLEAN_MEDIA_FILES", "false").lower() == "true"

# Log configuration for debugging Cloud Run startup (flush immediately)
import sys
print(f"[CONFIG] MAX_WORKERS={MAX_WORKERS}, JOB_MAX_AGE_SEC={JOB_MAX_AGE_SEC}, MEDIA_DIR={MEDIA_DIR}", flush=True)
print(f"[CONFIG] USE_GCS={USE_GCS}, DEFAULT_PRESET={DEFAULT_PRESET}, CLEAN_MEDIA_FILES={CLEAN_MEDIA_FILES}", flush=True)
sys.stdout.flush()
sys.stderr.flush()

def _ensure_media_dir():
    """Create media dir lazily when needed"""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# Ensure media dir exists and mount it
_ensure_media_dir()
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# ---------- In-memory state ----------
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = Lock()

PROCS: Dict[str, Popen] = {}
PROCS_LOCK = Lock()

EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def _new_id() -> str:
    return hashlib.sha1(os.urandom(16)).hexdigest()[:16]

# ---------- Job helpers ----------
def job_create(meta: Dict) -> str:
    jid = _new_id(); now = time.time()
    with JOBS_LOCK:
        JOBS[jid] = {"status":"queued","percent":0,"result":None,"error":None,
                     "created":now,"updated":now, **meta}
    return jid

def job_update(jid: str, **kw):
    with JOBS_LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)
            JOBS[jid]["updated"] = time.time()

def job_get(jid: str) -> Optional[Dict]:
    with JOBS_LOCK:
        j = JOBS.get(jid)
        return dict(j) if j else None

def proc_register(jid: str, p: Optional[Popen]):
    with PROCS_LOCK:
        if p is None: PROCS.pop(jid, None)
        else: PROCS[jid] = p

def proc_cancel(jid: str) -> bool:
    with PROCS_LOCK:
        p = PROCS.get(jid)
    if not p: return False
    try:
        p.terminate()
        try: p.wait(timeout=5)
        except: p.kill()
        return True
    finally:
        proc_register(jid, None)

def _prune_jobs():
    now = time.time()
    with JOBS_LOCK:
        delete_ids = [jid for jid, j in JOBS.items()
                      if (now - j.get("updated", j.get("created", now))) > JOB_MAX_AGE_SEC
                      and j.get("status") in {"done","error","canceled"}]
        for jid in delete_ids:
            res = (JOBS.get(jid) or {}).get("result") or {}
            if CLEAN_MEDIA_FILES:
                for k in ("video_url","thumb_url"):
                    u = res.get(k)
                    if u and u.startswith("/"):
                        p = Path(u[1:])
                        try:
                            if p.exists(): p.unlink()
                        except: pass
            JOBS.pop(jid, None)
    with PROCS_LOCK:
        for jid in list(PROCS.keys()):
            if jid not in JOBS:
                PROCS.pop(jid, None)

# ---------- Small media helpers ----------
def _ffprobe_duration(path: str) -> Optional[float]:
    try:
        p = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if p.returncode == 0:
            return float(p.stdout.strip())
    except: pass
    return None

def _chunks(text: str, wpm: int = 180) -> List[Tuple[float,float,str]]:
    words = (text or "").split()
    if not words: return [(0.0, 2.5, "Explaina")]
    cw = max(4, int(wpm*2.5/60)); out=[]; t=0.0
    for i in range(0, len(words), cw):
        seg = " ".join(words[i:i+cw]) or " "
        dur = max(2.0, 2.5*(len(seg.split())/cw))
        out.append((t, t+dur, seg)); t += dur
    return out

def _write_srt_single(chunks: List[Tuple[float,float,str]], path: Path):
    def fmt(x): return time.strftime("%H:%M:%S", time.gmtime(x)) + f",{int((x%1)*1000):03d}"
    with open(path, "w", encoding="utf-8") as f:
        for i,(a,b,txt) in enumerate(chunks,1):
            f.write(f"{i}\n{fmt(a)} --> {fmt(b)}\n{(txt or ' ')}\n\n")

def _write_srt_dual(chunks: List[Tuple[float,float,str]], path: Path):
    """Dual-line SRT (local + 'english'); for now the second line mirrors local text.
       Wire real translation later by replacing the second line."""
    def fmt(x): return time.strftime("%H:%M:%S", time.gmtime(x)) + f",{int((x%1)*1000):03d}"
    with open(path, "w", encoding="utf-8") as f:
        for i,(a,b,txt) in enumerate(chunks,1):
            line2 = txt  # TODO: replace with real translation if desired
            f.write(f"{i}\n{fmt(a)} --> {fmt(b)}\n{(txt or ' ')}\n{(line2 or ' ')}\n\n")

# ---------- Optional GCS upload ----------
def _gcs_upload(local_path: str) -> str:
    if not (USE_GCS and storage):
        raise RuntimeError("GCS not available")
    
    bucket_name = os.getenv("GCS_BUCKET", "").strip()
    folder      = (os.getenv("GCS_FOLDER", "renders") or "renders").strip().strip("/")
    sign        = os.getenv("GCS_SIGN_URLS","false").lower() == "true"
    ttl         = int(os.getenv("GCS_SIGN_EXP_SECONDS","31536000"))
    cachectl    = os.getenv("GCS_CACHE_CONTROL","public, max-age=31536000, immutable")

    client = storage.Client()  # type: ignore
    bucket = client.bucket(bucket_name)
    filename = Path(local_path).name
    name   = f"{folder}/{filename}" if folder else filename
    blob   = bucket.blob(name)
    blob.cache_control = cachectl
    blob.upload_from_filename(local_path)
    blob.patch()
    if sign:
        return blob.generate_signed_url(expiration=ttl, method="GET")
    else:
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return blob.generate_signed_url(expiration=ttl, method="GET")

# ---------- Visual Plan helpers ----------
from typing import Optional

# Map logical asset names to pre-rendered alpha loops (put these in /app/static/effects/)
ASSET_MAP = {
    "clouds":       "static/effects/clouds.mov",        # alpha or dark-on-black
    "rain":         "static/effects/rain.mov",          # alpha streaks
    "puddle":       "static/effects/puddle.mov",        # subtle ripples
    "airfoil":      "static/effects/airfoil.mov",
    "flow_arrows":  "static/effects/flow_arrows.mov",
    "lift_arrow":   "static/effects/lift_arrow.mov",
    "chart":        "static/effects/chart.mov",
    "arrow_up":     "static/effects/arrow_up.mov",
    "coin":         "static/effects/coin.mov"
}

def _resolve_asset(path: str) -> Optional[str]:
    """Return file path if it exists (absolute), else None."""
    p = Path(path)
    if p.exists():
        return str(p)
    # allow relative without leading slash
    p2 = Path("/app") / path if not path.startswith("/") else Path(path)
    return str(p2) if p2.exists() else None

def _pick_bg_from_plan(plan: dict, tgt_w: int, tgt_h: int, dur: Optional[float]):
    """
    Return (inputs_list, base_label, vf_prefix) building the background.
    Uses DEFAULT_BG if present; else color from plan.theme.bg; else default navy.
    """
    inputs = []
    vf_prefix = ""
    base_label = "[bg0]"
    # Try static image first
    env_bg = os.getenv("DEFAULT_BG", "").strip()
    plan_bg_img = (plan or {}).get("background_image")
    bg_img = None
    for cand in (plan_bg_img, env_bg):
        if cand:
            r = _resolve_asset(cand)
            if r: bg_img = r; break

    if bg_img:
        # loop image -> 0:v
        inputs += ["-loop", "1", "-i", bg_img]
        # scale & set sar
        vf_prefix += f"[0:v]scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=cover,setsar=1{base_label};"
    else:
        # solid color background based on theme.bg or default navy
        theme = (plan or {}).get("theme", {})
        bg_hex = str(theme.get("bg", "#0f1836")).lstrip("#")
        if len(bg_hex) == 6:
            bg_hex = "0x" + bg_hex
        else:
            # fallback in case of malformed
            bg_hex = "0x0f1836"
        color = f"color=c={bg_hex}:s={tgt_w}x{tgt_h}:r=30"
        if dur: color += f":d={max(1, int(dur + 0.5))}"
        inputs += ["-f", "lavfi", "-i", color]
        vf_prefix += f"[0:v]format=yuv420p{base_label};"
    return inputs, base_label, vf_prefix

def _collect_asset_inputs(plan_assets: list[str]) -> tuple[list[str], list[str]]:
    """
    Return (inputs_to_add, labels_for_overlay) for all found plan assets.
    Each asset becomes a new input index in ffmpeg: -i asset
    We'll layer them sequentially with overlay filters.
    """
    inputs = []
    labels = []
    for asset_name in plan_assets or []:
        path = ASSET_MAP.get(asset_name)
        if not path:
            continue
        r = _resolve_asset(path)
        if r:
            # Use -stream_loop -1 to keep it alive until shortest ends
            inputs += ["-stream_loop", "-1", "-i", r]
            labels.append("")  # placeholder, we compute labels by index later
    return inputs, labels

def _build_overlay_chain(base_lbl: str, num_asset_inputs: int, start_input_idx: int) -> tuple[str, str]:
    """
    Build overlay filter chain like:
    [bg0][1:v]overlay=... [bg1]; [bg1][2:v]overlay=... [bg2] ...
    Returns (vf_chain, last_label)
    """
    chain = ""
    cur = base_lbl
    for i in range(num_asset_inputs):
        inp = f"[{start_input_idx + i}:v]"
        nxt = f"[bg{i+1}]"
        # If your assets are alpha, overlay default works; if black matte, use 'screen' blend path.
        # Here we assume alpha MOVs:
        chain += f"{cur}{inp}overlay=(W-w)/2:(H-h)/2:format=auto{nxt};"
        cur = nxt
    return chain, cur

# ---------- Composer (background) ----------
def _compose(jid: str, *, text: str, audio_url: str, lang: str, preset: str, subtitle_mode: str, watermark: str, video_plan: dict | None = None):
    try:
        job_update(jid, status="preparing", percent=5)
        _ensure_media_dir()  # Create media dir lazily when first needed

        tgt_w,tgt_h,crf,abr = (1920,1080,"21","128k") if preset=="hd" else (960,540,"23","96k")

        a_path = audio_url if Path(audio_url).exists() else ("."+audio_url if str(audio_url).startswith("/") else str(audio_url))
        if not Path(a_path).exists():
            # Auto-create a tiny silent mp3 so renders never fail
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            silent_mp3 = MEDIA_DIR / f"silence-{uuid.uuid4().hex[:8]}.mp3"
            subprocess.run([
                "ffmpeg","-f","lavfi","-i","anullsrc=r=48000:cl=mono","-t","2",
                "-q:a","9","-acodec","libmp3lame", str(silent_mp3), "-y"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            a_path = str(silent_mp3)

        wm_path = watermark if Path(watermark).exists() else ("."+watermark if str(watermark).startswith("/") else str(watermark))
        if not Path(wm_path).exists():
            wm_path = ""  # optional

        vid_id   = hashlib.sha1(os.urandom(12)).hexdigest()[:12]
        srt      = MEDIA_DIR / f"{vid_id}.srt"
        out_mp4  = MEDIA_DIR / f"{vid_id}.mp4"
        out_jpg  = MEDIA_DIR / f"{vid_id}.jpg"

        # Subtitles (local or dual)
        chunks = _chunks(text or "Explaina")
        if subtitle_mode == "dual":
            _write_srt_dual(chunks, srt)
        else:
            _write_srt_single(chunks, srt)
        job_update(jid, status="subtitles", percent=20)

        dur = _ffprobe_duration(a_path)
        job_update(jid, status="encoding", percent=40)

        # ----- Build inputs: Background + Assets + Audio (+ Watermark) -----
        plan = video_plan or {}
        bg_inputs, base_label, vf_prefix = _pick_bg_from_plan(plan, tgt_w, tgt_h, dur)
        
        # Collect assets from all scenes (aggregate approach)
        plan_assets = []
        if plan.get("scenes"):
            for scene in plan["scenes"]:
                scene_assets = scene.get("assets", [])
                if scene_assets:
                    plan_assets.extend(scene_assets)
        else:
            # Fallback to top-level assets if no scenes
            plan_assets = plan.get("assets", [])
        
        asset_inputs, asset_labels = _collect_asset_inputs(plan_assets)
        num_assets = len(asset_labels)  # Count actual asset streams, not CLI tokens
        inputs = [] + bg_inputs + [ "-i", a_path ] + asset_inputs
        audio_idx = 1  # audio is always second input (after bg which is index 0)

        # Add overlay SVG if present in plan
        overlay_path = plan.get("overlay")
        overlay_idx = None
        if overlay_path:
            overlay_resolved = _resolve_asset(overlay_path)
            if overlay_resolved:
                inputs += ["-stream_loop", "-1", "-i", overlay_resolved]
                overlay_idx = 2 + num_assets  # bg=0, audio=1, assets=2..., overlay next
                num_assets += 1  # increment to account for overlay in watermark index calculation

        if wm_path:  # watermark input at the end if available
            inputs += ["-i", wm_path]
            wm_idx = 2 + num_assets  # bg=0, audio=1, assets=2..., (overlay?), watermark=last
        else:
            wm_idx = None

        # ----- Filter graph chain -----
        vf = vf_prefix                    # e.g., [0:v]... [bg0];
        # Overlay assets one by one (excluding the FX overlay which we handle separately)
        num_regular_assets = num_assets - (1 if overlay_idx else 0)
        asset_chain, after_assets = _build_overlay_chain(base_label, num_regular_assets, 2)  # assets start at [2:v]
        vf += asset_chain

        # Apply FX overlay with colorkey and screen blend mode (if present)
        if overlay_idx is not None:
            # First apply colorkey to the overlay stream to make transparent, then overlay/blend it
            vf += f"[{overlay_idx}:v]colorkey=0x000000:0.1:0.3[fx];"
            vf += f"{after_assets}[fx]overlay=0:0:format=auto[ovr];"
            after_overlay = "[ovr]"
        else:
            after_overlay = after_assets

        # --- subtitles node (safer quoting) ---
        srt_norm = norm(srt)
        style    = "Fontsize=30,PrimaryColour=&H00FFFFFF&,OutlineColour=&H80000000&,BorderStyle=3,Outline=2,Shadow=0"
        vf      += f"{after_overlay}subtitles='{srt_norm}':force_style='{style}'[vsub];"

        # Watermark (optional)
        if wm_idx is not None:
            vf += f"[vsub][{wm_idx}:v]overlay=W-w-20:H-h-20:format=auto[vout];"
            last_lbl = "vout"
        else:
            # No extra filter needed; just continue with vsub as the final label
            last_lbl = "vsub"

        # ----- Build FFmpeg command -----
        cmd = ["ffmpeg", *inputs, "-shortest",
               "-filter_complex", vf,
               "-map", f"[{last_lbl}]", "-map", f"{audio_idx}:a",  # video from graph, audio is the first audio input
               "-c:v","libx264","-preset","ultrafast","-crf", crf,
               "-profile:v","baseline","-level","3.1","-pix_fmt","yuv420p",
               "-g","60","-keyint_min","60",
               "-c:a","aac","-b:a", abr,
               "-movflags","+faststart",
               str(out_mp4), "-y"]

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        proc_register(jid, p)
        start = time.time()
        while True:
            rc = p.poll()
            if rc is not None: break
            if dur:
                elapsed = time.time() - start
                pct = 40 + min(45, max(0, int(45 * (elapsed/max(1.0, dur)))))
                job_update(jid, percent=pct)
            time.sleep(0.4)
        proc_register(jid, None)

        if p.returncode != 0:
            err_tail = (p.stderr.read() if p.stderr else "")[-400:]
            raise RuntimeError(f"ffmpeg error: {err_tail}")

        job_update(jid, status="thumbnail", percent=85)
        subprocess.run(["ffmpeg","-ss","00:00:01","-i",str(out_mp4),"-frames:v","1","-q:v","3",str(out_jpg),"-y"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # URLs (local or GCS)
        if USE_GCS and storage:
            try:
                print(f"[GCS] Uploading to bucket: {os.getenv('GCS_BUCKET')}", flush=True)
                vurl = _gcs_upload(str(out_mp4))
                turl = _gcs_upload(str(out_jpg))
                print(f"[GCS] Upload successful - video: {vurl[:80]}...", flush=True)
                if CLEAN_MEDIA_FILES:
                    for pth in (out_mp4, out_jpg):
                        try:
                            if Path(pth).exists(): Path(pth).unlink()
                        except: pass
            except Exception as e:
                import traceback
                print(f"[GCS] Upload failed: {e}", flush=True)
                print(f"[GCS] Traceback: {traceback.format_exc()}", flush=True)
                vurl = "/"+str(out_mp4); turl = "/"+str(out_jpg)
        else:
            print(f"[GCS] Skipping GCS upload (USE_GCS={USE_GCS}, storage={storage is not None})", flush=True)
            vurl = "/"+str(out_mp4); turl = "/"+str(out_jpg)

        job_update(jid, status="done", percent=100, result={"video_url": vurl, "thumb_url": turl})
        _prune_jobs()

    except Exception as e:
        job_update(jid, status="error", error=str(e), percent=100)

# ---------- API ----------
from fastapi import Body, HTTPException

@app.post("/api/render/start")
def render_start(payload: dict = Body(...)):
    """
    Input: { text, audio_url, watermark?, lang?, preset? (mobile|hd), subtitle_mode? ('local'|'dual') }
    """
    text          = (payload.get("text") or "Explaina").strip()
    audio_url     = payload.get("audio_url") or "static/sample.mp3"
    watermark     = payload.get("watermark") or "static/watermark.png"
    lang          = (payload.get("lang") or "en").lower()
    preset        = (payload.get("preset") or DEFAULT_PRESET).lower()
    subtitle_mode = (payload.get("subtitle_mode") or "local").lower()  # 'local' | 'dual'
    video_plan    = payload.get("video_plan")  # visual plan from /api/answer

    jid = job_create({"kind":"compose","lang":lang,"preset":preset,"subtitle_mode":subtitle_mode})
    EXECUTOR.submit(_compose, jid, text=text, audio_url=audio_url, lang=lang, preset=preset,
                    subtitle_mode=subtitle_mode, watermark=watermark, video_plan=video_plan)
    return {"ok": True, "job_id": jid}

@app.get("/api/progress")
def render_progress(job_id: str):
    j = job_get(job_id)
    if not j: raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True, "status": j["status"], "percent": j["percent"], "result": j.get("result"), "error": j.get("error")}

@app.post("/api/cancel")
def cancel_render(payload: dict = Body(...)):
    jid = (payload.get("job_id") or "").strip()
    if not jid: raise HTTPException(status_code=400, detail="job_id required")
    job_update(jid, status="canceled")
    canceled = proc_cancel(jid)
    return {"ok": True, "canceled": bool(canceled)}

@app.get("/api/last_jobs")
def last_jobs(limit: int = 10):
    with JOBS_LOCK:
        items = sorted(JOBS.items(), key=lambda kv: kv[1].get("updated", 0), reverse=True)[:max(1, min(limit, 50))]
        out = []
        for jid, j in items:
            out.append({
                "job_id": jid,
                "status": j.get("status"),
                "percent": j.get("percent"),
                "updated": j.get("updated"),
                "result": j.get("result"),
                "error": j.get("error"),
            })
        return {"ok": True, "jobs": out}

# Module loaded successfully
print("[WORKER] Module worker_main.py loaded successfully", flush=True)
