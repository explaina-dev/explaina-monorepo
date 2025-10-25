import os, json, time, uuid, subprocess, textwrap, hashlib, hmac, base64, zipfile, threading, re, copy
import io, csv
from subprocess import Popen
from collections import defaultdict, deque
from typing import Optional
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx
from fastapi import FastAPI, Request, Body, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------- Path normalization helper ----------
def norm(p: str | Path) -> str:
    """Normalize path for FFmpeg (convert backslashes to forward slashes)"""
    return str(p).replace("\\", "/")

# -------------------------------
# Config / env
# -------------------------------
ANSWER_API_URL = os.getenv("ANSWER_API_URL", "").strip()   # point to Azure later; for now we use /local-answer
RENDER_API_URL = os.getenv("RENDER_API_URL", "").strip()   # worker microservice URL
REQUIRE_HMAC   = os.getenv("REQUIRE_HMAC", "false").lower() == "true"
HMAC_SECRET    = os.getenv("HMAC_SECRET", "")
BACKUP_TOKEN   = os.getenv("BACKUP_TOKEN", "")

BLOCKED_COUNTRIES = set(os.getenv("BLOCKED_COUNTRIES","").split(",")) if os.getenv("BLOCKED_COUNTRIES") else set()
COUNTRY_RATE_MULTIPLIERS = {}
crm = os.getenv("COUNTRY_RATE_MULTIPLIERS", "")
if crm:
    for kv in crm.split(","):
        if ":" in kv:
            c,m = kv.split(":",1)
            try: COUNTRY_RATE_MULTIPLIERS[c.strip().upper()] = float(m)
            except: pass

VOICE_FOR = {
  "en": "en-US-JennyNeural",
  "es": "es-ES-ElviraNeural",
  "fr": "fr-FR-DeniseNeural",
  "ar": "ar-SA-ZariyahNeural",
  "hi": "hi-IN-SwaraNeural",
  "zh": "zh-CN-XiaoxiaoNeural",
  "pt": "pt-BR-FranciscaNeural",
  "ru": "ru-RU-SvetlanaNeural",
  "de": "de-DE-KatjaNeural",
  "ja": "ja-JP-NanamiNeural",
  "tr": "tr-TR-EmelNeural",
  "bn": "bn-BD-NabanitaNeural",
  "ur": "ur-PK-AsadNeural",
  "id": "id-ID-ArdiNeural",
  "sw": "sw-KE-RafikiNeural",
}

# ---- Landing / Meta ----
SITE_URL   = os.getenv("SITE_URL", "https://explaina.net")   # update later if needed
OG_IMAGE   = "/static/logo.png"                              # preview image
BRANDLINE  = "Ask anything → get a video explained instantly."

STATIC_DIR = Path("static")
MEDIA_DIR  = STATIC_DIR / "media"
BACKUP_DIR = Path("backups")
STATIC_DIR.mkdir(exist_ok=True, parents=True)
MEDIA_DIR.mkdir(exist_ok=True, parents=True)
BACKUP_DIR.mkdir(exist_ok=True, parents=True)
EVENTS_LOG = MEDIA_DIR / "events.jsonl"

# ---- Job store (in-memory) ----
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=2)  # tune workers for your Replit plan
JOBS: dict[str, dict] = {}
JOBS_LOCK = Lock()

JOB_PROCS: dict[str, Popen] = {}
JOB_PROCS_LOCK = Lock()

# --- Render cache ---
RENDER_CACHE: dict[str, dict] = {}
CACHE_LOCK = Lock()
CACHE_MAX = 200  # limit cache entries

# ---- Translation cache (simple in-memory) ----
TRANSLATE_CACHE: dict[tuple[str, str], str] = {}  # key=(lang, text), val=english
TRANSLATE_MAX_CHARS = 4500  # per request, under Google v2 limit

# Translation diagnostics
TRANSLATE_ENABLED = True  # you can flip this at runtime via /admin/translate/toggle
TRANSLATE_HITS = 0        # cache hits
TRANSLATE_MISSES = 0      # cache misses

# --- Job cleanup settings ---
JOB_MAX_AGE_SEC   = 24 * 3600        # prune jobs older than 24h
JOB_MAX_ENTRIES   = 500              # hard cap on how many jobs we remember
CLEAN_MEDIA_FILES = False            # set True to delete old mp4/jpg from /static/media when pruning

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="Explaina (Replit single-file)")

allow_origins = [
    "https://explaina.net",
    "https://www.explaina.net",
    "https://explaina-backend.azurewebsites.net",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Optional: periodic background cleanup (every 15 minutes) ---
def _cleanup_loop():
    while True:
        try:
            jobs_cleanup()
        except Exception:
            pass
        time.sleep(900)   # 15 minutes

threading.Thread(target=_cleanup_loop, daemon=True).start()

# -------------------------------
# Utilities
# -------------------------------
def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        ip = xff.split(",")[0].strip()
        if ip: return ip
    return request.client.host if request.client else "0.0.0.0"

def get_country_code(request: Request) -> str:
    for h in ("cf-ipcountry","x-vercel-ip-country","x-geo-country","x-country-code","fly-client-ip-country"):
        v = request.headers.get(h)
        if v: return v.strip().upper()
    return "ZZ"

def verify_hmac(request: Request, raw: bytes):
    if not REQUIRE_HMAC: return
    ts = request.headers.get("x-timestamp")
    sig = request.headers.get("x-signature")
    if not ts or not sig: raise HTTPException(status_code=401, detail="Missing auth headers")
    try: ts_i = int(ts)
    except: raise HTTPException(status_code=400, detail="Bad timestamp")
    if abs(int(time.time()) - ts_i) > 300: raise HTTPException(status_code=401, detail="Stale timestamp")
    if not HMAC_SECRET: raise HTTPException(status_code=401, detail="Server HMAC secret not set")
    msg = f"{ts}.{raw.decode('utf-8', errors='ignore')}".encode("utf-8")
    exp = base64.b64encode(hmac.new(HMAC_SECRET.encode(), msg, hashlib.sha256).digest()).decode()
    if not hmac.compare_digest(exp, sig): raise HTTPException(status_code=401, detail="Invalid signature")

# Simple rate limit + dedupe for /feedback
RATE_LIMIT_MAX, RATE_LIMIT_WINDOW = 10, 60
DUP_VOTE_WINDOW = 24*3600
_ip_hits: dict[str, deque] = defaultdict(deque)
_recent_votes: dict[str, float] = {}
_country_hits: dict[str, deque] = defaultdict(deque)

def rl_check(ip: str) -> bool:
    now = time.time()
    hits = _ip_hits[ip]
    while hits and now - hits[0] > RATE_LIMIT_WINDOW: hits.popleft()
    if len(hits) >= RATE_LIMIT_MAX: return False
    hits.append(now); return True

def rl_country_check(country:str) -> bool:
    now = time.time()
    mult = COUNTRY_RATE_MULTIPLIERS.get(country, 1.0)
    cap = max(10, int(RATE_LIMIT_MAX*100*mult))
    hits = _country_hits[country]
    while hits and now - hits[0] > RATE_LIMIT_WINDOW: hits.popleft()
    if len(hits) >= cap: return False
    hits.append(now); return True

def dedup_key(ip:str, session_id:str|None, answer_id:str)->str:
    return hashlib.sha256(f"{ip}|{session_id or ''}|{answer_id}".encode()).hexdigest()

def recent_vote(dkey:str)->bool:
    last = _recent_votes.get(dkey)
    return bool(last and (time.time() - last) < DUP_VOTE_WINDOW)

def remember_vote(dkey:str): _recent_votes[dkey] = time.time()

def job_create(initial: dict) -> str:
    job_id = uuid.uuid4().hex[:16]
    now = time.time()
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued", "percent": 0, "result": None, "error": None,
            "created": now, "updated": now, **initial
        }
    return job_id

def job_update(job_id: str, **kw):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kw)
            JOBS[job_id]["updated"] = time.time()

def job_get(job_id: str) -> dict | None:
    with JOBS_LOCK:
        return dict(JOBS.get(job_id) or {})

def job_register_proc(job_id: str, proc: Popen | None):
    with JOB_PROCS_LOCK:
        if proc is None:
            JOB_PROCS.pop(job_id, None)
        else:
            JOB_PROCS[job_id] = proc

def job_cancel(job_id: str) -> bool:
    """Returns True if a running process was found and signaled."""
    with JOB_PROCS_LOCK:
        proc = JOB_PROCS.get(job_id)
    if not proc:
        return False
    try:
        proc.terminate()  # send SIGTERM
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()   # SIGKILL as fallback
        return True
    except Exception:
        return False
    finally:
        job_register_proc(job_id, None)

def _safe_unlink(path_str: str):
    try:
        p = Path(path_str[1:] if path_str.startswith("/") else path_str)
        if p.exists() and p.is_file():
            p.unlink()
    except Exception:
        pass

def jobs_cleanup():
    """Prune old jobs and (optionally) their media outputs."""
    now = time.time()
    with JOBS_LOCK:
        # 1) Remove by age/status
        to_del = []
        for jid, j in JOBS.items():
            age = now - j.get("updated", j.get("created", now))
            if age >= JOB_MAX_AGE_SEC and j.get("status") in {"done","error","canceled"}:
                to_del.append(jid)

        # 2) If too many, also prune oldest beyond cap
        if len(JOBS) - len(to_del) > JOB_MAX_ENTRIES:
            # sort by updated ascending, skip ones already marked
            leftovers = sorted(
                ((jid, j.get("updated", j.get("created", 0))) for jid, j in JOBS.items() if jid not in to_del),
                key=lambda x: x[1]
            )
            extra = (len(JOBS) - len(to_del)) - JOB_MAX_ENTRIES
            to_del.extend([jid for jid, _ in leftovers[:max(0, extra)]])

        # 3) Delete
        for jid in to_del:
            j = JOBS.get(jid) or {}
            # optionally remove media files for completed jobs
            if CLEAN_MEDIA_FILES and j.get("result"):
                res = j["result"]
                if isinstance(res, dict):
                    v = res.get("video_url"); t = res.get("thumb_url")
                    if v: _safe_unlink(v)
                    if t: _safe_unlink(t)
            JOBS.pop(jid, None)

    # also clear any stray process handle (defensive)
    with JOB_PROCS_LOCK:
        for jid in list(JOB_PROCS.keys()):
            if jid not in JOBS:
                JOB_PROCS.pop(jid, None)

def run_ffmpeg(cmd:list[str]):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0: raise RuntimeError(p.stderr[-400:])
    return p

def write_srt(lines:list[tuple[float,float,str]], path:Path):
    with open(path,"w",encoding="utf-8") as f:
        for i,(a,b,t) in enumerate(lines,1):
            def fmt(x): return time.strftime("%H:%M:%S", time.gmtime(x)) + f",{int((x%1)*1000):03d}"
            f.write(f"{i}\n{fmt(a)} --> {fmt(b)}\n{t.strip()}\n\n")

def simple_caption_chunks(text:str, wpm:int=180):
    words = text.split()
    chunk_w = max(4, int(wpm*2.5/60))
    chunks = [" ".join(words[i:i+chunk_w]) for i in range(0,len(words),chunk_w)]
    t=0.0; out=[]
    for c in chunks:
        dur = max(2.0, 2.5*(len(c.split())/chunk_w))
        out.append((t,t+dur,c)); t+=dur
    return out

# --- Dual-subtitle helpers ---
def _chunk_text(s: str, maxlen: int = TRANSLATE_MAX_CHARS) -> list[str]:
    """
    Chunk at sentence-ish boundaries to keep responses aligned and under API limits.
    """
    if len(s) <= maxlen:
        return [s]
    chunks, buf = [], []
    total = 0
    for part in s.split(". "):  # light sentence split
        part2 = (part + ". ").strip()
        if total + len(part2) > maxlen and buf:
            chunks.append(" ".join(buf).strip())
            buf, total = [], 0
        buf.append(part2); total += len(part2)
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks

def translate_to_english(text: str, lang: str) -> str:
    """
    Translate arbitrary text to English using Google Translate v2 REST.
    - Respects TRANSLATE_API_KEY env var.
    - Falls back to original text on any failure.
    - Uses in-memory cache to reduce costs/latency.
    """
    global TRANSLATE_HITS, TRANSLATE_MISSES
    
    text = (text or "").strip()
    if not text:
        return text

    # If disabled or already English → return as-is
    if not TRANSLATE_ENABLED or (lang or "").lower() in ("", "en"):
        return text

    cache_key = (lang.lower(), text)
    if cache_key in TRANSLATE_CACHE:
        TRANSLATE_HITS += 1
        return TRANSLATE_CACHE[cache_key]

    TRANSLATE_MISSES += 1

    api_key = os.getenv("TRANSLATE_API_KEY", "").strip()
    if not api_key:
        # No key → graceful fallback
        return text

    endpoint = "https://translation.googleapis.com/language/translate/v2"
    chunks = _chunk_text(text)
    out = []

    try:
        # Use blocking httpx client (safe inside bg thread)
        with httpx.Client(timeout=20) as client:
            for chunk in chunks:
                payload = {
                    "q": chunk,
                    "target": "en",
                    # set 'source' if you trust detected lang; else let Google autodetect
                    "source": lang.lower() if lang else None,
                    "format": "text",
                    "key": api_key,
                }
                # Remove None to avoid API complaints
                payload = {k:v for k,v in payload.items() if v is not None}

                # Simple retry loop
                for attempt in range(3):
                    resp = client.post(endpoint, data=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        tr = (data.get("data", {})
                                   .get("translations", [{}])[0]
                                   .get("translatedText", "")) or ""
                        out.append(tr)
                        break
                    else:
                        time.sleep(0.7)  # backoff
                else:
                    # all retries failed → fallback for this chunk
                    out.append(chunk)
    except Exception:
        # Any exception → fallback
        return text

    english = " ".join(out).strip()
    TRANSLATE_CACHE[cache_key] = english
    return english or text

def build_dual_chunks(full_text: str, lang: str):
    """
    Build two caption tracks: local + English.
    """
    local_chunks = simple_caption_chunks(full_text or "Explaina")
    local_all = " ".join([t for _,_,t in local_chunks])
    english_all = translate_to_english(local_all, lang)
    words = english_all.split()
    step = max(1, len(words)//max(1,len(local_chunks)))
    english_chunks = []
    idx = 0
    for (a,b,_t) in local_chunks:
        part = " ".join(words[idx:idx+step]).strip()
        english_chunks.append((a,b, part if part else " "))
        idx += step
    while len(english_chunks) < len(local_chunks):
        english_chunks.append((local_chunks[len(english_chunks)][0], local_chunks[len(english_chunks)][1], " "))
    return local_chunks, english_chunks[:len(local_chunks)]

def write_dual_srt(local_chunks, english_chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, ((a,b,l_txt),(a2,b2,e_txt)) in enumerate(zip(local_chunks, english_chunks), 1):
            def fmt(t): 
                return time.strftime("%H:%M:%S", time.gmtime(t)) + f",{int((t%1)*1000):03d}"
            f.write(f"{i}\n{fmt(a)} --> {fmt(b)}\n{(l_txt or ' ')}\n{(e_txt or ' ')}\n\n")

def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = "\n".join(line.strip() for line in s.splitlines())
    while "  " in s:
        s = s.replace("  ", " ")
    return s

def to_podcast_text(question: str, answer: str, lang: str = "en") -> str:
    import textwrap
    q = (question or "").strip()
    a = _normalize_text(answer)

    openings = {
        "en": f"You're listening to Explaina. Today's question: {q}.",
        "fr": f"Vous écoutez Explaina. La question du jour : {q}.",
        "ar": f"أنتم تستمعون إلى إكسبلينا. سؤال اليوم: {q}.",
        "es": f"Estás escuchando Explaina. La pregunta de hoy: {q}.",
    }
    closings = {
        "en": "Thanks for listening to Explaina. Ask another question anytime.",
        "fr": "Merci pour votre écoute sur Explaina. Posez une autre question quand vous voulez.",
        "ar": "شكراً لاستماعكم إلى إكسبلينا. اطرح سؤالاً آخر في أي وقت.",
        "es": "Gracias por escuchar Explaina. Puedes hacer otra pregunta cuando quieras.",
    }
    intro = openings.get(lang, openings["en"])
    outro = closings.get(lang, closings["en"])

    wrap = lambda t: "\n".join(textwrap.fill(p.strip(), width=120) for p in t.split("\n") if p.strip())
    if "•" in a: a = a.replace("•", "- ")

    recap = ""
    if len(a.split()) > 120:
        recap = "\n\nIn short, here are the essentials:"

    return f"{intro}\n\n{wrap(a)}{recap}\n\n{outro}".strip()

def make_visual_plan(question: str, answer: str, lang: str = "en") -> dict:
    """
    Generate an intelligent scene plan tuned to the topic, mood, and visuals.
    """
    q = (question or "").lower()
    a = (answer or "").lower()
    text = q + " " + a

    def has(*keys): return any(k in text for k in keys)

    # --- topic detection ---
    if has("rain", "cloud", "storm", "weather", "water", "precipitation", "snow", "thunder"):
        topic, mood, accent = "rain", "calm", "#17c9c0"
        assets = ["clouds", "rain", "puddle"]
        animations = ["drift", "fall"]
        bg = "#0e1834"
        overlay = "/static/fx/clouds.svg"
    elif has("lift", "wing", "airfoil", "aerodynamic", "plane", "fly", "flight", "airplane", "aircraft", "aviation"):
        topic, mood, accent = "flight", "energetic", "#fbbf24"
        assets = ["airfoil", "flow_arrows", "lift_arrow"]
        animations = ["stream", "rise"]
        bg = "#0f214d"
        overlay = "/static/fx/airflow.svg"
    elif has("money", "finance", "stock", "investment", "interest", "compound", "savings", "bank", "profit", "wealth", "invest"):
        topic, mood, accent = "finance", "confident", "#22c55e"
        assets = ["chart", "arrow_up", "coin"]
        animations = ["grow", "pulse"]
        bg = "#0e1a36"
        overlay = "/static/fx/finance.svg"
    else:
        topic, mood, accent = "generic", "neutral", "#17c9c0"
        assets, animations, bg = ["bg_gradient"], ["fade"], "#0f1836"
        overlay = None

    # --- build plan ---
    plan = {
        "topic": topic,
        "mood": mood,
        "theme": {"bg": bg, "accent": accent},
        "overlay": overlay,
        "scenes": [
            {"type": "title", "text": question, "duration": 2.5},
            {
                "type": "main",
                "duration": max(8, min(45, len(a.split()) / 3)),
                "assets": assets,
                "animations": animations,
                "captions": [answer[:140] + ("…" if len(answer) > 140 else "")]
            },
            {"type": "recap", "text": f"In short: {answer.split('.')[0]}", "duration": 3.0},
        ],
        "music": {"mood": mood, "volume": 0.15},
        "watermark": "/static/watermark.png",
        "export": {"preset": "mobile", "format": "mp4"},
    }
    return plan

def bg_compose(job_id: str, *, text: str, audio_url: str, watermark: str):
    try:
        # Read lang and preset from job metadata
        j = job_get(job_id) or {}
        lang = j.get("lang", "en")
        preset_mode = (j.get("preset") or "mobile").lower()
        
        if preset_mode == "hd":
            target_w, target_h = 1920, 1080
            crf = "21"          # higher quality
            abr = "128k"        # audio bitrate
        else:
            target_w, target_h = 960, 540
            crf = "23"          # smaller files, faster
            abr = "96k"
        
        job_update(job_id, status="preparing", percent=5)

        # Resolve inputs
        a_path = "." + audio_url if audio_url.startswith("/") else audio_url
        if not os.path.exists(a_path):
            raise RuntimeError("audio file not found")
        wm_path = ("."+watermark) if watermark.startswith("/") and os.path.exists("."+watermark) \
                  else (watermark if os.path.exists(watermark) else "static/watermark.png")

        vid_id = uuid.uuid4().hex[:12]
        srt_path = MEDIA_DIR / f"{vid_id}.srt"
        out_mp4  = MEDIA_DIR / f"{vid_id}.mp4"
        out_jpg  = MEDIA_DIR / f"{vid_id}.jpg"

        # 1) Build dual-language SRT (≈ fast)
        local_chunks, english_chunks = build_dual_chunks(text or "Explaina", lang=lang)
        write_dual_srt(local_chunks, english_chunks, srt_path)
        job_update(job_id, status="subtitles", percent=20)

        # 2) Probe audio length to set duration (faster than a long color source)
        #    If ffprobe not available, we still proceed with -shortest.
        dur_sec = None
        try:
            p = subprocess.run(
                ["ffprobe","-v","error","-show_entries","format=duration",
                 "-of","default=noprint_wrappers=1:nokey=1", a_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if p.returncode == 0:
                dur_sec = float(p.stdout.strip())
        except: pass

        # Fast check in case cancel landed between queue and start
        job_data = job_get(job_id) or {}
        if job_data.get("status") == "canceled":
            return

        job_update(job_id, status="encoding", percent=40)

        # Build filter graph
        #    - base background at target resolution
        #    - render dual subtitles
        #    - overlay watermark bottom-right
        rtl = lang in {"ar","ur","fa","he"}
        alignment = "Alignment=9" if rtl else "Alignment=2"  # libass alignment codes
        srt_path_str = norm(srt_path)
        style = f"Fontsize=28,PrimaryColour=&H00FFFFFF&,OutlineColour=&H48000000&,BorderStyle=3,Outline=2,Shadow=0,{alignment}"
        vf = (
            f"[0:v]format=yuv420p,"
            f"subtitles={srt_path_str}:force_style='{style}'[bg];"
            f"[bg][2:v]overlay=W-w-20:H-h-20:format=auto,scale={target_w}:{target_h}:flags=lanczos"
        )

        # Duration clamp for background (color source) — keep it just over the audio length
        color_src = f"color=c=0x10244a:s={target_w}x{target_h}:r=30"
        if dur_sec:
            color_src += f":d={max(1, int(dur_sec + 0.5))}"

        # Optimized encode
        #    - ultrafast preset (fastest), CRF varies by preset (21 HD, 23 mobile)
        #    - GOP every 60 frames (2s at 30fps), better for scrubbing & social uploads
        #    - AAC bitrate varies by preset (128k HD, 96k mobile)
        cmd = [
            "ffmpeg",
            "-f","lavfi","-i", color_src,      # color_src built as before with duration
            "-i", a_path,
            "-i", str(wm_path if os.path.exists(str(wm_path)) else "static/watermark.png"),
            "-shortest",
            "-filter_complex", vf,
            "-c:v","libx264","-preset","ultrafast","-crf", crf,
            "-profile:v","baseline","-level","3.1","-pix_fmt","yuv420p",
            "-g","60","-keyint_min","60",
            "-c:a","aac","-b:a", abr,          # variable audio bitrate by preset
            "-movflags","+faststart",
            str(out_mp4), "-y"
        ]

        # Launch (keep your existing Popen/cancel logic if you have it)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        job_register_proc(job_id, proc)

        start_t = time.time()
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            # advance percent up to ~85% during encoding phase
            if dur_sec:
                elapsed = time.time() - start_t
                pcent = 40 + min(45, max(0, int(45 * (elapsed / max(1.0, dur_sec)))))
                job_update(job_id, percent=pcent)
            time.sleep(0.4)

        job_register_proc(job_id, None)
        if proc.returncode != 0:
            # honor cancel vs real error
            job_data = job_get(job_id) or {}
            if job_data.get("status") == "canceled":
                job_update(job_id, percent=100, error="canceled")
                return
            err_tail = (proc.stderr.read() if proc.stderr else "")[-400:]
            raise RuntimeError(f"FFmpeg error: {err_tail}")

        job_update(job_id, status="thumbnail", percent=85)

        # 4) Thumbnail
        run_ffmpeg(["ffmpeg","-ss","00:00:00.5","-i",str(out_mp4),"-frames:v","1","-q:v","3","-strict","unofficial",str(out_jpg),"-y"])

        # 5) Done
        job_update(job_id, status="done", percent=100,
                   result={"video_url": "/"+norm(out_mp4),
                           "thumb_url": "/"+norm(out_jpg)})
        # save to cache
        j = job_get(job_id) or {}
        cache_key = j.get("cache_key","")
        result = j.get("result")
        if cache_key and result:
            with CACHE_LOCK:
                if len(RENDER_CACHE) >= CACHE_MAX:
                    # drop oldest
                    RENDER_CACHE.pop(next(iter(RENDER_CACHE)))
                RENDER_CACHE[cache_key] = result
    except Exception as e:
        job_data = job_get(job_id) or {}
        st = job_data.get("status")
        if st != "canceled":
            job_update(job_id, status="error", error=str(e), percent=100)

# -------------------------------
# Landing & Health Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def landing():
    return HTMLResponse(INDEX_HTML)

# Optional: /index alias if you like
@app.get("/index", response_class=HTMLResponse)
def index_alias():
    return HTMLResponse(INDEX_HTML)

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")

# -------------------------------
# Feedback (in-memory demo; replace with DB in prod)
# -------------------------------
_feedback = []  # in-memory for now
FEEDBACK_TAGS = {"bug","unclear","incorrect","other"}

@app.post("/feedback")
async def feedback(request: Request):
    raw = await request.body()
    verify_hmac(request, raw)

    ip = get_client_ip(request)
    country = get_country_code(request)
    if BLOCKED_COUNTRIES and country in BLOCKED_COUNTRIES:
        raise HTTPException(status_code=451, detail="Access blocked from your region")
    if not rl_country_check(country): raise HTTPException(status_code=429, detail="Region rate limit")
    if not rl_check(ip): raise HTTPException(status_code=429, detail="Too many requests")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except: raise HTTPException(status_code=400, detail="Invalid JSON")

    answer_id = (payload.get("answer_id") or "").strip()
    helpful   = bool(payload.get("helpful"))
    website   = (payload.get("website") or "").strip()  # honeypot
    if website: raise HTTPException(status_code=418, detail="Bot detected")
    if not answer_id: raise HTTPException(status_code=400, detail="answer_id required")

    dkey = dedup_key(ip, payload.get("session_id"), answer_id)
    if recent_vote(dkey):
        # Don't error on duplicate - just return success with duplicate flag
        return {"ok": True, "duplicate": True}

    _feedback.append({
        "ts": time.time(),
        "ip": ip,
        "country": country,
        "answer_id": answer_id,
        "helpful": helpful,
        "question": payload.get("question"),
        "comment": payload.get("comment"),
        "session_id": payload.get("session_id"),
    })
    remember_vote(dkey)
    return {"ok": True, "duplicate": False}

@app.get("/feedback/stats")
def feedback_stats(answer_id: Optional[str] = None):
    rows = [r for r in _feedback if (answer_id is None or r["answer_id"]==answer_id)]
    total = len(rows); up = sum(1 for r in rows if r["helpful"]); rate = round((up/total)*100,2) if total else 0.0
    return {"answer_id": answer_id, "total": total, "helpful_up": up, "helpful_down": total-up, "helpful_rate_pct": rate}

# -------------------------------
# Analytics
# -------------------------------
@app.post("/events")
def log_event(payload: dict = Body(...)):
    rec = {"ts": time.time(), "type": payload.get("type"), "question": payload.get("question"),
           "answer_id": payload.get("answer_id"), "meta": payload.get("meta") or {}}
    with open(EVENTS_LOG, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}

# -------------------------------
# Answer normalizer + local demo
# -------------------------------
@app.post("/local-answer")
def local_answer(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    canned = {
        "explain lift on an airplane.": "Lift is generated when airflow over and under the wing creates a pressure difference...",
        "what are the best ways to reduce stress quickly?": "Try 4-7-8 breathing, a 5-minute walk, progressive muscle relaxation and box breathing.",
        "how can i stop procrastinating and stay motivated?": "Use the 2-minute rule, time boxing, and reward small wins; remove friction.",
    }
    return {"answer": canned.get(q.lower(), f"Demo mode reply for: {q}"),
            "audio_url": "/static/sample.mp3",
            "video_url": "/static/sample.mp4"}

@app.post("/api/answer")
async def proxy_answer(req: Request):
    raw = await req.body()
    payload = json.loads(raw or b"{}")
    question = (payload.get("question") or "").strip()
    lang = (payload.get("lang") or "en").lower()
    want_podcast = bool(payload.get("podcast")) or (payload.get("mode") == "podcast")

    if not ANSWER_API_URL:
        # local demo path
        data = local_answer(payload)
        text_answer = data.get("answer") or data.get("text") or data.get("message") or ""
        audio_url = data.get("audio_url") or "/static/sample.mp3"
        video_url = data.get("video_url") or "/static/sample.mp4"
        plan = make_visual_plan(question, text_answer, lang)
        if want_podcast:
            podcast_text = to_podcast_text(question, text_answer, lang)
            return {"ok": True, "format": "podcast",
                    "podcast_text": podcast_text,
                    "audio_url": audio_url,
                    "video_url": video_url,
                    "video_plan": plan,
                    "raw": data}
        return {"ok": True, "answer": text_answer,
                "audio_url": audio_url,
                "video_url": video_url,
                "video_plan": plan,
                "raw": data}

    # proxy to upstream answerer
    headers = {"Content-Type":"application/json"}
    for h in ("x-timestamp","x-signature"):
        if h in req.headers: headers[h] = req.headers[h]
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(ANSWER_API_URL, content=raw, headers=headers)
    try:
        data = r.json()
    except:
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON")

    text_answer = data.get("answer") or data.get("text") or data.get("message") or ""
    audio_url = data.get("audio_url") or "/static/sample.mp3"
    video_url = data.get("video_url") or "/static/sample.mp4"
    plan = make_visual_plan(question, text_answer, lang)
    if want_podcast:
        podcast_text = to_podcast_text(question, text_answer, lang)
        return {"ok": r.is_success, "format": "podcast",
                "podcast_text": podcast_text,
                "audio_url": audio_url,
                "video_url": video_url,
                "video_plan": plan,
                "raw": data}

    return {"ok": r.is_success, "answer": text_answer,
            "audio_url": audio_url,
            "video_url": video_url,
            "video_plan": plan,
            "raw": data}

@app.post("/api/podcast")
async def api_podcast(payload: dict = Body(...)):
    """
    Input: { question: str, lang?: 'en'|'fr'|'ar'|'es'|..., answer?: str }
    Returns: { ok, podcast_text, audio_url, video_url?, raw }
    """
    question = (payload.get("question") or "").strip()
    lang = (payload.get("lang") or "en").lower()
    if not question:
        raise HTTPException(status_code=400, detail="question required")

    # 1) get a base textual answer (from upstream or local demo)
    if payload.get("answer"):
        base_answer = str(payload["answer"])
        raw = {"source": "client_answer"}
        audio_url = "/static/sample.mp3"
        video_url = ""
    else:
        if not ANSWER_API_URL:
            data = local_answer({"question": question})
            base_answer = data.get("answer") or data.get("text") or data.get("message") or ""
            audio_url   = data.get("audio_url") or "/static/sample.mp3"
            video_url   = data.get("video_url") or ""
            raw = data
        else:
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(ANSWER_API_URL, json={"question": question, "lang": lang})
            if not r.is_success:
                raise HTTPException(status_code=r.status_code, detail=f"Upstream error: {r.status_code}")
            try:
                data = r.json()
            except Exception:
                raise HTTPException(status_code=502, detail="Upstream returned non-JSON")
            base_answer = data.get("answer") or data.get("text") or data.get("message") or ""
            audio_url   = data.get("audio_url") or "/static/sample.mp3"
            video_url   = data.get("video_url") or ""
            raw = data

    # 2) adapt for podcast
    podcast_text = to_podcast_text(question, base_answer, lang)

    # (optional) if you later synthesize TTS from podcast_text, replace audio_url here
    return {
        "ok": True,
        "format": "podcast",
        "podcast_text": podcast_text,
        "audio_url": audio_url,
        "video_url": video_url,
        "raw": raw
    }

# -------------------------------
# Render helpers (subtitles + watermark) and Portrait export
# -------------------------------
@app.post("/api/render/compose")
def compose_video(payload: dict = Body(...)):
    text = (payload.get("text") or "Explaina").strip()
    audio_url = payload.get("audio_url") or "/static/sample.mp3"
    wm = payload.get("watermark") or "/static/watermark.png"
    lang = payload.get("lang") or "en"

    a_path = "." + audio_url if audio_url.startswith("/") else audio_url
    if not os.path.exists(a_path): raise HTTPException(status_code=400, detail="audio file not found")

    vid_id = uuid.uuid4().hex[:12]
    srt_path = MEDIA_DIR / f"{vid_id}.srt"
    out_mp4  = MEDIA_DIR / f"{vid_id}.mp4"
    out_jpg  = MEDIA_DIR / f"{vid_id}.jpg"

    # Build dual-language SRT for non-English
    chunks_local = simple_caption_chunks(text)
    if lang != "en":
        chunks_en = [(a, b, translate_to_english(txt, lang)) for a, b, txt in chunks_local]
        write_dual_srt(chunks_local, chunks_en, srt_path)
    else:
        write_srt(chunks_local, srt_path)

    srt_path_str = norm(srt_path)
    style = "Fontsize=28,PrimaryColour=&H00FFFFFF&,OutlineColour=&H48000000&,BorderStyle=3,Outline=2,Shadow=0"
    filter_complex = f"[0:v]format=yuv420p,subtitles={srt_path_str}:force_style='{style}'[bg];[bg][2:v]overlay=W-w-20:H-h-20"

    cmd = [
        "ffmpeg",
        "-f","lavfi","-i","color=c=0x10244a:s=1280x720:d=300:r=30",
        "-i", a_path,
        "-i", ("."+wm) if wm.startswith("/") and os.path.exists("."+wm) else (wm if os.path.exists(wm) else "static/watermark.png"),
        "-shortest",
        "-filter_complex", filter_complex,
        "-c:v","libx264","-profile:v","baseline","-level","3.1","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","128k","-movflags","+faststart",
        str(out_mp4),"-y"
    ]
    run_ffmpeg(cmd)
    run_ffmpeg(["ffmpeg","-ss","00:00:01","-i",str(out_mp4),"-frames:v","1","-q:v","3",str(out_jpg),"-y"])
    return {"ok": True, "video_url": "/" + norm(out_mp4), "thumb_url": "/" + norm(out_jpg)}

@app.post("/api/render/start")
async def render_start(payload: dict = Body(...)):
    # Generate the visual plan automatically
    text = (payload.get("text") or "Explaina").strip()
    question = (payload.get("question") or text).strip()
    plan = make_visual_plan(question, text, payload.get("lang","en"))
    
    payload["video_plan"] = plan
    
    # Use the module constant so behavior matches /api/progress
    target = RENDER_API_URL
    if target:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(f"{target.rstrip('/')}/api/render/start", json=payload)
            return r.json()
    
    # Otherwise, run locally (fallback)
    audio_url = payload.get("audio_url") or "/static/sample.mp3"
    watermark = payload.get("watermark") or "/static/watermark.png"
    lang = (payload.get("lang") or "en").lower()
    preset = (payload.get("preset") or "mobile").lower()
    
    cache_key = hashlib.sha1(f"{text}|{audio_url}|{watermark}|{lang}|{preset}".encode()).hexdigest()
    with CACHE_LOCK:
        if cache_key in RENDER_CACHE:
            return {"ok": True, "cached": True, "result": RENDER_CACHE[cache_key]}

    job_id = job_create({"kind":"compose","cache_key":cache_key,"lang":lang,"preset":preset})
    JOB_EXECUTOR.submit(bg_compose, job_id, text=text, audio_url=audio_url, watermark=watermark)
    return {"ok": True, "job_id": job_id}

@app.get("/api/progress")
async def render_progress(job_id: str):
    # If RENDER_API_URL is set, proxy to worker microservice
    if RENDER_API_URL:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{RENDER_API_URL}/api/progress?job_id={job_id}")
            data = resp.json()
            # Convert relative worker URLs to absolute URLs
            if data.get("ok") and data.get("result"):
                result = data["result"]
                worker_base = RENDER_API_URL.rstrip("/")
                if result.get("video_url") and result["video_url"].startswith("/"):
                    result["video_url"] = worker_base + result["video_url"]
                if result.get("thumb_url") and result["thumb_url"].startswith("/"):
                    result["thumb_url"] = worker_base + result["thumb_url"]
            return data
    
    # Otherwise, check local job queue
    jobs_cleanup()
    job = job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "ok": True,
        "status": job["status"],
        "percent": job["percent"],
        "result": job.get("result"),
        "error": job.get("error"),
    }

@app.post("/api/cancel")
async def cancel_render(payload: dict = Body(...)):
    """
    Input: { "job_id": "..." }
    Output: { "ok": true, "canceled": true/false }
    """
    job_id = (payload.get("job_id") or "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id required")

    # If RENDER_API_URL is set, proxy to worker microservice
    if RENDER_API_URL:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{RENDER_API_URL}/api/cancel", json={"job_id": job_id})
            return resp.json()
    
    # Otherwise, cancel local job
    job_update(job_id, status="canceled")
    canceled = job_cancel(job_id)
    return {"ok": True, "canceled": bool(canceled)}

@app.post("/api/export/portrait")
def export_portrait(payload: dict = Body(...)):
    vurl = payload.get("video_url")
    if not vurl: raise HTTPException(status_code=400, detail="video_url required")
    vpath = Path("." + vurl if vurl.startswith("/") else vurl)
    if not vpath.exists(): raise HTTPException(status_code=400, detail="video not found")
    out = vpath.with_stem(vpath.stem + "-portrait")
    cmd = [
        "ffmpeg","-i",str(vpath),
        "-filter_complex",
        "[0:v]scale=1080:-2:flags=lanczos,setsar=1[v];"
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,boxblur=20:5,crop=1080:1920,setsar=1[bg];"
        "[bg][v]overlay=(W-w)/2:(H-h)/2",
        "-c:v","libx264","-profile:v","baseline","-level","3.1","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","128k","-movflags","+faststart",
        str(out),"-y"
    ]
    run_ffmpeg(cmd)
    return {"ok": True, "portrait_url": "/" + norm(out)}

# -------------------------------
# YouTube stub
# -------------------------------
@app.post("/api/youtube/publish")
def youtube_publish(payload: dict = Body(None)):
    vurl = (payload or {}).get("video_url")
    if not vurl: raise HTTPException(status_code=400, detail="video_url required")
    return {"ok": True, "videoId": "YT_DEMO_" + str(int(time.time()))}

# -------------------------------
# Backup creation + admin listing
# -------------------------------
def zip_project(out_path: Path, include_media_mp4: bool = False):
    include_exts = {".py", ".txt", ".md", ".json", ".png", ".jpg", ".jpeg", ".svg", ".mp3", ".srt"}
    if include_media_mp4:
        include_exts.add(".mp4")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # code/config
        for p in Path(".").iterdir():
            if p.name in {".git","__pycache__","backups"}: continue
            if p.is_file() and p.suffix.lower() in {".py",".txt",".md"}:
                z.write(p, p.name)
        # static
        for p in STATIC_DIR.rglob("*"):
            if p.is_dir(): continue
            if p.suffix.lower() in include_exts:
                z.write(p, p.as_posix())
        manifest = {
            "service":"explaina-backup",
            "created_utc": datetime.utcnow().isoformat()+"Z",
            "include_media_mp4": include_media_mp4,
        }
        z.writestr("BACKUP-MANIFEST.json", json.dumps(manifest, indent=2))

@app.get("/admin/backup")
def backup(include_media_mp4: bool = False, token: str = "", name: str = ""):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # serve existing file
    if name:
        path = BACKUP_DIR / name
        if path.exists() and path.is_file():
            return FileResponse(path.as_posix(), filename=path.name, media_type="application/zip")
        raise HTTPException(status_code=404, detail="File not found")
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = BACKUP_DIR / f"explaina-backup-{ts}{'-full' if include_media_mp4 else ''}.zip"
    zip_project(out, include_media_mp4=include_media_mp4)
    return FileResponse(out.as_posix(), filename=out.name, media_type="application/zip")

@app.get("/admin/backups", response_class=HTMLResponse)
def list_backups(token: str = Query(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        return HTMLResponse("<h3>Unauthorized</h3>", status_code=401)
    rows = []
    if BACKUP_DIR.exists():
        for p in sorted(BACKUP_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_file() and p.suffix.lower() == ".zip":
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(p.stat().st_mtime))
                size_mb = p.stat().st_size / (1024*1024)
                rows.append((p.name, ts, f"{size_mb:.2f} MB"))
    html_head = """
    <!doctype html><meta charset="utf-8">
    <title>Explaina – Backups</title>
    <style>
      :root{--font-latin:'Noto Sans',system-ui,-apple-system,'Segoe UI',sans-serif;--font-arabic:'Noto Sans Arabic',sans-serif;--font-cjk:'Noto Sans CJK SC','Noto Sans CJK JP',sans-serif;--font-devanagari:'Noto Sans Devanagari',sans-serif}
      body{font-family:var(--font-latin);background:#0f1836;color:#eaf2ff;margin:24px}
      [lang="ar"],[lang="ur"],[lang="he"],[lang="fa"]{font-family:var(--font-arabic);direction:rtl;text-align:right}
      [lang="zh"],[lang="ja"],[lang="ko"]{font-family:var(--font-cjk)}
      [lang="hi"],[lang="bn"]{font-family:var(--font-devanagari)}
      a{color:#17c9c0;text-decoration:none} a:hover{text-decoration:underline}
      table{border-collapse:collapse;width:100%;max-width:900px;background:#13214a;border:1px solid #27407a;border-radius:12px;overflow:hidden}
      th,td{padding:10px 12px;border-bottom:1px solid #27407a}
      th{background:#0f1b3f;text-align:left}
      tr:last-child td{border-bottom:none}
      .row{display:flex;gap:8px;margin-bottom:12px;align-items:center}
      .btn{display:inline-block;padding:8px 12px;border-radius:8px;border:1px solid #27407a;background:#152a5a;color:#eaf2ff}
      .btn:hover{filter:brightness(1.05)}
      .note{opacity:.8;font-size:14px}
    </style>
    """
    html_top = f"""
    <h1>Explaina – Backups</h1>
    <div class="row">
      <a class="btn" href="/admin/jobs?{'token='+token if token else ''}">Jobs</a>
      <a class="btn" href="/admin/backup?{'token='+token if token else ''}">Create new backup</a>
      <a class="btn" href="/admin/backup?include_media_mp4=true&{'token='+token if token else ''}">Create full (with MP4)</a>
    </div>
    <div class="note">Tip: right-click a filename to copy its link. All links require the same token.</div>
    <table>
      <tr><th>Filename</th><th>Created (UTC)</th><th>Size</th><th>Actions</th></tr>
    """
    html_rows = ""
    for name, ts, size in rows:
        q = f"token={token}&file={name}" if token else f"file={name}"
        dl = f"/admin/backups/download?{q}"
        rm = f"/admin/backups/delete?{q}"
        html_rows += f"""
        <tr>
          <td><a href="{dl}">{name}</a></td>
          <td>{ts}</td>
          <td>{size}</td>
          <td>
            <a class="btn" href="{dl}">Download</a>
            <a class="btn" href="{rm}" onclick="return confirm('Delete {name}?')">Delete</a>
          </td>
        </tr>"""
    if not rows:
        html_rows = '<tr><td colspan="4">No backups yet. Click “Create new backup”.</td></tr>'
    html_tail = "</table>"
    return HTMLResponse(html_head + html_top + html_rows + html_tail)

@app.get("/admin/backups/download")
def download_backup(file: str, token: str = Query(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    path = BACKUP_DIR / file
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path.as_posix(), filename=path.name, media_type="application/zip")

@app.get("/admin/backups/delete")
def delete_backup(file: str, token: str = Query(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    path = BACKUP_DIR / file
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        os.remove(path)
        return HTMLResponse(f"<script>location.href='/admin/backups?token={token}'</script>")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/dashboard", response_class=HTMLResponse)
def dashboard(token: str = Query(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        return HTMLResponse("<h3>Unauthorized</h3>", status_code=401)

    # Worker /health probe
    wrkr = {"status":"offline"}
    try:
        wurl = RENDER_API_URL.rstrip("/")
        if wurl:
            with httpx.Client(timeout=6) as c:
                r = c.get(f"{wurl}/health")
                if r.status_code == 200:
                    wrkr = r.json()
    except Exception:
        pass

    # Probe worker avatar status
    avatar_stat = {"status":"offline"}
    try:
        if RENDER_API_URL:
            with httpx.Client(timeout=6) as c:
                r = c.get(f"{RENDER_API_URL.rstrip('/')}/admin/avatar/status")
                if r.status_code == 200:
                    avatar_stat = r.json()
    except Exception:
        pass

    gcs_val = wrkr.get("gcs")
    gcs = gcs_val if isinstance(gcs_val, dict) else {}

    # ---- Gather events ----
    total_asks = total_answers = total_shares = total_dl_mp3 = total_dl_mp4 = 0
    try:
        if EVENTS_LOG.exists():
            with open(EVENTS_LOG,"r",encoding="utf-8") as f:
                for line in f:
                    try: e = json.loads(line)
                    except: continue
                    t = e.get("type")
                    if t=="ask": total_asks+=1
                    elif t=="answer_ready": total_answers+=1
                    elif t=="share_try": total_shares+=1
                    elif t=="download_podcast": total_dl_mp3+=1
                    elif t=="download_video": total_dl_mp4+=1
    except: pass

    # ---- Feedback ----
    rows = _feedback
    total_votes = len(rows)
    helpful_up = sum(1 for r in rows if r["helpful"])
    helpful_rate = round(helpful_up/total_votes*100,1) if total_votes else 0.0

    # ---- Revenue model (tweak here or via env vars) ----
    CPM = float(os.getenv("EST_CPM","4"))   # $4 default
    PREMIUM_PCT = float(os.getenv("EST_PREMIUM_PCT","0.02")) # 2%
    PREMIUM_ARPU = float(os.getenv("EST_PREMIUM_ARPU","5")) # $5
    monthly_views = total_answers  # rough proxy
    ad_rev = (monthly_views/1000.0)*CPM
    prem_rev = monthly_views*PREMIUM_PCT*PREMIUM_ARPU/30.0  # daily to monthly scale
    total_rev = ad_rev+prem_rev

    # Translation status
    has_key = bool(os.getenv("TRANSLATE_API_KEY", "").strip())
    tr_enabled = TRANSLATE_ENABLED
    tr_hits = TRANSLATE_HITS
    tr_misses = TRANSLATE_MISSES
    tr_cache_size = len(TRANSLATE_CACHE)

    # ---- Profitability probability heuristic ----
    prob = "Low"
    if monthly_views>=2_500_000 and helpful_rate>=80: prob="High"
    elif monthly_views>=500_000 and helpful_rate>=75: prob="Medium"

    # ---- HTML ----
    html = f"""
    <!doctype html><meta charset="utf-8">
    <title>Explaina – Profitability Dashboard</title>
    <style>
      :root{{--font-latin:'Noto Sans',system-ui,-apple-system,'Segoe UI',sans-serif;--font-arabic:'Noto Sans Arabic',sans-serif;--font-cjk:'Noto Sans CJK SC','Noto Sans CJK JP',sans-serif}}
      body{{font-family:var(--font-latin);background:#0f1836;color:#eaf2ff;margin:24px}}
      [lang="ar"],[lang="ur"],[lang="he"],[lang="fa"]{{font-family:var(--font-arabic);direction:rtl;text-align:right}}
      [lang="zh"],[lang="ja"],[lang="ko"]{{font-family:var(--font-cjk)}}
      h1{{margin-bottom:6px}}
      .card{{background:#13214a;border:1px solid #27407a;border-radius:12px;padding:16px;margin:12px 0;max-width:800px}}
      .num{{font-size:22px;font-weight:700}}
      table{{border-collapse:collapse;width:100%}}th,td{{padding:6px 10px;border-bottom:1px solid #27407a}}
      th{{text-align:left}}
      .low{{color:#ff5a79}}.med{{color:#facc15}}.high{{color:#22c55e}}
      .row{{display:flex;gap:8px;margin-bottom:12px;align-items:center}}
      .btn{{display:inline-block;padding:8px 12px;border-radius:8px;border:1px solid #27407a;background:#152a5a;color:#eaf2ff;text-decoration:none}}
      .btn:hover{{filter:brightness(1.05)}}
    </style>
    <h1>Explaina – Profitability Dashboard</h1>
    <div class="row">
      <a class="btn" href="/admin/jobs?{'token='+token if token else ''}">Jobs</a>
    </div>

    <div class="card">
      <h2>Traffic</h2>
      <table>
        <tr><th>Total Asks</th><td class="num">{total_asks}</td></tr>
        <tr><th>Answers Rendered</th><td class="num">{total_answers}</td></tr>
        <tr><th>Shares</th><td class="num">{total_shares}</td></tr>
        <tr><th>Podcast downloads</th><td class="num">{total_dl_mp3}</td></tr>
        <tr><th>Video downloads</th><td class="num">{total_dl_mp4}</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Quality</h2>
      <table>
        <tr><th>Total Feedback Votes</th><td class="num">{total_votes}</td></tr>
        <tr><th>Helpful Rate</th><td class="num">{helpful_rate}%</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Revenue Model (estimates)</h2>
      <table>
        <tr><th>Monthly Views (≈ answers)</th><td class="num">{monthly_views:,}</td></tr>
        <tr><th>Ad CPM</th><td>${CPM:.2f}</td></tr>
        <tr><th>Premium % of users</th><td>{PREMIUM_PCT*100:.1f}%</td></tr>
        <tr><th>Premium ARPU</th><td>${PREMIUM_ARPU:.2f}</td></tr>
        <tr><th>Ad Revenue</th><td class="num">${ad_rev:,.0f}</td></tr>
        <tr><th>Premium Revenue</th><td class="num">${prem_rev:,.0f}</td></tr>
        <tr><th><b>Total Revenue</b></th><td class="num">${total_rev:,.0f}</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Probability of Global Success</h2>
      <p class="{ 'low' if prob=='Low' else 'med' if prob=='Medium' else 'high'}"><b>{prob}</b></p>
      <p class="note">Heuristic: based on monthly views + helpfulness. <br>
      Low &lt;500k views, Medium ~0.5–2.5M, High ≥2.5M and helpful≥80%.</p>
    </div>

    <div class="card">
      <h2>Translation (Subtitles)</h2>
      <table>
        <tr><th>API Key Detected</th><td class="num">{'Yes' if has_key else 'No'}</td></tr>
        <tr><th>Enabled</th><td class="num">{'Yes' if tr_enabled else 'No'}</td></tr>
        <tr><th>Cache</th><td class="num">{tr_cache_size} entries</td></tr>
        <tr><th>Cache Hits</th><td class="num">{tr_hits}</td></tr>
        <tr><th>Cache Misses</th><td class="num">{tr_misses}</td></tr>
      </table>
      <div style="margin-top:8px">
        <form method="post" action="/admin/translate/toggle" style="display:inline">
          <input type="hidden" name="token" value="{token}">
          <button class="btn">{'Disable' if tr_enabled else 'Enable'} Translation</button>
        </form>
        <form method="post" action="/admin/translate/clear" style="display:inline;margin-left:8px">
          <input type="hidden" name="token" value="{token}">
          <button class="btn">Clear Cache</button>
        </form>
      </div>
    </div>

    <div class="card">
      <h2>Render Worker & GCS</h2>
      <table>
        <tr><th>Worker Status</th><td class="num">{wrkr.get('status','offline')}</td></tr>
        <tr><th>Workers</th><td class="num">{wrkr.get('workers','—')}</td></tr>
        <tr><th>Cache Entries</th><td class="num">{wrkr.get('cache','—')}</td></tr>
        <tr><th>GCS Bucket</th><td class="num">{gcs.get('bucket','—')}</td></tr>
        <tr><th>GCS Folder</th><td class="num">{gcs.get('folder','—')}</td></tr>
        <tr><th>Public Objects</th><td class="num">{'Yes' if gcs.get('public') else 'No'}</td></tr>
        <tr><th>Signed URLs</th><td class="num">{'Yes' if gcs.get('signed_urls') else 'No'}</td></tr>
        <tr><th>Last Upload (UTC)</th><td class="num">{gcs.get('last_upload_utc') or '—'}</td></tr>
        <tr><th>Worker Health URL</th><td><a href="{RENDER_API_URL.rstrip('/')}/health" target="_blank">open</a></td></tr>
        <tr><th>GCS Objects</th><td><a href="{RENDER_API_URL.rstrip('/')}/admin/gcs/list" target="_blank">latest 10</a></td></tr>
      </table>
      <div style="margin-top:8px">
        <form method="post" action="/admin/gcs/test" style="display:inline">
          <input type="hidden" name="token" value="{token}">
          <button class="btn">Test GCS Upload</button>
        </form>
      </div>
    </div>
    """
    
    # Avatar settings card
    prov = avatar_stat.get("provider","—")
    ab_on = avatar_stat.get("ab_enabled", False)
    ab_rt = avatar_stat.get("ab_ratio", 50)

    html += f"""
    <div class="card">
      <h2>Avatar Settings (D-ID / Synthesia)</h2>
      <table>
        <tr><th>Current Provider</th><td class="num">{prov}</td></tr>
        <tr><th>A/B Enabled</th><td class="num">{'Yes' if ab_on else 'No'}</td></tr>
        <tr><th>A/B Ratio (D-ID %)</th><td class="num">{ab_rt}%</td></tr>
      </table>

      <div style="margin-top:8px">
        <form method="post" action="/admin/avatar/set" style="display:inline">
          <input type="hidden" name="token" value="{token}">
          <select name="provider">
            <option value="none" {"selected" if prov=="none" else ""}>None</option>
            <option value="did" {"selected" if prov=="did" else ""}>D-ID</option>
            <option value="synthesia" {"selected" if prov=="synthesia" else ""}>Synthesia</option>
          </select>
          <button class="btn">Set Provider</button>
        </form>

        <form method="post" action="/admin/avatar/ab" style="display:inline; margin-left:10px">
          <input type="hidden" name="token" value="{token}">
          <label>
            <select name="enable">
              <option value="off" {"selected" if not ab_on else ""}>AB Off</option>
              <option value="on" {"selected" if ab_on else ""}>AB On</option>
            </select>
          </label>
          <label style="margin-left:6px">
            D-ID %:
            <input type="number" name="ratio" min="0" max="100" value="{ab_rt}" style="width:70px">
          </label>
          <button class="btn">Update A/B</button>
        </form>
      </div>
    </div>
    """
    
    return HTMLResponse(html)

@app.get("/admin/jobs", response_class=HTMLResponse)
async def admin_jobs(token: str = Query(default="")):
    """Simple jobs monitor (lists queued/running/done; supports cancel)."""
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        return HTMLResponse("<h3>Unauthorized</h3>", status_code=401)

    # Fetch worker metrics if using worker microservice
    worker_metrics = None
    target = os.getenv("RENDER_API_URL", "")
    if target:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{target}/api/metrics")
                worker_metrics = r.json()
        except:
            pass
    
    # Get local metrics as fallback
    if not worker_metrics:
        with JOBS_LOCK:
            queued = sum(1 for j in JOBS.values() if j.get("status") == "queued")
            running = sum(1 for j in JOBS.values() if j.get("status") not in {"queued", "done", "error", "canceled"})
            durations = []
            for j in JOBS.values():
                if "start_time" in j and "end_time" in j:
                    duration = j["end_time"] - j["start_time"]
                    durations.append(duration)
            avg_duration = sum(durations) / len(durations) if durations else 0
            worker_metrics = {
                "total_jobs": len(JOBS),
                "queued": queued,
                "running": running,
                "avg_duration_sec": round(avg_duration, 2),
                "completed_count": len(durations)
            }

    # snapshot to avoid holding the lock while rendering HTML
    with JOBS_LOCK:
        items = [(jid, dict(j)) for jid, j in JOBS.items()]
    now = time.time()

    def fmt_age(ts):
        if not ts: return "—"
        s = int(now - ts)
        if s < 60: return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60: return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"

    rows = []
    for jid, j in sorted(items, key=lambda kv: kv[1].get("updated", 0), reverse=True):
        st   = j.get("status", "—")
        pct  = j.get("percent", 0)
        kind = j.get("kind", "compose")
        preset_mode = j.get("preset", "mobile")
        age  = fmt_age(j.get("updated", j.get("created")))
        err  = (j.get("error") or "")[:140]
        res  = j.get("result") or {}
        vurl = res.get("video_url", "")
        turl = res.get("thumb_url", "")

        # status pill
        cls = "pill gray"
        if st in ("queued","preparing","subtitles"): cls="pill blue"
        if st in ("encoding","thumbnail"): cls="pill amber"
        if st == "done": cls="pill green"
        if st in ("error","canceled"): cls="pill red"

        # cancel action (only for queued/encoding/preparing/subtitles/thumbnail)
        can_cancel = st in {"queued","preparing","subtitles","encoding","thumbnail"}
        cancel_btn = f"""
          <form method="post" action="/admin/jobs/cancel" style="display:inline">
            <input type="hidden" name="job_id" value="{jid}">
            {'<input type="hidden" name="token" value="'+token+'">' if token else ''}
            <button class="btn danger" {'disabled' if not can_cancel else ''}>Cancel</button>
          </form>
        """

        video_link = f'<a href="{vurl}" target="_blank">video</a>' if vurl else "—"
        thumb_link = f'<a href="{turl}" target="_blank">thumb</a>' if turl else "—"

        rows.append(f"""
          <tr>
            <td class="mono">{jid}</td>
            <td>{kind} / {preset_mode}</td>
            <td><span class="{cls}">{st}</span></td>
            <td>{pct}%</td>
            <td>{age}</td>
            <td>{video_link} · {thumb_link}</td>
            <td class="err">{err}</td>
            <td>{cancel_btn}</td>
          </tr>
        """)

    html = f"""
    <!doctype html><meta charset="utf-8">
    <title>Explaina – Jobs</title>
    <style>
      :root {{ --bg:#0f1836; --card:#13214a; --text:#eaf2ff; --muted:#b8c4d9; --border:#27407a;
               --green:#22c55e; --red:#ff5a79; --amber:#f59e0b; --blue:#38bdf8; }}
      body {{ margin:24px; font-family:'Noto Sans',system-ui,-apple-system,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); }}
      [lang="ar"],[lang="ur"],[lang="he"],[lang="fa"] {{ font-family:'Noto Sans Arabic',sans-serif; direction:rtl; text-align:right }}
      [lang="zh"],[lang="ja"],[lang="ko"] {{ font-family:'Noto Sans CJK SC','Noto Sans CJK JP',sans-serif }}
      h1 {{ margin:0 0 8px 0 }}
      .row {{ display:flex; gap:8px; align-items:center; margin-bottom:12px; }}
      .btn {{ padding:8px 12px; border-radius:8px; border:1px solid var(--border); background:#152a5a; color:var(--text); cursor:pointer; }}
      .btn:hover {{ filter:brightness(1.06) }}
      .btn.danger {{ border-color: rgba(255,90,121,.4); }}
      .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px; max-width:1100px; }}
      table {{ border-collapse:collapse; width:100%; }}
      th,td {{ padding:8px 10px; border-bottom:1px solid var(--border); vertical-align:top; }}
      th {{ text-align:left }}
      tr:last-child td {{ border-bottom:none }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px }}
      .err {{ color:var(--muted); font-size:12px }}
      .pill {{ padding:4px 8px; border-radius:999px; font-size:12px; border:1px solid var(--border) }}
      .pill.green {{ background: rgba(34,197,94,.15); border-color: rgba(34,197,94,.35); }}
      .pill.red   {{ background: rgba(255,90,121,.15); border-color: rgba(255,90,121,.35); }}
      .pill.amber {{ background: rgba(245,158,11,.15); border-color: rgba(245,158,11,.35); }}
      .pill.blue  {{ background: rgba(56,189,248,.15); border-color: rgba(56,189,248,.35); }}
      .pill.gray  {{ background: rgba(148,163,184,.15); border-color: rgba(148,163,184,.35); }}
      .muted {{ color:var(--muted); font-size:13px }}
    </style>

    <h1>Jobs</h1>
    <div class="row">
      <a class="btn" href="/admin/jobs?{'token='+token if token else ''}">Refresh</a>
      <form method="post" action="/admin/killall" style="display:inline">
        <input type="hidden" name="token" value="{token}">
        <button class="btn danger" title="Cancel all running jobs">Kill All</button>
      </form>
      <span class="muted">Auto-refreshes every 5s</span>
      <span style="margin-left:auto" class="muted">
        Showing {len(items)} jobs · Finished jobs older than 24h are auto-pruned
      </span>
    </div>

    <div class="card" style="margin-bottom:12px">
      <h2 style="margin:0 0 12px 0; font-size:16px">Queue Health</h2>
      <table>
        <tr><th>Total Jobs</th><td>{worker_metrics['total_jobs']}</td></tr>
        <tr><th>Queued</th><td>{worker_metrics['queued']}</td></tr>
        <tr><th>Running</th><td>{worker_metrics['running']}</td></tr>
        <tr><th>Avg Duration</th><td>{worker_metrics['avg_duration_sec']}s (from {worker_metrics['completed_count']} completed)</td></tr>
      </table>
    </div>

    <div class="card">
      <table>
        <tr>
          <th>Job ID</th><th>Kind</th><th>Status</th><th>%</th><th>Age</th><th>Result</th><th>Error</th><th>Action</th>
        </tr>
        {''.join(rows) if rows else '<tr><td colspan="8">No jobs yet.</td></tr>'}
      </table>
    </div>

    <script>
      setTimeout(function(){{ location.href = location.pathname + location.search; }}, 5000);
    </script>
    """
    return HTMLResponse(html)

@app.post("/admin/jobs/cancel")
def admin_jobs_cancel(job_id: str = Body(...), token: str = Body(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # mark and signal cancel
    job_update(job_id, status="canceled")
    _ = job_cancel(job_id)
    # bounce back to list
    q = f"?token={token}" if token else ""
    return HTMLResponse(f"<script>location.href='/admin/jobs{q}'</script>")

@app.post("/admin/killall")
def admin_killall(token: str = Form(default="")):
    """Cancel all in-flight or queued jobs (token-protected)."""
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    killed = 0
    # mark all jobs canceled and terminate any process
    with JOBS_LOCK:
        ids = list(JOBS.keys())
    for jid in ids:
        job = job_get(jid)
        if not job:
            continue
        st = job.get("status")
        if st in {"queued","preparing","subtitles","encoding","thumbnail"}:
            job_update(jid, status="canceled")
            if job_cancel(jid):
                killed += 1

    # opportunistic cleanup
    jobs_cleanup()
    # bounce back to jobs list
    q = f"?token={token}" if token else ""
    return HTMLResponse(f"<script>location.href='/admin/jobs{q}'</script>")

@app.post("/admin/translate/toggle")
def admin_translate_toggle(token: str = Form(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    global TRANSLATE_ENABLED
    TRANSLATE_ENABLED = not TRANSLATE_ENABLED
    return HTMLResponse(f"<script>location.href='/admin/dashboard?token={token}'</script>")

@app.post("/admin/translate/clear")
def admin_translate_clear(token: str = Form(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    TRANSLATE_CACHE.clear()
    # optional: reset counters as well
    global TRANSLATE_HITS, TRANSLATE_MISSES
    TRANSLATE_HITS = 0
    TRANSLATE_MISSES = 0
    return HTMLResponse(f"<script>location.href='/admin/dashboard?token={token}'</script>")

@app.post("/admin/gcs/test")
def admin_gcs_test(token: str = Form(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    wurl = RENDER_API_URL.rstrip("/")
    if not wurl:
        return HTMLResponse("<script>alert('RENDER_API_URL not set');history.back()</script>")
    try:
        with httpx.Client(timeout=15) as c:
            r = c.post(f"{wurl}/admin/gcs/test")
            j = r.json()
        if j.get("ok"):
            url = j.get("url","")
            return HTMLResponse(f"<script>alert('OK: {url}');location.href='/admin/dashboard?token={token}'</script>")
        else:
            err = j.get("error","unknown")
            return HTMLResponse(f"<script>alert('GCS test failed: {err}');location.href='/admin/dashboard?token={token}'</script>")
    except Exception as e:
        return HTMLResponse(f"<script>alert('GCS test error: {e}');location.href='/admin/dashboard?token={token}'</script>")

@app.post("/admin/avatar/set")
def admin_avatar_set(provider: str = Form(...), token: str = Form(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not RENDER_API_URL:
        return HTMLResponse("<script>alert('RENDER_API_URL not set');history.back()</script>")
    try:
        with httpx.Client(timeout=10) as c:
            r = c.post(f"{RENDER_API_URL.rstrip('/')}/admin/avatar/set", data={"provider": provider})
            j = r.json()
        msg = f"Provider set to {j.get('provider','?')}; A/B = {'on' if j.get('ab_enabled') else 'off'}"
        return HTMLResponse(f"<script>alert('{msg}');location.href='/admin/dashboard?token={token}'</script>")
    except Exception as e:
        return HTMLResponse(f"<script>alert('Failed: {e}');history.back()</script>")

@app.post("/admin/avatar/ab")
def admin_avatar_ab(enable: str = Form(...), ratio: int = Form(50), token: str = Form(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not RENDER_API_URL:
        return HTMLResponse("<script>alert('RENDER_API_URL not set');history.back()</script>")
    try:
        with httpx.Client(timeout=10) as c:
            r = c.post(f"{RENDER_API_URL.rstrip('/')}/admin/avatar/ab", data={"enable": enable, "ratio": str(ratio)})
            j = r.json()
        msg = f"A/B={'on' if j.get('ab_enabled') else 'off'}; D-ID%={j.get('ab_ratio')}"
        return HTMLResponse(f"<script>alert('{msg}');location.href='/admin/dashboard?token={token}'</script>")
    except Exception as e:
        return HTMLResponse(f"<script>alert('Failed: {e}');history.back()</script>")

@app.get("/admin/moderation", response_class=HTMLResponse)
def moderation(token: str = Query(default=""), tag: str = Query(default=""), q: str = Query(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        return HTMLResponse("<h3>Unauthorized</h3>", status_code=401)

    # filter 👎 or with comments
    items = []
    for r in _feedback:
        if (not r.get("helpful")) or (r.get("comment")):
            if tag and r.get("mod_tag") != tag: 
                continue
            if q and q.lower() not in (r.get("question") or "").lower():
                continue
            items.append(r)

    rows = []
    for i, r in enumerate(sorted(items, key=lambda x: x["ts"], reverse=True), 1):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r["ts"]))
        mod_tag = r.get("mod_tag","")
        rows.append(f"""
          <tr>
            <td>{i}</td>
            <td class="mono">{r.get('answer_id','')}</td>
            <td>{'👍' if r.get('helpful') else '👎'}</td>
            <td>{(r.get('question') or '')[:120]}</td>
            <td>{(r.get('comment') or '')[:160]}</td>
            <td>{ts}</td>
            <td>{mod_tag or '—'}</td>
            <td>
              <form method="post" action="/admin/moderation/tag">
                <input type="hidden" name="token" value="{token}">
                <input type="hidden" name="answer_id" value="{r.get('answer_id','')}">
                <select name="tag">
                  <option value="">—</option>
                  {''.join([f'<option {"selected" if t==mod_tag else ""}>{t}</option>' for t in sorted(FEEDBACK_TAGS)])}
                </select>
                <button class="btn">Save</button>
              </form>
            </td>
          </tr>
        """)

    html = f"""
    <!doctype html><meta charset="utf-8">
    <title>Explaina – Moderation</title>
    <style>
      body{{font-family:'Noto Sans',system-ui,-apple-system,'Segoe UI',sans-serif;background:#0f1836;color:#eaf2ff;margin:24px}}
      [lang="ar"],[lang="ur"],[lang="he"],[lang="fa"]{{font-family:'Noto Sans Arabic',sans-serif;direction:rtl}}
      a{{color:#17c9c0}} .btn{{padding:6px 10px;border:1px solid #27407a;border-radius:8px;background:#152a5a;color:#eaf2ff}}
      table{{border-collapse:collapse;width:100%}} th,td{{padding:8px 10px;border-bottom:1px solid #27407a;vertical-align:top}}
      th{{text-align:left}} .mono{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}}
      .row{{display:flex;gap:8px;align-items:center;margin-bottom:12px}}
      select{{background:#152a5a;color:#eaf2ff;border:1px solid #27407a;border-radius:6px;padding:4px}}
      input[type=text]{{background:#152a5a;color:#eaf2ff;border:1px solid #27407a;border-radius:6px;padding:6px}}
    </style>
    <h1>Moderation</h1>
    <div class="row">
      <form method="get" action="/admin/moderation">
        {'<input type="hidden" name="token" value="'+token+'">' if token else ''}
        Filter tag:
        <select name="tag"><option value="">(all)</option>
          {''.join([f'<option {"selected" if t==tag else ""}>{t}</option>' for t in sorted(FEEDBACK_TAGS)])}
        </select>
        Search:
        <input type="text" name="q" value="{q}">
        <button class="btn">Apply</button>
      </form>
      <a class="btn" href="/admin/export.csv?token={token}">Export CSV</a>
    </div>
    <table>
      <tr><th>#</th><th>Answer ID</th><th>Vote</th><th>Question</th><th>Comment</th><th>UTC</th><th>Tag</th><th>Action</th></tr>
      { ''.join(rows) if rows else '<tr><td colspan="8">No items.</td></tr>' }
    </table>
    """
    return HTMLResponse(html)

@app.post("/admin/moderation/tag")
def set_mod_tag(answer_id: str = Form(...), tag: str = Form(""), token: str = Form("")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if tag and tag not in FEEDBACK_TAGS:
        raise HTTPException(status_code=400, detail="Unknown tag")
    for r in _feedback:
        if r.get("answer_id") == answer_id:
            r["mod_tag"] = tag
    # Redirect back
    return HTMLResponse(f"<script>location.href='/admin/moderation?token={token}'</script>")

@app.get("/admin/export.csv")
def export_csv(token: str = Query(default="")):
    if BACKUP_TOKEN and token != BACKUP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    buf = io.StringIO()
    w = csv.writer(buf)
    # Header
    w.writerow(["ts","type","question","answer_id","helpful","comment","mod_tag"])
    # Events (as simple lines)
    if EVENTS_LOG.exists():
        with open(EVENTS_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    w.writerow([e.get("ts"), "event:"+str(e.get("type")), e.get("question",""), e.get("answer_id",""), "", "", ""])
                except: pass
    # Feedback
    for r in _feedback:
        w.writerow([r.get("ts"), "feedback", r.get("question",""), r.get("answer_id",""),
                    "1" if r.get("helpful") else "0", (r.get("comment") or "").replace("\n"," "),
                    r.get("mod_tag","")])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv")

# -------------------------------
# Landing & UI Pages
# -------------------------------
INDEX_HTML = rf"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Explaina — {BRANDLINE}</title>

<meta name="description" content="Explaina — the fastest way to see any answer explained with an instant AI video.">
<meta property="og:title" content="Explaina — Instant Video Answers">
<meta property="og:description" content="Ask anything, get a narrated video explained instantly.">
<meta property="og:image" content="/static/logo.png">
<meta property="og:url" content="https://explaina.net">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="/static/favicon.ico">

<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/css/landing.css">
</head>
<body>
  <header class="nav">
    <div class="brand">
      <img src="/static/logo.png" alt="Explaina logo" class="logo"/>
      <span class="name">Explaina</span>
    </div>
    <nav class="links">
      <div class="langbar">
        <label for="landingLang" style="font-size:14px">🌐</label>
        <select id="landingLang"></select>
      </div>
      <a href="/ask">Ask</a>
      <a href="/admin/dashboard">Dashboard</a>
      <a href="/admin/backups">Backups</a>
    </nav>
  </header>

  <main class="hero">
    <div class="hero-text">
      <h1>{BRANDLINE}</h1>
      <p class="sub">Multi-language, shareable, portrait-ready videos for TikTok/Shorts/Reels.</p>

      <div class="langbar">
        <label for="lpLang" style="font-size:14px;opacity:.85">Language:</label>
        <select id="lpLang">
          <option value="auto">Auto</option>
          <option value="en">English</option><option value="fr">Français</option>
          <option value="ar">العربية</option><option value="es">Español</option>
          <option value="hi">हिन्दी</option><option value="zh">中文</option>
          <option value="pt">Português</option><option value="ru">Русский</option>
          <option value="de">Deutsch</option><option value="ja">日本語</option>
          <option value="tr">Türkçe</option><option value="bn">বাংলা</option>
          <option value="ur">اُردُو</option><option value="id">Bahasa Indonesia</option>
          <option value="sw">Kiswahili</option>
        </select>
      </div>

      <div class="askbar">
        <input id="landingQ" type="text" placeholder="Type your question…" />
        <button id="goAsk" title="Generate video">🔍</button>
      </div>

      <div class="badges">
        <a class="store-badge" href="#"><img src="/static/badges/app-store.svg" alt="Download on the App Store"></a>
        <a class="store-badge" href="#"><img src="/static/badges/google-play.svg" alt="Get it on Google Play"></a>
      </div>

      <div class="cta">
        <a class="btn" href="/ask">Try Explaina now</a>
      </div>
    </div>

    <div class="preview">
      <!-- Optional: replace with a short silent preview mp4/gif -->
      <img class="mock" src="/static/hero/happiness-preview.png" alt="Explaina preview">
    </div>
  </main>

  <section class="trusted">
    <h2>Popular topics</h2>
    <div id="miniGrid" class="grid"></div>
  </section>

  <footer class="foot">
    <span>© {time.strftime('%Y')} Explaina</span>
    <a href="/ask">Ask</a>
    <a href="/admin/dashboard">Admin</a>
  </footer>

<script src="/static/js/landing.js"></script>
</body>
</html>"""

ASK_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="description" content="Explaina — the fastest way to see any answer explained with an instant AI video.">
<meta property="og:title" content="Explaina — Instant Video Answers">
<meta property="og:description" content="Ask anything, get a narrated video explained instantly.">
<meta property="og:image" content="/static/logo.png">
<meta property="og:url" content="https://explaina.net">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="/static/favicon.ico">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/css/landing.css">
<link rel="stylesheet" href="/static/css/portrait.css" media="(max-width: 768px)">
<title>Explaina — Ask</title>
<style>
:root{--accent:#17c9c0;--ring:rgba(23,201,192,.45);--bg-1:#0f214d;--bg-2:#12305f;--bg-0:#0e1834;--panel:#14234a;--panel2:#162b5a;--text:#eef4ff;--muted:#b8c4d9;--border:#27407a;--shadow:0 24px 60px rgba(10,18,40,.45)}
*{box-sizing:border-box}body{margin:0;color:var(--text);font-family:"Noto Sans",system-ui,"Segoe UI",Arial,sans-serif;background:
radial-gradient(1200px 800px at 50% -10%, rgba(23,201,192,.08), transparent 60%),
linear-gradient(180deg,var(--bg-1),var(--bg-2) 55%,var(--bg-0))}
.wrap{max-width:1100px;margin:32px auto 100px;padding:0 20px}
.topbar{display:flex;justify-content:flex-end;margin-bottom:6px}
.lang{background:#162652;color:var(--text);border:1px solid var(--border);border-radius:10px;padding:8px 10px}
.lang:focus{outline:none;box-shadow:0 0 0 3px var(--ring)}
.brand{display:grid;place-items:center;text-align:center;gap:10px;margin:6px 0 16px}
.brand img{width:128px;height:128px;object-fit:contain;filter:drop-shadow(0 8px 22px rgba(0,0,0,.45))}
.brand h1{margin:0;font-size:26px;font-weight:800}.tag{color:var(--muted);font-size:14px}
.card{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--border);border-radius:18px;padding:16px;box-shadow:var(--shadow)}
.ask-area{margin-top:10px}.ask-box{position:relative;background:#152a5a;border:1px solid var(--border);border-radius:16px;padding:14px;box-shadow:var(--shadow)}
.ask-box textarea{width:100%;min-height:80px;background:transparent;border:0;color:var(--text);font-size:18px;line-height:1.6;resize:vertical;outline:none}
.ask-box textarea::placeholder{color:var(--muted)}.ask-status{margin-top:8px;color:var(--muted);font-size:14px}
.ask-controls{position:absolute;top:12px;right:12px;display:grid;grid-template-rows:48px 40px;gap:8px}
.ask-g{display:inline-flex;align-items:center;justify-content:center;width:48px;height:48px;border-radius:10px;background:#1a356e;color:#eef4ff;border:1px solid var(--border);box-shadow:0 8px 18px rgba(0,0,0,.25);cursor:pointer}
.ask-g .mag{font-size:18px}.ask-g:hover{filter:brightness(1.06);transform:translateY(-1px)}.ask-g:focus{outline:none;box-shadow:0 0 0 3px var(--ring)}
.more-btn{width:48px;height:40px;border-radius:10px;background:#152a5a;color:var(--text);border:1px solid var(--border);cursor:pointer}
.menu{position:relative}.menu-list{position:absolute;top:46px;right:0;background:#13285a;border:1px solid var(--border);border-radius:12px;min-width:230px;padding:6px;display:none;z-index:10}
.menu.open .menu-list{display:block}.item{display:flex;align-items:center;gap:8px;padding:9px 12px;border-radius:10px;cursor:pointer;color:var(--text);text-decoration:none}
.item:hover{background:#1a3a7f}.item[aria-disabled="true"]{opacity:.5;pointer-events:none}
.action-dock{display:grid;grid-template-columns:1fr;gap:12px;margin-top:12px}
.dock-left{display:flex;gap:8px;flex-wrap:wrap}.dock-btn{background:#152a5a;color:var(--text);border:1px solid var(--border);padding:10px 14px;border-radius:12px;cursor:pointer}
.media{display:grid;gap:12px;margin-top:12px}audio,video{width:100%;border-radius:12px;border:1px solid var(--border);background:#000}
.summary{margin-top:12px;background:#152a5a;border:1px solid var(--border);border-radius:14px;padding:14px 16px;line-height:1.7;font-size:17px;letter-spacing:.1px}
.related{margin-top:10px;color:var(--muted);font-size:14px}.related .pill{display:inline-flex;gap:6px;margin:6px 6px 0 0;padding:8px 12px;border:1px solid var(--border);border-radius:999px;background:#152a5a;color:var(--text);cursor:pointer}
.save-row{margin-top:8px;display:flex;gap:8px;align-items:center}.save-btn{background:#152a5a;color:var(--text);border:1px solid var(--border);padding:8px 12px;border-radius:10px;cursor:pointer}
.lib-drawer{position:fixed;right:14px;bottom:80px;width:360px;max-height:60vh;background:#13214a;border:1px solid var(--border);border-radius:12px;box-shadow:0 12px 28px rgba(0,0,0,.35);padding:10px;display:none;overflow:auto;z-index:3000}
.lib-drawer[aria-hidden="false"]{display:block}
.lib-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.lib-quota{font-size:12px;opacity:.8;margin-left:8px}
.lib-export,.lib-clear,.lib-close{background:transparent;border:1px solid var(--border);color:var(--text);border-radius:8px;padding:4px 8px;cursor:pointer;font-size:12px}
.lib-export:hover,.lib-clear:hover,.lib-close:hover{background:#1a3a7f}
.lib-list .item{border:1px solid var(--border);border-radius:10px;padding:8px;margin:6px 0;background:#0f1a3b}
.lib-list .meta{font-size:12px;opacity:.8}
.lib-list .actions{display:flex;gap:8px;margin-top:6px}
.lib-list .actions button{background:#152a5a;color:var(--text);border:1px solid var(--border);padding:6px 8px;border-radius:8px;cursor:pointer;font-size:12px}
.hero-title{margin:18px 6px 6px;font-size:16px;font-weight:800;color:var(--muted)}
.hero-gallery{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:6px}
@media(max-width:1100px){.hero-gallery{grid-template-columns:repeat(4,1fr)}}
@media(max-width:900px){.hero-gallery{grid-template-columns:repeat(3,1fr)}}
@media(max-width:640px){.hero-gallery{grid-template-columns:repeat(2,1fr)}}
@media(max-width:420px){.hero-gallery{grid-template-columns:1fr}}
.hero-card{position:relative;background:#152a5a;border:1px solid var(--border);border-radius:14px;overflow:hidden;cursor:pointer;transition:transform .08s ease, box-shadow .2s ease, border-color .2s ease}
.hero-card:hover{transform:translateY(-2px);border-color:#2d4c8c;box-shadow:0 10px 24px rgba(8,16,36,.35)}
.hero-thumb{aspect-ratio:16/9;background:#0f1a3b}.hero-thumb img{width:100%;height:100%;object-fit:cover;display:block}
.hero-badge{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.45);color:#fff;font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid rgba(255,255,255,.15);backdrop-filter:blur(3px)}
.hero-cap{padding:10px 12px;color:var(--text);font-size:14px;line-height:1.35;min-height:48px}
.store-badges{display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin:18px 0 6px}
.store-badge{display:inline-flex;align-items:center;justify-content:center;border-radius:12px;overflow:hidden;background:#0f1a3b;border:1px solid var(--border)}
.store-badge img{display:block;height:52px;width:auto}
#toast-root{position:fixed;left:50%;bottom:calc(110px + env(safe-area-inset-bottom,0px));transform:translateX(-50%);z-index:10000;display:grid;gap:8px;pointer-events:none}
.toast{pointer-events:auto;display:inline-flex;align-items:center;gap:10px;background:rgba(20,35,74,.96);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px 14px;box-shadow:0 10px 24px rgba(0,0,0,.35);font-size:14px}
.toast .close{margin-left:6px;cursor:pointer;opacity:.8;border:none;background:transparent;color:inherit}
.toast .close:hover{opacity:1}.toast.success{border-color:rgba(34,197,94,.5)}.toast.error{border-color:rgba(239,68,68,.55)}
.toast.progress{gap:12px}.toast .spinner{width:16px;height:16px;border-radius:50%;border:2px solid rgba(255,255,255,.25);border-top-color:var(--accent);animation:spin .8s linear infinite}
.toast .bar{width:140px;height:6px;border-radius:999px;background:rgba(255,255,255,.12);overflow:hidden;border:1px solid rgba(255,255,255,.15)}
.toast .bar>i{display:block;height:100%;width:0%;background:linear-gradient(90deg,var(--accent),#2dd4cf);transition:width .2s ease}
.toast .ptext{min-width:70px;text-align:right;opacity:.9}
@keyframes spin{to{transform:rotate(360deg)}}
/* RTL support for Arabic/Urdu/etc. */
[dir="rtl"] .ask-box textarea { text-align: right; }
[dir="rtl"] .summary { text-align: right; }
[dir="rtl"] .hero-cap { text-align: right; }
/* Presenter toggle */
.presenter-row{display:flex;align-items:center;gap:10px;margin:6px 2px;font-size:14px}
.toggle-switch{position:relative;width:44px;height:24px;background:#152a5a;border:1px solid var(--border);border-radius:999px;cursor:pointer;transition:background .2s}
.toggle-switch.on{background:var(--accent);border-color:var(--accent)}
.toggle-knob{position:absolute;top:2px;left:2px;width:18px;height:18px;background:#fff;border-radius:50%;transition:transform .2s}
.toggle-switch.on .toggle-knob{transform:translateX(20px)}
.presenter-label{color:var(--muted);cursor:pointer}
.premium-badge{display:inline-block;background:linear-gradient(135deg,#fbbf24,#f59e0b);color:#000;font-size:11px;font-weight:800;padding:2px 7px;border-radius:999px;margin-left:4px}
/* Language badge (top-right) */
.lang-chip{
  position: fixed; top: 14px; right: 14px; z-index: 2000;
  background:#152a5a; color:#eaf2ff; border:1px solid var(--border);
  border-radius: 999px; padding: 6px 10px; cursor: pointer;
  font-size: 13px; user-select: none; box-shadow: 0 6px 18px rgba(0,0,0,.25);
}
.lang-chip:focus{ outline: none; box-shadow: 0 0 0 3px var(--ring); }
.lang-menu{
  position: absolute; top: 34px; right: 0; min-width: 180px;
  background:#13285a; border:1px solid var(--border); border-radius: 10px;
  padding: 6px; display: none; box-shadow: 0 10px 22px rgba(0,0,0,.35);
}
.lang-menu.open{ display: block; }
.lang-menu button{
  display:block; width:100%; text-align:left; padding:8px 10px; margin:2px 0;
  background:transparent; border:none; color:#eaf2ff; cursor:pointer; border-radius:8px; font-size:13px;
}
.lang-menu button:hover{ background:#1a3a7f; }
.lang-subtoggle input[type="checkbox"]{
  width:14px; height:14px; accent-color:#17c9c0;
}

/* Respect RTL: keep chip in the visual corner */
[dir="rtl"] .lang-chip { right: auto; left: 14px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar"><select id="langSel" class="lang" title="Language"></select></div>
  <header class="brand">
    <img src="/static/logo.png" alt="Explaina logo" onerror="this.style.display='none'"/>
    <h1>Explaina — the fastest way to see an answer explained</h1>
    <div class="tag">Ask anything. Or explore popular topics below.</div>
  </header>

  <section class="card">
    <!-- ASK -->
    <p class="tag" style="margin:0 0 6px 2px; font-size:15px">
  💡 Type any question and click the magnifying glass to see it explained.
</p>
    <div class="presenter-row">
      <div id="presenterToggle" class="toggle-switch" role="switch" aria-checked="false" tabindex="0">
        <div class="toggle-knob"></div>
      </div>
      <label class="presenter-label" for="presenterToggle">
        🎭 AI Presenter<span class="premium-badge">PREMIUM</span>
      </label>
    </div>
    <section class="ask-area">
      <div class="ask-box">
        <label for="q" class="sr-only">Type your question</label>
        <textarea id="q" placeholder="Type your question… (Ctrl/Cmd + Enter)"></textarea>
        <div class="ask-status" id="status">Ready</div>

        <div class="ask-controls">
          <button id="askBtn" type="button" onclick="window.ask && window.ask()" class="ask-g" title="Generate video"><span class="mag">&#128269;</span></button>
          <div class="menu" id="moreMenu">
            <button id="moreBtn" class="more-btn" aria-expanded="false">⋯</button>
            <div class="menu-list">
              <div id="listenItem"  class="item" aria-disabled="true">🎧 Listen</div>
              <a   id="podcastLink" class="item" aria-disabled="true">⬇ Podcast</a>
              <a   id="videoDlLink" class="item" aria-disabled="true">⬇ Video</a>
              <div id="portraitBtn" class="item" aria-disabled="true">⬇ Share Portrait (TikTok/Shorts)</div>
              <div id="shareBtn"    class="item" aria-disabled="true">📲 Share</div>
              <div id="copyBtn"     class="item">🔗 Copy link</div>
              <div id="ytBtn"       class="item" aria-disabled="true">⮕ Publish to YouTube</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Votes only -->
    <div class="action-dock">
      <div class="dock-left">
        <button id="fbGood" class="dock-btn">👍 Helpful</button>
        <button id="fbBad"  class="dock-btn">👎 Not helpful</button>
        <span id="fbMsg" class="tag" style="margin-left:8px"></span>
      </div>
    </div>
    <div id="fbCommentRow" style="display:none;margin-top:8px">
      <textarea id="fbComment" rows="2" placeholder="What was missing?" style="width:100%;background:#152a5a;border:1px solid var(--border);border-radius:10px;color:var(--text);padding:8px"></textarea>
      <div style="display:flex;gap:8px;margin-top:6px;justify-content:flex-end">
        <button id="fbSubmit" class="dock-btn">Send feedback</button>
        <button id="fbCancel" class="dock-btn">Cancel</button>
      </div>
    </div>
    <div id="helpfulBadge" class="tag" style="margin-top:6px"></div>

    <!-- Media + summary -->
    <div class="media">
      <audio id="audio" controls style="display:none"></audio>
      <video id="video" controls style="display:none" playsinline></video>
    </div>
    <div class="summary" id="summary" style="display:none"></div>
    <div id="planPreview" style="display:none;margin-top:16px;border-radius:12px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,0.15);"></div>

    <!-- Save + Library controls -->
    <div class="save-row">
      <button id="saveCurrentBtn" class="save-btn">⭐ Save</button>
      <button id="openLibraryBtn" class="save-btn" title="View your saved Explainas">📚 Library</button>
      <span id="saveMsg" class="tag"></span>
    </div>
    <div class="related" id="related" style="display:none"><div class="tag">Related questions:</div><div id="relWrap"></div></div>

    <!-- Hero gallery -->
    <h3 class="hero-title">Explore popular answers</h3>
    <div class="hero-gallery" id="heroGallery"></div>

    <!-- Store badges -->
    <div class="store-badges" aria-label="Get the app">
      <a class="store-badge" id="appStoreLink" href="#"><img src="/static/badges/app-store.svg" alt="Download on the App Store"></a>
      <a class="store-badge" id="googlePlayLink" href="#"><img src="/static/badges/google-play.svg" alt="Get it on Google Play"></a>
    </div>
  </section>
</div>

<!-- Library Drawer -->
<div id="libraryDrawer" class="lib-drawer" aria-hidden="true">
  <div class="lib-head">
    <strong>Your Library</strong>
    <span id="libQuota" class="lib-quota">0 / 100 saved</span>
    <div style="display:flex; gap:6px; align-items:center">
      <button id="importLibraryBtn" class="lib-export" title="Restore a backup of your library">Import my library</button>
      <button id="exportLibraryBtn" class="lib-export" title="Download a copy of your saved Explainas">Export my library</button>
      <button id="shareLibraryBtn" class="lib-export" title="Share a backup of your library with your contacts">Share my library</button>
      <button id="clearLibraryBtn"  class="lib-clear"  title="Remove all saved items">Clear all</button>
      <button id="closeLibraryBtn"  class="lib-close"  title="Close">✕</button>
    </div>
  </div>
  <div id="libList" class="lib-list"></div>
</div>

<input id="importLibraryInput" type="file" accept=".backup,.json,application/json,text/plain" style="display:none">

<!-- Language badge (top-right) -->
<div class="lang-chip" id="langChip" aria-haspopup="true" aria-expanded="false" title="Language">
  <span id="langChipLabel">EN</span> ▾
  <div class="lang-menu" id="langMenu" role="menu" aria-label="Select language">
    <button data-lang="auto"  role="menuitem">Auto</button>
    <button data-lang="en"    role="menuitem">English</button>
    <button data-lang="fr"    role="menuitem">Français</button>
    <button data-lang="ar"    role="menuitem">العربية</button>
    <button data-lang="es"    role="menuitem">Español</button>
    <button data-lang="hi"    role="menuitem">हिन्दी</button>
    <button data-lang="zh"    role="menuitem">中文</button>
    <button data-lang="pt"    role="menuitem">Português</button>
    <button data-lang="ru"    role="menuitem">Русский</button>
    <button data-lang="de"    role="menuitem">Deutsch</button>
    <button data-lang="ja"    role="menuitem">日本語</button>
    <button data-lang="tr"    role="menuitem">Türkçe</button>
    <button data-lang="bn"    role="menuitem">বাংলা</button>
    <button data-lang="ur"    role="menuitem">اُردُو</button>
    <button data-lang="id"    role="menuitem">Bahasa Indonesia</button>
    <button data-lang="sw"    role="menuitem">Kiswahili</button>
    <!-- Translate-to-English toggle (default ON) -->
    <div class="lang-subtoggle" style="margin-top:6px; padding-top:6px; border-top:1px solid rgba(255,255,255,.12)">
      <label style="display:flex; gap:8px; align-items:center; font-size:12px; opacity:.95; cursor:pointer">
        <input id="toggleTranslateEn" type="checkbox" checked>
        Translate subtitles to English
      </label>
    </div>
  </div>
</div>

<!-- Toast root -->
<div id="toast-root" aria-live="polite" aria-atomic="true"></div>

<script>
// ---- FX Overlay System ----
// Create a single FX root container
(function ensureFxRoot(){
  if (!document.getElementById('fxRoot')) {
    const fx = document.createElement('div');
    fx.id = 'fxRoot';
    const layer = document.createElement('div');
    layer.className = 'fx-layer';
    fx.appendChild(layer);
    document.body.appendChild(fx);
  }
})();

function setOverlayForTopic(topic){
  const layer = document.querySelector('#fxRoot .fx-layer');
  if (!layer) return;
  const map = {
    'rain':    '/static/fx/clouds.svg',
    'flight':  '/static/fx/airflow.svg',
    'finance': '/static/fx/finance.svg'
  };
  const url = map[(topic||'').toLowerCase()] || '';
  layer.style.backgroundImage = url ? `url('${url}')` : 'none';
}

function applyThemeFromPlan(plan){
  if (!plan || !plan.topic) return;
  const t = plan.topic.toLowerCase();
  document.body.classList.remove('theme-rain','theme-flight','theme-finance','theme-generic');
  document.body.classList.add(`theme-${t}`);
  setOverlayForTopic(t);
  console.log(`[THEME] Applied theme-${t} with overlay from backend topic`);
}

// ---- feedback helpers ----
function makeAnswerId(q, lang, videoUrl, audioUrl){
  function miniHash(s){
    let h=0x811c9dc5; for(let i=0;i<s.length;i++){
      h^=s.charCodeAt(i); h+=(h<<1)+(h<<4)+(h<<7)+(h<<8)+(h<<24);
    }
    return ("0000000"+(h>>>0).toString(16)).slice(-8);
  }
  const base = [String(q||"").trim().toLowerCase(), String(lang||"en"), String(videoUrl||""), String(audioUrl||"")].join("|");
  return "ans_"+miniHash(base);
}

// globals the buttons will read
window.CURRENT_QUESTION   = null;
window.CURRENT_ANSWER_ID  = null;
window.CURRENT_AUDIO_URL  = null;
window.CURRENT_VIDEO_URL  = null;
window.CURRENT_LANG       = "en";  // update if you have a language switcher

/* Language auto-detect */
const LANGS=[["auto","Auto"],["en","English"],["zh","中文"],["hi","हिन्दी"],["es","Español"],["fr","Français"],["ar","العربية"],["bn","বাংলা"],["pt","Português"],["ru","Русский"],["ur","اُردُو"],["id","Bahasa Indonesia"],["de","Deutsch"],["ja","日本語"],["tr","Türkçe"],["sw","Kiswahili"]];
const MAP={"en":"en","zh":"zh","hi":"hi","es":"es","fr":"fr","ar":"ar","bn":"bn","pt":"pt","ru":"ru","ur":"ur","id":"id","de":"de","ja":"ja","tr":"tr","sw":"sw"};
const langSel=document.getElementById("langSel"); LANGS.forEach(([v,l])=>{const o=document.createElement("option");o.value=v;o.textContent=l;langSel.appendChild(o);});
function autoLang(){const s=localStorage.getItem("lang");if(s&&s!=="auto")return s;const c=(navigator.languages||[navigator.language||"en"]).map(x=>x.toLowerCase());for(const v of c){if(MAP[v])return save(MAP[v]);const b=v.split("-")[0];if(MAP[b])return save(MAP[b]);}return save("en");function save(v){localStorage.setItem("lang",v);return v;}}

/* Read lang from URL and store/pick it */
const params = new URLSearchParams(location.search);
const langParam = params.get("lang");
if (langParam) {
  localStorage.setItem("lang", langParam);
  window.EXPLAINA_LANG = langParam;
}

(function initLang(){const cur=localStorage.getItem("lang")||"auto";langSel.value=cur;window.EXPLAINA_LANG = (cur==="auto"?autoLang():cur);})();

/* RTL direction handling */
function isRTL(lang){ return ["ar","ur","fa","he","ps"].includes((lang||"").toLowerCase()); }
function applyDirByLang(lang){
  document.documentElement.setAttribute("dir", isRTL(lang) ? "rtl" : "ltr");
  // also flip the textarea immediately (helps when user switches language)
  const qEl = document.getElementById("q");
  if (qEl) qEl.dir = isRTL(lang) ? "rtl" : "ltr";
}
applyDirByLang(window.EXPLAINA_LANG);
if(langSel) langSel.addEventListener("change",()=>{const v=langSel.value; if(v==="auto"){localStorage.setItem("lang","auto"); window.EXPLAINA_LANG=autoLang();} else {localStorage.setItem("lang",v); window.EXPLAINA_LANG=v;} applyDirByLang(window.EXPLAINA_LANG);});

/* Toasts */
function showToast(message, type="info", opts={}) {
  const root=document.getElementById("toast-root"); if(!root) return;
  const t=document.createElement("div"); t.className="toast"+(type==="success"?" success":type==="error"?" error":""); t.setAttribute("role","status"); t.setAttribute("aria-live","polite");
  const icon=document.createElement("span"); icon.textContent=type==="success"?"✔":type==="error"?"⚠":"•";
  const msg=document.createElement("span"); msg.className="msg"; msg.textContent=message;
  const close=document.createElement("button"); close.className="close"; close.setAttribute("aria-label","Close"); close.textContent="✕"; close.onclick=remove;
  t.append(icon,msg,close); root.appendChild(t);
  const ttl=opts.duration ?? 2200; let timer=setTimeout(remove, ttl);
  t.onmouseenter=()=>clearTimeout(timer); t.onmouseleave=()=>timer=setTimeout(remove,1200);
  function remove(){ if(!t.isConnected) return; t.style.transition="opacity .15s"; t.style.opacity="0"; setTimeout(()=>t.remove(),150); }
  return { remove };
}
function showProgress({ text="Rendering…", percent=null }={}) {
  const root=document.getElementById("toast-root"); if(!root) return;
  const t=document.createElement("div"); t.className="toast progress"; t.setAttribute("role","status"); t.setAttribute("aria-live","polite");
  const spin=document.createElement("span"); spin.className="spinner";
  const msg=document.createElement("span"); msg.className="msg"; msg.textContent=text;
  const bar=document.createElement("div"); bar.className="bar"; const fill=document.createElement("i"); bar.appendChild(fill);
  const ptxt=document.createElement("span"); ptxt.className="ptext"; ptxt.textContent = (percent==null)?"":(Math.round(percent)+"%");
  const cancelBtn=document.createElement("button"); cancelBtn.className="close"; cancelBtn.textContent="Cancel"; cancelBtn.onclick=()=>{ cancelCurrentJob(); remove(); };
  const close=document.createElement("button"); close.className="close"; close.textContent="✕"; close.onclick=remove;
  t.append(spin,msg,bar,ptxt,cancelBtn,close); root.appendChild(t);
  function setP(p){ if(p==null){ fill.style.width="0%"; ptxt.textContent=""; return;} const c=Math.max(0,Math.min(100,Math.round(p))); fill.style.width=c+"%"; ptxt.textContent=c+"%"; }
  function remove(){ if(!t.isConnected) return; t.style.transition="opacity .15s"; t.style.opacity="0"; setTimeout(()=>t.remove(),150); }
  function done(message="Answer ready"){ msg.textContent=message; spin.remove(); cancelBtn.remove(); setP(100); setTimeout(remove,1200); }
  function error(message="Render failed"){ t.classList.add("error"); msg.textContent=message; spin.remove(); cancelBtn.remove(); setTimeout(remove,1600); }
  setP(percent); return { update:(p,m)=>{ if(m) msg.textContent=m; setP(p); }, done, error, remove };
}

/* Global error handler */
window.addEventListener('error', (e) => {
  console.error('[GLOBAL ERROR]', e.message, e.filename, e.lineno, e.colno, e.error);
});
window.addEventListener('unhandledrejection', (e) => {
  console.error('[UNHANDLED REJECTION]', e.reason);
});

/* Elements */
const qEl=document.getElementById("q"); const statusEl=document.getElementById("status"); const summary=document.getElementById("summary");
console.log('[DEBUG] Elements initialized, qEl:', qEl, 'statusEl:', statusEl);
const audioEl=document.getElementById("audio"); const videoEl=document.getElementById("video");

// Prefill from query parameter ?q= (params already declared above)
if (params.get("q") && qEl) { qEl.value = params.get("q"); }
const moreMenu=document.getElementById("moreMenu"); const moreBtn=document.getElementById("moreBtn");
const listenItem=document.getElementById("listenItem"); const podcastLink=document.getElementById("podcastLink"); const videoDlLink=document.getElementById("videoDlLink"); const portraitBtn=document.getElementById("portraitBtn");
const shareBtn=document.getElementById("shareBtn"); const copyBtn=document.getElementById("copyBtn"); const ytBtn=document.getElementById("ytBtn");
const askBtn=document.getElementById("askBtn");
const fbGood=document.getElementById("fbGood"); const fbBad=document.getElementById("fbBad"); const fbMsg=document.getElementById("fbMsg");
const fbCommentRow=document.getElementById("fbCommentRow"); const fbComment=document.getElementById("fbComment"); const fbSubmit=document.getElementById("fbSubmit"); const fbCancel=document.getElementById("fbCancel");
const helpfulBadge=document.getElementById("helpfulBadge");
const rel=document.getElementById("related"); const relWrap=document.getElementById("relWrap"); const saveBtn=document.getElementById("saveBtn"); const saveMsg=document.getElementById("saveMsg");

/* Helpers */
function setStatus(t){ if(statusEl) statusEl.textContent=t; }
function showSummary(text){ if(!summary) return; if(!text){ summary.style.display="none"; summary.textContent=""; return;} summary.style.display="block"; summary.textContent=text; }
function enableMenu(el,on,href){ if(!el) return; el.setAttribute("aria-disabled", on?"false":"true"); if("href" in el){ if(on && href){ el.href=href; } else { el.removeAttribute("href"); } if(on && href && el.id==="podcastLink") el.download="explaina-podcast.mp3"; if(on && href && el.id==="videoDlLink") el.download="explaina-video.mp4"; } }
(function ensureSession(){ const k="session_id"; if(!localStorage.getItem(k)){ const v=(crypto.randomUUID && crypto.randomUUID()) || (Math.random().toString(36).slice(2)+Date.now().toString(36)); localStorage.setItem(k,v); }})();

// Pretty-print the plan under the summary
function renderPlan(plan){
  if(!plan) return "";
  const s = plan.scenes || [];
  let out = "🧩 Visual Plan\n";
  out += `• Topic: ${plan.topic}\n`;
  out += `• Theme: bg ${plan.theme?.bg || "-"}, accent ${plan.theme?.accent || "-"}\n`;
  out += `• Subtitles: ${plan.subtitles?.mode || "dual"}\n`;
  out += `• Watermark: ${plan.watermark?.path || "/static/watermark.png"}\n\n`;
  s.forEach((sc, i) => {
    out += `Scene ${i+1} — ${sc.type} (${(Number(sc.duration)||0).toFixed(1)}s)\n`;
    if (sc.text) out += `  text: ${sc.text}\n`;
    if (sc.bullets?.length) out += `  bullets: ${sc.bullets.join(" • ")}\n`;
    if (sc.assets?.length)  out += `  assets: ${sc.assets.map(a=>a.type).join(", ")}\n`;
    if (sc.animations?.length) out += `  anim: ${sc.animations.map(a=>a.kind).join(", ")}\n`;
    out += `\n`;
  });
  return out;
}

/* Menu open/close */
if(moreBtn) moreBtn.addEventListener("click",()=>{ const open=moreMenu.classList.toggle("open"); moreBtn.setAttribute("aria-expanded", open?"true":"false"); });
document.addEventListener("click",(e)=>{ if(moreMenu && !moreMenu.contains(e.target)) moreMenu.classList.remove("open"); });
document.querySelectorAll('#moreMenu .item').forEach(it=>{
  it.addEventListener('click', ()=>{ if(moreMenu) moreMenu.classList.remove('open'); });
});

/* Podcast helper */
async function ensurePodcast() {
  try {
    const question = (qEl?.value || '').trim();
    if(!question){ setStatus && setStatus('Type a question'); return null; }
    const lang = window.EXPLAINA_LANG || 'en';

    const r = await fetch('/api/podcast', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ question, lang })
    });
    if(!r.ok) throw new Error('podcast HTTP '+r.status);
    const j = await r.json();

    const podScript = j.podcast_text || j.answer || window._current?.answer || '';
    showSummary(podScript);

    if (j.audio_url) {
      if(audioEl) {
        audioEl.src = j.audio_url;
        audioEl.style.display = '';
      }
      enableMenu(podcastLink, true, j.audio_url);
      enableMenu(listenItem, true);
    }

    // refresh globals so feedback works immediately
    window.CURRENT_QUESTION  = question;
    window.CURRENT_LANG      = lang;
    window.CURRENT_AUDIO_URL = j.audio_url || window.CURRENT_AUDIO_URL || '/static/sample.mp3';
    window.CURRENT_VIDEO_URL = window.CURRENT_VIDEO_URL || '';
    window.CURRENT_ANSWER_ID = makeAnswerId(window.CURRENT_QUESTION, window.CURRENT_LANG, window.CURRENT_VIDEO_URL, window.CURRENT_AUDIO_URL);
    updateHelpfulBadge(window.CURRENT_ANSWER_ID);

    return j;
  } catch(e){
    console.error('ensurePodcast failed:', e);
    showToast && showToast('Could not prepare podcast', 'error');
    return null;
  }
}

/* Actions */
if (listenItem) {
  listenItem.addEventListener("click", async ()=>{
    const j = await ensurePodcast();
    if (j && audioEl?.src) { audioEl.play(); showToast && showToast('Playing podcast'); }
  });
}
if(copyBtn) copyBtn.addEventListener("click", async ()=>{ try{ await navigator.clipboard.writeText(location.href); showToast("Link copied","success"); }catch{ showToast("Copy failed","error"); }});
if(shareBtn) shareBtn.addEventListener("click", async ()=>{ if(!navigator.share) return; try{ await navigator.share({title:"Explaina",text:(qEl?.value || '').trim()||"Explaina",url:location.href}); showToast("Shared","success"); }catch{ showToast("Share canceled"); }});
if(ytBtn) ytBtn.addEventListener("click", async ()=>{ if(!videoEl || !videoEl.src) return; showToast("Publishing…"); try{ const r=await fetch('/api/youtube/publish',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_url:videoEl.src,title:(qEl?.value || '').trim()||"Explaina export"})}); const j=await r.json(); j.ok?showToast("Published ✔","success"):showToast("Publish failed","error"); }catch{ showToast("Publish failed","error"); }});
if(podcastLink){
  podcastLink.addEventListener("click", async (e)=>{
    const j = await ensurePodcast();
    if (!j || !j.audio_url) {
      e.preventDefault();
      showToast && showToast('Podcast not ready', 'error');
    } else {
      // allow the link to download (href set in ensurePodcast via enableMenu)
      showToast && showToast('Podcast download starting', 'success');
    }
  });
}
if(videoDlLink) videoDlLink.addEventListener("click", ()=>showToast("Video download started","success"));
if(portraitBtn) portraitBtn.addEventListener("click", async ()=>{ if(!window._current?.video_url) return; showToast("Preparing portrait…"); try{ const r=await fetch('/api/export/portrait',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_url: window._current.video_url})}); const j=await r.json(); j.ok?(showToast("Portrait ready","success"),window.open(j.portrait_url,"_blank")):showToast("Portrait failed","error"); }catch{ showToast("Portrait failed","error"); }});

/* Feedback */
async function postFeedback(payload){
  const r = await fetch('/feedback', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  if(!r.ok) throw new Error("HTTP "+r.status);
  return r.json();
}

function requireAnswerId(){
  if(!window.CURRENT_ANSWER_ID || !window.CURRENT_QUESTION){
    if (fbMsg) fbMsg.textContent = "Ask a question first";
    return false;
  }
  return true;
}
async function fetchHelpfulness(ans){ const r=await fetch('/feedback/stats?answer_id='+encodeURIComponent(ans)); return r.ok? r.json(): null; }
// CURRENT_ANSWER_ID and CURRENT_QUESTION now on window object (set in ask() function)

/* Job cancellation */
let CURRENT_JOB_ID = null;
async function cancelCurrentJob() {
  if (!CURRENT_JOB_ID) return;
  try {
    await fetch('/api/cancel', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ job_id: CURRENT_JOB_ID })
    });
    showToast("Rendering canceled", "success");
  } catch {
    showToast("Cancel failed", "error");
  } finally {
    CURRENT_JOB_ID = null;
  }
}
fbGood?.addEventListener('click', async ()=>{
  if(!requireAnswerId()) return;
  try{
    fbMsg.textContent = "Saving…";
    const result = await postFeedback({
      answer_id: window.CURRENT_ANSWER_ID,
      helpful: true,
      question: window.CURRENT_QUESTION,
      comment: null,
      session_id: localStorage.getItem("session_id") || null,
      website: "" // honeypot must be empty
    });
    fbMsg.textContent = result.duplicate ? "Already recorded your vote" : "Thanks for the feedback!";
    updateHelpfulBadge(window.CURRENT_ANSWER_ID);
  }catch(e){ 
    fbMsg.textContent = "Could not save feedback."; 
    console.error(e);
    updateHelpfulBadge(window.CURRENT_ANSWER_ID); // Still update stats
  }
});
fbBad?.addEventListener('click', async ()=>{
  if(!requireAnswerId()) return;
  // show your "add a comment" UI if you have it; here's a quick no-comment path:
  try{
    fbMsg.textContent = "Noted — we'll improve this.";
    const result = await postFeedback({
      answer_id: window.CURRENT_ANSWER_ID,
      helpful: false,
      question: window.CURRENT_QUESTION,
      comment: null, // or pass the text from a textarea
      session_id: localStorage.getItem("session_id") || null,
      website: ""
    });
    if(result.duplicate) fbMsg.textContent = "Already recorded your vote";
    updateHelpfulBadge(window.CURRENT_ANSWER_ID);
  }catch(e){ 
    fbMsg.textContent = "Could not save feedback."; 
    console.error(e);
    updateHelpfulBadge(window.CURRENT_ANSWER_ID); // Still update stats
  }
});
fbCancel?.addEventListener('click', ()=>{ fbCommentRow.style.display="none"; fbMsg.textContent=""; });
fbSubmit?.addEventListener('click', async ()=>{ const comment=fbComment?.value.trim(); if(!comment){ fbMsg.textContent="Please add a short note."; return; } try{ fbMsg.textContent="Saving…"; const result = await postFeedback({answer_id:window.CURRENT_ANSWER_ID, helpful:false, question:window.CURRENT_QUESTION, comment, session_id:localStorage.getItem("session_id"), website:""}); fbCommentRow.style.display="none"; fbMsg.textContent= result.duplicate ? "Already recorded your vote" : "Thanks — we'll improve this."; updateHelpfulBadge(window.CURRENT_ANSWER_ID);}catch{ fbMsg.textContent="Could not save feedback."; updateHelpfulBadge(window.CURRENT_ANSWER_ID); }});
async function updateHelpfulBadge(ans){ try{ const s=await fetchHelpfulness(ans); helpfulBadge.textContent = (s && "total" in s) ? `Helpful: ${s.helpful_rate_pct}% • ${s.total} votes` : ""; }catch{ helpfulBadge.textContent=""; }}

/* Hero gallery - daily rotation */
async function loadHeroQuestionsDaily(limit = 10) {
  const res = await fetch('/static/hero_questions.json');
  const list = await res.json();

  // Deterministic day seed
  const today = new Date().toISOString().split("T")[0].replace(/-/g, "");
  // Stable pseudo-shuffle using today's seed
  const seeded = [...list].map((it, i) => {
    // simple hash mix for order
    const h = (it.q + today).split("").reduce((a,c)=>((a<<5)-a) + c.charCodeAt(0) | 0, 0);
    return {h, it};
  }).sort((a,b)=> a.h - b.h).map(x=>x.it);

  const daily = seeded.slice(0, limit);
  const gallery = document.getElementById("heroGallery");
  gallery.innerHTML = "";

  daily.forEach(item => {
    const card = document.createElement("article");
    card.className = "hero-card";
    card.setAttribute("data-q", item.q);
    card.innerHTML = `
      <div class="hero-thumb"><img loading="lazy" decoding="async" src="${item.img}" alt=""></div>
      <span class="hero-badge">${item.cat}</span>
      <div class="hero-cap">${item.q}</div>
    `;
    card.addEventListener("click", ()=>{ if(qEl) qEl.value = item.q; if(window.ask) window.ask(); });
    gallery.appendChild(card);
  });
}
loadHeroQuestionsDaily(10);

/* Hero theme swap */
function applyHeroThemeSwap() {
  const light = document.documentElement.classList.contains('light');
  document.querySelectorAll('.hero-card img').forEach(img => {
    if (light && img.src.includes('-preview.png') && !img.src.includes('-light.png')) {
      img.src = img.src.replace('-preview.png', '-preview-light.png');
    } else if (!light && img.src.includes('-light.png')) {
      img.src = img.src.replace('-preview-light.png', '-preview.png');
    }
  });
}
applyHeroThemeSwap();
document.addEventListener('explaina:themechange', applyHeroThemeSwap);

/* Save + related (stubs) */
if(saveBtn) saveBtn.addEventListener("click", async ()=>{ try{ const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:(qEl?.value || '').trim()})}); const j=await r.json(); saveMsg.textContent = j.ok?"Saved to your library":"Save failed"; }catch{ saveMsg.textContent="Save failed"; }});

/* Presenter toggle */
const presenterToggle = document.getElementById("presenterToggle");
let presenterEnabled = false;  // Free users default to OFF
if(presenterToggle) {
  presenterToggle.addEventListener("click", () => {
    presenterEnabled = !presenterEnabled;
    presenterToggle.classList.toggle("on", presenterEnabled);
    presenterToggle.setAttribute("aria-checked", presenterEnabled ? "true" : "false");
  });
  presenterToggle.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      presenterToggle.click();
    }
  });
}

/* Ask flow */
// ---- status helpers ----
const st  = document.getElementById('st') || document.getElementById('status');
function setStatus(t){ if(st) st.textContent = t; }

// ---- theme detection based on question keywords ----
function applyDynamicTheme(question) {
  const q = question.toLowerCase();
  const body = document.body;
  
  // Remove all existing theme classes
  body.classList.remove('theme-rain', 'theme-flight', 'theme-finance');
  
  // Weather/rain theme - cyan with clouds and rain
  if (q.includes('rain') || q.includes('weather') || q.includes('storm') || q.includes('cloud') || q.includes('snow') || q.includes('precipitation')) {
    body.classList.add('theme-rain');
    console.log('[THEME] Applied theme-rain');
  }
  // Flight/aviation theme - gold with airflow lines
  else if (q.includes('fly') || q.includes('flight') || q.includes('airplane') || q.includes('aircraft') || q.includes('plane') || q.includes('aviation') || q.includes('aerodynamic') || q.includes('lift') || q.includes('wing')) {
    body.classList.add('theme-flight');
    console.log('[THEME] Applied theme-flight');
  }
  // Finance theme - green with upward shimmer
  else if (q.includes('interest') || q.includes('compound') || q.includes('invest') || q.includes('money') || q.includes('finance') || q.includes('stock') || q.includes('bank') || q.includes('profit') || q.includes('wealth') || q.includes('savings')) {
    body.classList.add('theme-finance');
    console.log('[THEME] Applied theme-finance');
  }
  // Default: no theme class (original gradient)
  else {
    console.log('[THEME] Using default theme');
  }
}

// ---- ask function (keep existing full logic) ----
console.log('[DEBUG] Defining window.ask...');
window.ask = async function ask(){
  console.log('[DEBUG] window.ask called, qEl:', qEl);
  try{
    const question=(qEl?.value || '').trim(); 
    if(!question) { setStatus('Type a question'); return; }
    const lang=window.EXPLAINA_LANG||"en";
    
    // Apply dynamic theme based on question content
    applyDynamicTheme(question);
    
    setStatus('Thinking…');
    const prog = showProgress({ text:"Thinking… 0%", percent:0 });
    showSummary(""); 
    if(audioEl) audioEl.style.display='none'; 
    if(videoEl) videoEl.style.display='none';
    const previewEl = document.getElementById('planPreview');
    if (previewEl) previewEl.style.display = 'none';
    
    fetch('/events',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'ask',question})});
    
    prog.update(30, "Generating answer…");
    const r = await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,lang})});
    if(!r.ok) throw new Error('answer HTTP '+r.status);
    const data = await r.json();

    // Set globals immediately after receiving data
    const textAnswer = data.answer || data.text || data.message || "";
    const audioUrl   = data.audio_url || "/static/sample.mp3";
    const videoUrl   = data.video_url || ""; // often blank until render finishes

    window.CURRENT_QUESTION  = (qEl?.value || "").trim();
    window.CURRENT_LANG      = lang;
    window.CURRENT_AUDIO_URL = audioUrl;
    window.CURRENT_VIDEO_URL = videoUrl || "";
    // set a preliminary answer_id immediately so feedback works even before render
    window.CURRENT_ANSWER_ID = makeAnswerId(window.CURRENT_QUESTION, window.CURRENT_LANG, window.CURRENT_VIDEO_URL, window.CURRENT_AUDIO_URL);

    // Instantly show text + audio (feels immediate!)
    prog.update(50, "Answer ready");
    const text = (typeof textAnswer==="string") ? textAnswer : JSON.stringify(data);
    showSummary(text);
    
    // If backend returned a plan, append it so users see *exact* visual instructions
    const plan = data.video_plan;
    
    // Apply backend-driven theme and overlay based on video_plan.topic
    if (data.video_plan) {
      applyThemeFromPlan(data.video_plan);
    }
    
    if (plan) {
      const planText = renderPlan(plan);
      if (planText) {
        const existing = summary.textContent || "";
        showSummary((existing ? existing + "\n\n" : "") + planText);
      }
      
      // Add visual preview
      const previewEl = document.getElementById('planPreview');
      if (previewEl && plan.theme) {
        previewEl.style.display = 'block';
        previewEl.style.background = `linear-gradient(135deg, ${plan.theme.bg || '#0f1836'} 0%, ${plan.theme.accent || '#17c9c0'} 100%)`;
        previewEl.innerHTML = `
          <div style="padding:20px;color:white;border-radius:8px;">
            <div style="font-size:18px;font-weight:bold;margin-bottom:10px;">🎬 Video Preview</div>
            <div style="font-size:14px;opacity:0.9;">Topic: ${plan.topic || 'generic'}</div>
            <div style="font-size:14px;opacity:0.9;">Mood: ${plan.mood || 'neutral'}</div>
            <div style="font-size:12px;opacity:0.7;margin-top:8px;">Theme Colors: ${plan.theme.bg} / ${plan.theme.accent}</div>
          </div>
        `;
      }
    }
    
    setStatus('Answer ready');
    
    window._current = { audio_url:audioUrl, video_url:videoUrl||null, answer:text, title:question };
    
    if (audioUrl) {
      if(audioEl) {
        audioEl.src = audioUrl;
        audioEl.style.display = '';
      }
      enableMenu(listenItem, true);
      enableMenu(podcastLink, true, audioUrl);
    } else {
      if(audioEl) audioEl.style.display = 'none';
      enableMenu(listenItem, false);
      enableMenu(podcastLink, false);
    }
    
    // Video not ready yet - disable video menus
    if(videoEl) videoEl.style.display = 'none';
    enableMenu(videoDlLink, false);
    enableMenu(ytBtn, false);
    enableMenu(portraitBtn, false);
    
    const canShare = !!(navigator.share && window._current.answer);
    enableMenu(shareBtn, canShare);

    // Update feedback badge
    updateHelpfulBadge(window.CURRENT_ANSWER_ID);

    // Start async video render in background
    try {
      const isPremium = false;
      const preset = (isPremium && presenterEnabled) ? "hd" : "mobile";
      
      const avatar = {
        provider: presenterEnabled ? 'did' : 'none',
        voice: 'en-US-jenny',
        style: 'clean'
      };
      
      const start = await fetch('/api/render/start', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          question: window.CURRENT_QUESTION,
          text: (data.answer || ""),
          audio_url: data.audio_url,
          watermark: '/static/watermark.png',
          lang: lang,
          preset,
          subtitle_mode: (window.SUBTITLE_MODE || localStorage.getItem('subtitles_mode') || 'dual'),
          avatar,
          video_plan: data.video_plan
        })
      }).then(r=>r.json());

      let result = null;

      if (start.ok && start.cached && start.result) {
        prog.update(100, "Video ready (cached)");
        result = start.result;
      } else if (start.ok && start.job_id) {
        CURRENT_JOB_ID = start.job_id;
        prog.update(55, "Composing video…");
        const poll = async () => {
          const pr = await fetch('/api/progress?job_id=' + start.job_id).then(r=>r.json());
          if (pr && pr.ok) {
            if (typeof pr.percent === 'number') {
              prog.update(pr.percent, `Composing… ${pr.percent}%`);
            }
            if (pr.status === 'done' && pr.result?.video_url) {
              CURRENT_JOB_ID = null;
              window.CURRENT_VIDEO_URL = pr.result.video_url || "";
              window.CURRENT_ANSWER_ID = makeAnswerId(window.CURRENT_QUESTION, window.CURRENT_LANG, window.CURRENT_VIDEO_URL, window.CURRENT_AUDIO_URL);
              return pr.result;
            }
            if (pr.status === 'canceled') {
              CURRENT_JOB_ID = null;
              throw new Error('Rendering canceled');
            }
            if (pr.status === 'error') {
              CURRENT_JOB_ID = null;
              throw new Error(pr.error || 'compose failed');
            }
          }
          await new Promise(res => setTimeout(res, 700));
          return poll();
        };
        result = await poll();
      }

      if (result?.video_url) {
        window._current.video_url = result.video_url;
        if(videoEl) {
          videoEl.src = result.video_url;
          videoEl.style.display = '';
        }
        enableMenu(videoDlLink, true, result.video_url);
        enableMenu(ytBtn, true);
        enableMenu(portraitBtn, true);
        enableMenu(shareBtn, !!(navigator.share && (window._current.video_url || window._current.answer)));
        
        // Update globals with video URL and regenerate answer_id
        window.CURRENT_VIDEO_URL = result.video_url;
        window.CURRENT_ANSWER_ID = makeAnswerId(window.CURRENT_QUESTION, window.CURRENT_LANG, window.CURRENT_VIDEO_URL, window.CURRENT_AUDIO_URL);
        updateHelpfulBadge(window.CURRENT_ANSWER_ID);
      }
    } catch(e) {
      console.error('render start failed:', e);
    }

    prog.update(100,"Complete"); prog.done("Complete");

    // Related questions
    try{
      const rr=await fetch('/api/related',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})}); const jj=await rr.json();
      if(jj.ok && jj.related && jj.related.length){ rel.style.display="block"; relWrap.innerHTML=""; jj.related.forEach(t=>{ const b=document.createElement('button'); b.className="pill"; b.textContent=t; b.onclick=()=>{ if(qEl) qEl.value=t; ask(); }; relWrap.appendChild(b); });}
      else rel.style.display="none";
    }catch{}

    fetch('/events',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'answer_ready', meta:{hasAudio:!!window._current.audio_url, hasVideo:!!window._current.video_url}})});

  }catch(e){
    console.error('ask() failed:', e);
    const prog_err = showProgress({ text:"Failed", percent:0 });
    prog_err.error("Failed");
    setStatus('Error — see console');
    showSummary(String(e));
    alert('Could not generate the video. Open DevTools → Console for details.');
  }
};

// ---- robust binder (works with askBtn / go / goAsk / data-action) ----
(function bindGenerate(){
  function findBtn(){
    return document.getElementById('askBtn')
        || document.getElementById('go')
        || document.getElementById('goAsk')
        || document.querySelector('[data-action="generate"]')
        || document.querySelector('button');
  }
  const btn = findBtn();
  if(btn){
    // ensure it's not submitting a form
    if(!btn.type || btn.type.toLowerCase()!=='button') btn.type = 'button';
    // unblock disabled state
    btn.removeAttribute('disabled');
    btn.addEventListener('click', (e)=>{ e.preventDefault(); e.stopPropagation(); window.ask && window.ask(); });
  }
  // keyboard shortcut
  if(qEl){
    qEl.addEventListener('keydown', (e)=>{
      if(e.key==='Enter' && (e.ctrlKey || e.metaKey)){
        e.preventDefault(); window.ask && window.ask();
      }
    });
  }
})();

// ensure binding after DOM ready as well
document.addEventListener('DOMContentLoaded', ()=>{ /* no-op; binder ran */ });
console.log('[DEBUG] End of script 1, window.ask defined:', typeof window.ask);
</script>

<script>
(function(){
  const chip = document.getElementById('langChip');
  const menu = document.getElementById('langMenu');
  const label = document.getElementById('langChipLabel');

  // Fallback helpers if not present
  window.isRTL = window.isRTL || (lang => ["ar","ur","fa","he","ps"].includes((lang||"").toLowerCase()));
  window.applyDirByLang = window.applyDirByLang || function(lang){
    document.documentElement.setAttribute("dir", isRTL(lang) ? "rtl" : "ltr");
    const qEl = document.getElementById('q');
    if (qEl) qEl.dir = isRTL(lang) ? "rtl" : "ltr";
  };

  function detectAutoLang(){
    const cand = (navigator.languages || [navigator.language || "en"]).map(x=>x.toLowerCase());
    const MAP = {en:"en",fr:"fr",ar:"ar",es:"es",hi:"hi",zh:"zh",pt:"pt",ru:"ru",de:"de",ja:"ja",tr:"tr",bn:"bn",ur:"ur",id:"id",sw:"sw"};
    for (const v of cand){ if(MAP[v]) return MAP[v]; const b=v.split('-')[0]; if(MAP[b]) return MAP[b]; }
    return "en";
  }

  function currentLang(){
    const saved = localStorage.getItem('lang') || 'auto';
    return (saved === 'auto') ? detectAutoLang() : saved;
  }

  function updateBadge(){
    const cur = currentLang().toUpperCase();
    label.textContent = (cur.length > 3 ? cur.slice(0,3) : cur); // keep compact
    applyDirByLang(currentLang());
  }

  // Initialize from URL param if present (keeps in sync with landing → ask)
  const params = new URLSearchParams(location.search);
  const langParam = params.get("lang");
  if (langParam){
    localStorage.setItem('lang', langParam);
    window.EXPLAINA_LANG = langParam;
  } else {
    // If EXPLAINA_LANG already set elsewhere, persist it
    if (window.EXPLAINA_LANG) localStorage.setItem('lang', window.EXPLAINA_LANG);
  }

  // Sync global and badge
  window.EXPLAINA_LANG = localStorage.getItem('lang') === 'auto' ? detectAutoLang() : (localStorage.getItem('lang') || 'en');
  updateBadge();

  // Open/close
  chip.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = !menu.classList.contains('open');
    menu.classList.toggle('open', open);
    chip.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  document.addEventListener('click', () => {
    menu.classList.remove('open');
    chip.setAttribute('aria-expanded','false');
  });

  // Selection
  menu.querySelectorAll('button[data-lang]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const val = btn.getAttribute('data-lang');
      // persist user choice
      localStorage.setItem('lang', val);
      // derive effective lang for rendering
      window.EXPLAINA_LANG = (val === 'auto') ? detectAutoLang() : val;
      updateBadge();

      // Optional UX: if there is a current answer/video on screen, we keep it.
      // If you want to immediately re-ask with new language, uncomment:
      // const qEl = document.getElementById('q');
      // if (qEl && qEl.value.trim()) ask();

      // close menu
      menu.classList.remove('open');
      chip.setAttribute('aria-expanded','false');
      // Optional toast if you have showToast()
      if (typeof showToast === 'function') showToast(`Language set to ${btn.textContent}`, "success");
    });
  });
})();
</script>

<script>
(function(){
  const chk = document.getElementById('toggleTranslateEn');

  // Default ON (dual). Persist user preference in localStorage: 'dual' or 'local'
  const MODE_KEY = 'subtitles_mode';
  function getMode(){ return localStorage.getItem(MODE_KEY) || 'dual'; }
  function setMode(m){ localStorage.setItem(MODE_KEY, m); window.SUBTITLE_MODE = m; }

  // Initialize
  const initial = getMode();                      // 'dual' (default) or 'local'
  chk.checked = (initial === 'dual');
  window.SUBTITLE_MODE = initial;

  // Change handler
  chk.addEventListener('change', ()=>{
    const m = chk.checked ? 'dual' : 'local';
    setMode(m);
    // Optional toast if available
    if (typeof showToast === 'function')
      showToast(m === 'dual' ? "English translation ON" : "English translation OFF", "success");
  });

  // When starting a render, include the mode in the payload (patch your existing call):
  //   body: JSON.stringify({ ..., subtitle_mode: window.SUBTITLE_MODE || getMode() })
  // See the patch below for the exact place.
})();
</script>

<script>
(function(){
  const LS_KEY  = "explaina_library_v1";
  const LIB_MAX = 100;

  const saveBtn    = document.getElementById('saveCurrentBtn');
  const openLib    = document.getElementById('openLibraryBtn');
  const drawer     = document.getElementById('libraryDrawer');
  const closeLib   = document.getElementById('closeLibraryBtn');
  const clearLib   = document.getElementById('clearLibraryBtn');
  const exportLib  = document.getElementById('exportLibraryBtn');
  const importLib  = document.getElementById('importLibraryBtn');
  const shareLibBtn = document.getElementById('shareLibraryBtn');
  const importInp  = document.getElementById('importLibraryInput');
  const libList    = document.getElementById('libList');
  const libQuota   = document.getElementById('libQuota');
  const saveMsg    = document.getElementById('saveMsg');

  function loadLib(){
    try { return JSON.parse(localStorage.getItem(LS_KEY) || "[]"); }
    catch { return []; }
  }
  function saveLib(arr){
    if (arr.length > LIB_MAX) arr.length = LIB_MAX;
    localStorage.setItem(LS_KEY, JSON.stringify(arr));
    updateQuota();
  }
  function updateQuota(){
    const n = loadLib().length;
    if (libQuota) libQuota.textContent = `${n} / ${LIB_MAX} saved`;
  }

  function buildId(q, lang, videoUrl, audioUrl){
    const s = (q||"").trim().toLowerCase()+"|"+(lang||"en")+"|"+(videoUrl||"")+"|"+(audioUrl||"");
    let h = 0x811c9dc5; for (let i=0;i<s.length;i++){ h ^= s.charCodeAt(i); h += (h<<1)+(h<<4)+(h<<7)+(h<<8)+(h<<24); }
    return "sv_"+("0000000"+(h>>>0).toString(16)).slice(-8);
  }
  function currentItem(){
    const q = (document.getElementById('q')?.value || "").trim();
    const lang = window.EXPLAINA_LANG || "en";
    const v = window._current?.video_url || null;
    const a = window._current?.audio_url || null;
    return { id: buildId(q, lang, v, a), q, lang, video_url: v, audio_url: a, ts: Date.now() };
  }
  function isSaved(id){ return loadLib().some(it => it.id === id); }

  function setSavedState(){
    const it = currentItem();
    saveBtn.textContent = it.q && isSaved(it.id) ? "✅ Saved" : "⭐ Save";
  }

  async function doSave(){
    const it = currentItem();
    if (!it.q){ saveMsg.textContent = "Ask something first."; return; }
    const lib = loadLib();
    if (lib.some(x => x.id === it.id)){ saveMsg.textContent = "Already in your library."; return; }
    lib.unshift(it); saveLib(lib);
    saveMsg.textContent = "Saved to your library.";
    if (typeof showToast === 'function') showToast("Saved to Library", "success");
    setSavedState();
  }

  function renderLib(){
    const lib = loadLib();
    updateQuota();
    if (!lib.length){ libList.innerHTML = "<div class='item'>Your library is empty.</div>"; return; }
    libList.innerHTML = "";
    lib.forEach(it=>{
      const when = new Date(it.ts).toLocaleString();
      const div = document.createElement('div');
      div.className = "item";
      div.innerHTML = `
        <div class="q"><strong>${it.q || '(untitled)'}</strong></div>
        <div class="meta">${it.lang.toUpperCase()} • ${when}</div>
        <div class="actions">
          <button data-act="play">▶ Play</button>
          <button data-act="copy">🔗 Copy Link</button>
          <button data-act="remove">🗑 Remove</button>
        </div>
      `;
      div.querySelector('[data-act="play"]').onclick = ()=>{
        const qEl = document.getElementById('q');
        if (qEl) qEl.value = it.q;
        if (it.video_url){
          window._current = { video_url: it.video_url, audio_url: it.audio_url };
          const v = document.getElementById('video'); const a = document.getElementById('audio');
          if (v){ v.src = it.video_url; v.style.display=''; }
          if (a && it.audio_url){ a.src = it.audio_url; a.style.display=''; }
          if (typeof showToast === 'function') showToast("Loaded from Library", "success");
        } else if (typeof ask === 'function'){ ask(); }
        drawer.setAttribute('aria-hidden',"true");
      };
      div.querySelector('[data-act="copy"]').onclick = ()=>{
        const url = it.video_url || it.audio_url || location.href;
        navigator.clipboard.writeText(url).then(()=>{ if (typeof showToast==='function') showToast("Link copied","success"); });
      };
      div.querySelector('[data-act="remove"]').onclick = ()=>{
        saveLib(loadLib().filter(x => x.id !== it.id));
        renderLib();
      };
      libList.appendChild(div);
    });
  }

  function downloadTextAsFile(filename, text){
    const blob = new Blob([text], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = filename; document.body.appendChild(a);
    a.click(); setTimeout(()=>{ URL.revokeObjectURL(url); a.remove(); }, 0);
  }

  function exportLibrary(){
    const lib = loadLib();
    if (!lib.length){ (typeof showToast==='function') ? showToast("Your library is empty","error") : alert("Your library is empty."); return; }
    const today = new Date().toISOString().slice(0,10);
    const fname = `Explaina-Library-${today}.backup`;
    downloadTextAsFile(fname, JSON.stringify(lib, null, 2));
    if (typeof showToast === 'function') showToast("Library backup downloaded","success");
  }

  function libToBackupText(){
    const lib = loadLib();
    return JSON.stringify(lib, null, 2);
  }

  async function shareLibrary(){
    const lib = loadLib();
    if (!lib.length){
      (typeof showToast==='function') ? showToast("Your library is empty","error") : alert("Your library is empty.");
      return;
    }
    const today = new Date().toISOString().slice(0,10);
    const fname = `Explaina-Library-${today}.backup`;
    const text  = libToBackupText();
    const blob  = new Blob([text], { type: 'application/json' });

    try {
      // Prefer modern Web Share with files (mobile & many desktops)
      if (navigator.canShare && navigator.canShare({ files: [new File([blob], fname, { type:'application/json' })] })) {
        const file = new File([blob], fname, { type:'application/json' });
        await navigator.share({
          title: 'Explaina Library',
          text:  'Here is my Explaina library backup file.',
          files: [file]
        });
        if (typeof showToast==='function') showToast("Shared via system share", "success");
        return;
      }
    } catch (e) {
      // If user cancels share or share fails, fall through to clipboard
    }

    // Fallback 1: copy to clipboard
    try {
      await navigator.clipboard.writeText(text);
      (typeof showToast==='function')
        ? showToast("Library copied to clipboard — paste it into a message", "success")
        : alert("Library copied to clipboard — paste it into a message.");
      return;
    } catch (e) {
      // Fallback 2: open in new tab as a data URL so user can save/share manually
      const url = URL.createObjectURL(blob);
      const w = window.open(url, "_blank");
      // Revoke later to avoid leaks
      setTimeout(()=>URL.revokeObjectURL(url), 30_000);
      (typeof showToast==='function')
        ? showToast("Opened backup in a new tab", "success")
        : alert("Opened backup in a new tab.");
    }
  }

  function normalizeEntry(raw){
    const it = Object(raw || {});
    const q   = (typeof it.q === 'string' ? it.q : '').trim();
    const lang= (typeof it.lang === 'string' && it.lang ? it.lang : 'en').toLowerCase();
    const vu  = (typeof it.video_url === 'string' ? it.video_url : null);
    const au  = (typeof it.audio_url === 'string' ? it.audio_url : null);
    const ts  = Number.isFinite(it.ts) ? it.ts : Date.now();
    const id  = (typeof it.id === 'string' && it.id) ? it.id : buildId(q, lang, vu, au);
    if (!q) return null;
    return { id, q, lang, video_url: vu, audio_url: au, ts };
  }

  function mergeLibraries(existing, incoming){
    const map = new Map();
    existing.forEach(e => { if (e && e.id) map.set(e.id, e); });
    incoming.forEach(n => {
      if (!n || !n.id) return;
      const prev = map.get(n.id);
      if (!prev || (Number(n.ts)||0) > (Number(prev.ts)||0)) map.set(n.id, n);
    });
    const merged = Array.from(map.values()).sort((a,b)=> (b.ts||0)-(a.ts||0));
    if (merged.length > LIB_MAX) merged.length = LIB_MAX;
    return merged;
  }

  function importLibraryFromText(txt){
    let data;
    try { data = JSON.parse(txt); }
    catch { throw new Error("Invalid file format"); }
    if (!Array.isArray(data)) throw new Error("Backup must contain a list of saved items");

    const cleaned = data.map(normalizeEntry).filter(Boolean);
    if (!cleaned.length) throw new Error("No valid items found in the file");

    const merged = mergeLibraries(loadLib(), cleaned);
    saveLib(merged);
    renderLib();
    if (typeof showToast === 'function') showToast(`Imported ${cleaned.length} item(s)`, "success");
  }

  function handleImportFile(file){
    const maxSize = 2 * 1024 * 1024;
    if (file.size > maxSize) { throw new Error("Backup file is too large"); }
    const reader = new FileReader();
    reader.onload = e => {
      try { importLibraryFromText(String(e.target.result || "")); }
      catch (err) {
        (typeof showToast==='function') ? showToast(String(err.message||err), "error") : alert(String(err.message||err));
      }
    };
    reader.readAsText(file, 'utf-8');
  }

  // Wire up buttons
  saveBtn?.addEventListener('click', doSave);
  openLib?.addEventListener('click', ()=>{ renderLib(); drawer.setAttribute('aria-hidden',"false"); });
  closeLib?.addEventListener('click', ()=>{ drawer.setAttribute('aria-hidden',"true"); });
  clearLib?.addEventListener('click', ()=>{
    const n = loadLib().length; if (!n) return;
    if (!confirm(`Remove all ${n} saved items?`)) return;
    saveLib([]); renderLib();
    if (typeof showToast==='function') showToast("Library cleared","success");
  });
  exportLib?.addEventListener('click', exportLibrary);
  shareLibBtn?.addEventListener('click', shareLibrary);
  importLib?.addEventListener('click', () => importInp?.click());
  importInp?.addEventListener('change', () => {
    const f = importInp.files && importInp.files[0];
    if (!f) return;
    try { handleImportFile(f); }
    catch (err) {
      (typeof showToast==='function') ? showToast(String(err.message||err), "error") : alert(String(err.message||err));
    } finally {
      importInp.value = "";
    }
  });

  // Hook ask() to refresh button state after answers
  if (typeof window.ask === 'function') {
    const _origAsk = window.ask;
    window.ask = async function(){ saveMsg.textContent=""; try { return await _origAsk.apply(this, arguments); } finally { setSavedState(); } };
  }

  // Init
  updateQuota();
  setSavedState();
})();
</script>
</body></html>
"""

PRIVACY_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Explaina — Privacy Policy</title>
<link rel="stylesheet" href="/static/css/landing.css">
<style>
.page{max-width:900px;margin:32px auto;padding:0 18px}
.page h1{margin:0 0 10px}
.section{background:#13214a;border:1px solid #27407a;border-radius:12px;padding:16px;margin:12px 0}
.page a{color:#17c9c0}
.small{opacity:.85;font-size:14px}
</style>
</head>
<body>
<header class="nav">
  <div class="brand"><img src="/static/logo.png" class="logo" alt="Explaina"><span class="name">Explaina</span></div>
  <nav class="links"><a href="/">Home</a><a href="/ask">Ask</a></nav>
</header>

<main class="page">
  <h1>Privacy Policy</h1>
  <div class="small">Last updated: {{DATE}}</div>

  <div class="section">
    <h3>What we collect</h3>
    <p>We collect the question you ask, basic usage events (ask, answer ready, share, downloads), and optional feedback you submit. If you enable cookies/local storage, we store a random session ID to improve quality and prevent spam.</p>
  </div>

  <div class="section">
    <h3>How we use it</h3>
    <p>We use data to render answers, improve quality (e.g., helpful vs. not helpful), detect abuse, and measure aggregate usage. We do not sell personal data.</p>
  </div>

  <div class="section">
    <h3>Third-party services</h3>
    <p>Explaina may use cloud providers (e.g., hosting/CDN/TTS/translation) to deliver the product. These providers process data on our behalf under contractual safeguards.</p>
  </div>

  <div class="section">
    <h3>Data retention</h3>
    <p>Events and feedback are retained for analytics and product improvement. You can request deletion of feedback you submitted by contacting us.</p>
  </div>

  <div class="section">
    <h3>Your choices</h3>
    <ul>
      <li>Use Explaina without logging in (session-based).</li>
      <li>Clear your browser storage to reset your session ID.</li>
      <li>Contact us to request removal of specific feedback comments.</li>
    </ul>
  </div>

  <div class="section">
    <h3>Contact</h3>
    <p>Email: <a href="mailto:support@explaina.net">support@explaina.net</a></p>
  </div>
</main>

<footer class="foot"><span>© {{YEAR}} Explaina</span> <a href="/terms">Terms</a></footer>
</body></html>"""

TERMS_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Explaina — Terms of Service</title>
<link rel="stylesheet" href="/static/css/landing.css">
<style>
.page{max-width:900px;margin:32px auto;padding:0 18px}
.page h1{margin:0 0 10px}
.section{background:#13214a;border:1px solid #27407a;border-radius:12px;padding:16px;margin:12px 0}
.page a{color:#17c9c0}
.small{opacity:.85;font-size:14px}
</style>
</head>
<body>
<header class="nav">
  <div class="brand"><img src="/static/logo.png" class="logo" alt="Explaina"><span class="name">Explaina</span></div>
  <nav class="links"><a href="/">Home</a><a href="/ask">Ask</a></nav>
</header>

<main class="page">
  <h1>Terms of Service</h1>
  <div class="small">Last updated: {{DATE}}</div>

  <div class="section">
    <h3>Use of Explaina</h3>
    <p>By using Explaina you agree to these terms. Do not misuse the service or attempt to disrupt it. Content generated by Explaina is for informational purposes and may contain errors—use judgment and verify important information.</p>
  </div>

  <div class="section">
    <h3>Accounts & access</h3>
    <p>Explaina may be used without an account. If accounts are introduced later, you're responsible for safeguarding your credentials and activity under your account.</p>
  </div>

  <div class="section">
    <h3>Content & IP</h3>
    <p>Explaina's software, branding, and templates are owned by Explaina. You retain rights to your own inputs; by using the service you grant us a license to process and display your inputs for the purpose of providing Explaina.</p>
  </div>

  <div class="section">
    <h3>Fair use</h3>
    <p>Don't upload illegal content or infringe others' rights. Don't automate abusive traffic. We may rate-limit, block, or terminate access to protect service integrity.</p>
  </div>

  <div class="section">
    <h3>Disclaimers & limitation of liability</h3>
    <p>Explaina is provided "as is". To the maximum extent permitted by law, Explaina disclaims all warranties and liability for any damages arising from use of the service.</p>
  </div>

  <div class="section">
    <h3>Changes</h3>
    <p>We may update these terms. Continued use after changes means you accept the updated terms.</p>
  </div>

  <div class="section">
    <h3>Contact</h3>
    <p>Email: <a href="mailto:legal@explaina.net">legal@explaina.net</a></p>
  </div>
</main>

<footer class="foot"><span>© {{YEAR}} Explaina</span> <a href="/privacy">Privacy</a></footer>
</body></html>"""

PORTRAIT_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Explaina — Portrait</title>
<link rel="icon" href="/static/favicon.ico">
<link rel="stylesheet" href="/static/css/landing.css">
<link rel="stylesheet" href="/static/css/portrait.css">
</head>
<body class="theme-generic">
  <div class="portrait-wrap">
    <header class="p-header">
      <div class="brand"><img src="/static/logo.png" class="logo" alt=""><span class="name">Explaina</span></div>
      <div class="p-chip" id="pStatus">Ready</div>
    </header>

    <section class="p-card p-ask-box">
      <textarea id="q" placeholder="Type your question… (Ctrl/Cmd+Enter)"></textarea>
      <div class="p-actions">
        <div class="p-status" id="st">Ready</div>
        <button id="askBtn" class="p-btn" type="button" onclick="window.ask && window.ask()">Generate video</button>
      </div>
    </section>

    <section class="p-media">
      <video id="video" controls playsinline style="display:none"></video>
      <audio id="audio" controls style="display:none"></audio>
    </section>

    <section class="p-summary" id="summary" style="display:none"></section>
    <div class="p-row">
      <button id="listenItem" class="p-btn" type="button">🎧 Listen</button>
      <a id="podcastLink" class="p-btn" download>⬇ Podcast</a>
    </div>
  </div>

<script>
// ---- FX Overlay System ----
(function ensureFxRoot(){
  if (!document.getElementById('fxRoot')) {
    const fx = document.createElement('div');
    fx.id = 'fxRoot';
    const layer = document.createElement('div');
    layer.className = 'fx-layer';
    fx.appendChild(layer);
    document.body.appendChild(fx);
  }
})();

function setOverlayForTopic(topic){
  const layer = document.querySelector('#fxRoot .fx-layer');
  if (!layer) return;
  const map = {
    'rain':    '/static/fx/clouds.svg',
    'flight':  '/static/fx/airflow.svg',
    'finance': '/static/fx/finance.svg'
  };
  const url = map[(topic||'').toLowerCase()] || '';
  layer.style.backgroundImage = url ? `url('${url}')` : 'none';
}

function applyThemeFromPlan(plan){
  if (!plan || !plan.topic) return;
  const t = plan.topic.toLowerCase();
  document.body.classList.remove('theme-rain','theme-flight','theme-finance','theme-generic');
  document.body.classList.add(`theme-${t}`);
  setOverlayForTopic(t);
}

// Reuse core logic from /ask
const qEl=document.getElementById("q"); const statusEl=document.getElementById("st"); 
const summary=document.getElementById("summary");
const audioEl=document.getElementById("audio"); const videoEl=document.getElementById("video");
const listenItem=document.getElementById("listenItem"); const podcastLink=document.getElementById("podcastLink");
const pStatus=document.getElementById("pStatus");

function setStatus(t){ if(statusEl) statusEl.textContent = t; if(pStatus) pStatus.textContent = t; }
function showSummary(text){ if(!summary) return; if(!text){ summary.style.display="none"; summary.textContent=""; return;} summary.style.display="block"; summary.textContent=text; }

window.EXPLAINA_LANG = "en";

// Simple ask function (reuses /api/answer endpoint)
window.ask = async function ask(){
  try{
    const question=(qEl?.value || '').trim(); 
    if(!question) { setStatus('Type a question'); return; }
    const lang=window.EXPLAINA_LANG||"en";
    
    setStatus('Thinking…');
    showSummary(""); 
    if(audioEl) audioEl.style.display='none'; 
    if(videoEl) videoEl.style.display='none';
    
    const r = await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,lang})});
    if(!r.ok) throw new Error('answer HTTP '+r.status);
    const data = await r.json();

    // Apply backend-driven theme and overlay
    if (data.video_plan) {
      applyThemeFromPlan(data.video_plan);
    }

    const text = data.answer || data.text || data.message || "";
    showSummary(text);
    
    if (data.audio_url) {
      if(audioEl) {
        audioEl.src = data.audio_url;
        audioEl.style.display = '';
      }
      if(podcastLink) podcastLink.href = data.audio_url;
    }
    
    if (data.video_url) {
      if(videoEl) {
        videoEl.src = data.video_url;
        videoEl.style.display = '';
      }
    }
    
    setStatus('Answer ready');
  } catch(e){
    console.error(e);
    setStatus('Error: '+e.message);
  }
};

// Listen button
if(listenItem) {
  listenItem.addEventListener("click", ()=>{ if(audioEl?.src) audioEl.play(); });
}

// Ctrl+Enter to ask
qEl.addEventListener("keydown", (e)=>{
  if((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); window.ask(); }
});
</script>
</body></html>
"""

@app.get("/ask", response_class=HTMLResponse)
def ask_page(): return HTMLResponse(ASK_HTML)

@app.get("/ask/portrait", response_class=HTMLResponse)
def ask_portrait(): return HTMLResponse(PORTRAIT_HTML)

def _simple_fill(html: str) -> str:
    return (html
            .replace("{{DATE}}", time.strftime("%Y-%m-%d"))
            .replace("{{YEAR}}", time.strftime("%Y")))

@app.get("/privacy", response_class=HTMLResponse)
def privacy_page():
    return HTMLResponse(_simple_fill(PRIVACY_HTML))

@app.get("/terms", response_class=HTMLResponse)
def terms_page():
    return HTMLResponse(_simple_fill(TERMS_HTML))

# -------------------------------
# Save + Related stubs (keep for UI hooks)
# -------------------------------
@app.post("/api/save")
def save_answer(payload: dict = Body(None)):
    return {"ok": True}

@app.post("/api/related")
def related(payload: dict = Body(None)):
    q = (payload or {}).get("question","").lower()
    import random
    pool = {
        "growth": [
            "How to build discipline fast?",
            "Best technique to set goals that stick?",
            "How to avoid burnout while working hard?",
        ],
        "health": [
            "Is 8 hours of sleep really necessary?",
            "How to improve posture at a desk?",
            "What is a balanced plate for dinner?",
        ],
        "money": [
            "How to start an emergency fund?",
            "Index funds vs individual stocks?",
            "How to budget if income is irregular?",
        ],
        "lifestyle": [
            "How to make new friends as an adult?",
            "How to communicate better in conflicts?",
            "Small habits to boost daily happiness?",
        ]
    }
    key = "growth" if "motivat" in q or "procrast" in q else \
          "health" if "stress" in q or "sleep" in q else \
          "money"  if "money" in q or "invest" in q else "lifestyle"
    return {"ok": True, "related": random.sample(pool[key], k=3)}
