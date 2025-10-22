# Explaina Monorepo Setup Script
# Run this from Desktop\Explaina folder

Write-Host "Setting up Explaina Monorepo..." -ForegroundColor Cyan

# 1. Initialize Git
Write-Host "Initializing Git repository..." -ForegroundColor Yellow
git init

# 2. Create .gitignore
Write-Host "Creating .gitignore..." -ForegroundColor Yellow
$gitignore = @"
__pycache__/
*.py[cod]
node_modules/
dist/
.env
.env.local
www/
*.zip
*.log
"@
Set-Content -Path ".gitignore" -Value $gitignore

# 3. Create directories
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "routes" | Out-Null
New-Item -ItemType Directory -Force -Path "services" | Out-Null
New-Item -ItemType Directory -Force -Path "static" | Out-Null
New-Item -ItemType Directory -Force -Path "www" | Out-Null
New-Item -ItemType Directory -Force -Path ".github\workflows" | Out-Null

# 4. Create requirements.txt
Write-Host "Creating requirements.txt..." -ForegroundColor Yellow
$requirements = @"
fastapi
uvicorn
gunicorn
httpx
python-multipart
aiofiles
google-generativeai
google-cloud-storage
structlog
"@
Set-Content -Path "requirements.txt" -Value $requirements

# 5. Create startup.txt
Write-Host "Creating startup.txt..." -ForegroundColor Yellow
$startup = "gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind=0.0.0.0:8000"
Set-Content -Path "startup.txt" -Value $startup

# 6. Create routes/health.py
Write-Host "Creating routes/health.py..." -ForegroundColor Yellow
$health = @"
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}
"@
Set-Content -Path "routes\health.py" -Value $health

# 7. Create routes/metrics.py
Write-Host "Creating routes/metrics.py..." -ForegroundColor Yellow
$metrics = @"
from fastapi import APIRouter
from services.metrics import dump

router = APIRouter(prefix="/api", tags=["metrics"])

@router.get("/metrics")
async def get_metrics():
    return dump()
"@
Set-Content -Path "routes\metrics.py" -Value $metrics

# 8. Create routes/answer.py
Write-Host "Creating routes/answer.py..." -ForegroundColor Yellow
$answer = @"
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["answer"])

# TODO: Move answer endpoints from main.py here
"@
Set-Content -Path "routes\answer.py" -Value $answer

# 9. Create services/metrics.py
Write-Host "Creating services/metrics.py..." -ForegroundColor Yellow
$metricsService = @"
from threading import Lock
from typing import Optional

_counters: dict[str, int] = {}
_lock = Lock()

def mark(key: str) -> None:
    with _lock:
        _counters[key] = _counters.get(key, 0) + 1

def dump() -> dict:
    with _lock:
        return {"counters": dict(_counters)}
"@
Set-Content -Path "services\metrics.py" -Value $metricsService

# 10. Create main.py
Write-Host "Creating main.py..." -ForegroundColor Yellow
$mainPy = @"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.health import router as health_router
from routes.answer import router as answer_router
from routes.metrics import router as metrics_router

app = FastAPI(title="Explaina API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(health_router)
app.include_router(answer_router)
app.include_router(metrics_router)

app.mount("/", StaticFiles(directory="www", html=True), name="spa")
"@
Set-Content -Path "main.py" -Value $mainPy

# 11. Create frontend directory
Write-Host "Creating frontend structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "frontend\src" | Out-Null

# 12. Create frontend/package.json
$packageJson = @"
{
  "name": "explaina-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.5.3",
    "vite": "^5.4.1"
  }
}
"@
Set-Content -Path "frontend\package.json" -Value $packageJson

# 13. Create frontend/.env
$frontendEnv = @"
# Leave empty to use same-origin
VITE_API_BASE=
"@
Set-Content -Path "frontend\.env" -Value $frontendEnv

# 14. Create frontend/index.html
$indexHtml = @"
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Explaina</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"@
Set-Content -Path "frontend\index.html" -Value $indexHtml

# 15. Create frontend/vite.config.ts
$viteConfig = @"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5000,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
"@
Set-Content -Path "frontend\vite.config.ts" -Value $viteConfig

# 16. Create frontend/tsconfig.json
$tsconfig = @"
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true
  },
  "include": ["src"]
}
"@
Set-Content -Path "frontend\tsconfig.json" -Value $tsconfig

# 17. Create frontend/src/main.tsx
$mainTsx = @"
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"@
Set-Content -Path "frontend\src\main.tsx" -Value $mainTsx

# 18. Create frontend/src/App.tsx
$appTsx = @"
import { useState } from 'react'

function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')

  const handleAsk = async () => {
    const apiBase = import.meta.env.VITE_API_BASE || ''
    const res = await fetch(``${apiBase}/api/answer``, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: question })
    })
    const data = await res.json()
    setAnswer(data.answer || 'No answer')
  }

  return (
    <div style={{ padding: '2rem' }}>
      <h1>Explaina</h1>
      <input 
        value={question}
        onChange={e => setQuestion(e.target.value)}
        placeholder="Ask anything..."
        style={{ width: '300px', padding: '8px' }}
      />
      <button onClick={handleAsk} style={{ marginLeft: '8px', padding: '8px 16px' }}>
        Ask
      </button>
      {answer && <p style={{ marginTop: '1rem' }}>{answer}</p>}
    </div>
  )
}

export default App
"@
Set-Content -Path "frontend\src\App.tsx" -Value $appTsx

# 19. Create frontend/src/vite-env.d.ts
$viteEnv = "/// <reference types=`"vite/client`" />"
Set-Content -Path "frontend\src\vite-env.d.ts" -Value $viteEnv

# 20. Create GitHub Actions workflow
$workflow = @"
name: Deploy Explaina Backend (API + SPA) to Azure

on:
  push:
    branches: [ "main" ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
      - name: Build frontend
        working-directory: frontend
        run: |
          npm ci
          npm run build
          mkdir -p ../www
          cp -r dist/* ../www/

      - name: Zip package
        run: |
          zip -r package.zip . -x "*.git*" "frontend/node_modules/*"
      - name: Deploy to Azure Web App
        uses: azure/webapps-deploy@v3
        with:
          app-name: explaina-backend
          publish-profile: `${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
          package: package.zip
"@
Set-Content -Path ".github\workflows\deploy.yml" -Value $workflow

# 21. Create www/index.html
$wwwIndex = @"
<!DOCTYPE html>
<html>
<head><title>Explaina</title></head>
<body><h1>Build frontend to replace this</h1></body>
</html>
"@
Set-Content -Path "www\index.html" -Value $wwwIndex

# 22. Create README
$readme = @"
# Explaina Monorepo

## Quick Start

Backend:
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

Frontend:
cd frontend
npm install
npm run dev

## Deployment
Push to main branch for automatic deployment to Azure
"@
Set-Content -Path "README.md" -Value $readme

# 23. Initial commit
Write-Host "Creating initial commit..." -ForegroundColor Yellow
git add .
git commit -m "Initial commit: Explaina monorepo setup"

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. cd frontend" -ForegroundColor White
Write-Host "  2. npm install" -ForegroundColor White
Write-Host "  3. pip install -r requirements.txt" -ForegroundColor White