# Simple deployment script for Windows
$ErrorActionPreference = "Stop"

$SERVICE_NAME = "explaina-worker"
$REGION = "europe-west1"
$TIMESTAMP = Get-Date -Format "yyyyMMddHHmmss"

Write-Host "=== DEPLOYING WORKER ===" -ForegroundColor Cyan
Write-Host ""

# Verify hash
$FILE_HASH = (Get-FileHash -Path "worker_main.py" -Algorithm MD5).Hash
Write-Host "worker_main.py MD5: $FILE_HASH" -ForegroundColor Yellow

if ($FILE_HASH -ne "A2A566E6CADC1ACDED4D5B4AFCFA80DD") {
    Write-Host "ERROR: File hash mismatch!" -ForegroundColor Red
    exit 1
}

Write-Host "File verified OK" -ForegroundColor Green
Write-Host ""

# Delete old service
Write-Host "Deleting old service..." -ForegroundColor Yellow
gcloud run services delete $SERVICE_NAME --region=$REGION --quiet 2>&1 | Out-Null
Start-Sleep -Seconds 15
Write-Host ""

# Deploy
Write-Host "Deploying (3-5 minutes)..." -ForegroundColor Yellow
gcloud run deploy $SERVICE_NAME --source . --dockerfile=Dockerfile --region=$REGION --allow-unauthenticated --memory=512Mi --cpu=1 --timeout=300 --no-cache --set-env-vars="DEPLOY_TIMESTAMP=$TIMESTAMP"

if ($LASTEXITCODE -ne 0) {
    Write-Host "DEPLOYMENT FAILED" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Getting service URL..." -ForegroundColor Yellow
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
Write-Host "URL: $SERVICE_URL" -ForegroundColor Green
Write-Host ""

# Test
Write-Host "Testing health endpoint..." -ForegroundColor Yellow
Start-Sleep -Seconds 20
$response = Invoke-RestMethod -Uri "$SERVICE_URL/health" -Method Get -TimeoutSec 15
Write-Host "Status: $($response.status)" -ForegroundColor White

if ($response.status -eq "ok") {
    Write-Host ""
    Write-Host "SUCCESS! Service is running!" -ForegroundColor Green
} else {
    Write-Host "Warning: Unexpected status" -ForegroundColor Yellow
}
