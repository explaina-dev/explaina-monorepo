# Nuclear deployment option - completely bypasses all caching
$ErrorActionPreference = "Stop"

$SERVICE_NAME = "explaina-worker"
$REGION = "europe-west1"
$TIMESTAMP = Get-Date -Format "yyyyMMddHHmmss"

Write-Host "=== NUCLEAR DEPLOYMENT ===" -ForegroundColor Red
Write-Host "This will force a completely fresh build with zero caching" -ForegroundColor Yellow
Write-Host ""

# Create unique timestamp
Set-Content -Path ".build_timestamp" -Value $TIMESTAMP
Write-Host "[1] Build timestamp: $TIMESTAMP" -ForegroundColor Yellow
Write-Host ""

# Verify current file
$FILE_HASH = (Get-FileHash -Path "worker_main.py" -Algorithm MD5).Hash
Write-Host "[2] worker_main.py MD5: $FILE_HASH" -ForegroundColor Yellow
Write-Host "    Expected: a2a566e6cadc1acded4d5b4afcfa80dd" -ForegroundColor Gray

if ($FILE_HASH -ne "a2a566e6cadc1acded4d5b4afcfa80dd") {
    Write-Host "    ⚠ WARNING: File hash mismatch!" -ForegroundColor Red
    exit 1
}
Write-Host "    ✓ File hash matches" -ForegroundColor Green
Write-Host ""

# Delete service
Write-Host "[3] Deleting old service..." -ForegroundColor Yellow
try {
    gcloud run services delete $SERVICE_NAME --region=$REGION --quiet 2>&1 | Out-Null
    Write-Host "    ✓ Deleted, waiting 15 seconds..." -ForegroundColor Green
    Start-Sleep -Seconds 15
} catch {
    Write-Host "    ✓ No service to delete" -ForegroundColor Green
}
Write-Host ""

# Deploy
Write-Host "[4] Deploying (this will take 3-5 minutes)..." -ForegroundColor Yellow
Write-Host "    Using Dockerfile with syntax verification step" -ForegroundColor Gray

$deployOutput = gcloud run deploy $SERVICE_NAME `
    --source . `
    --dockerfile=Dockerfile `
    --region=$REGION `
    --allow-unauthenticated `
    --memory=512Mi `
    --cpu=1 `
    --timeout=300 `
    --no-cache `
    --set-env-vars="DEPLOY_TIMESTAMP=$TIMESTAMP" `
    2>&1 | Out-String

Write-Host $deployOutput

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ DEPLOYMENT FAILED" -ForegroundColor Red
    
    # Check if syntax error appears in build logs
    if ($deployOutput -match "SyntaxError") {
        Write-Host ""
        Write-Host "ERROR: SyntaxError detected in BUILD output!" -ForegroundColor Red
        Write-Host "This means the OLD file is being uploaded, not the current one." -ForegroundColor Red
        Write-Host ""
        Write-Host "The file on your Windows machine might be different from what's being uploaded." -ForegroundColor Yellow
        Write-Host "Please verify:" -ForegroundColor Yellow
        Write-Host "  1. Check MD5 hash: (Get-FileHash worker_main.py -Algorithm MD5).Hash" -ForegroundColor Gray
        Write-Host "  2. Should be: a2a566e6cadc1acded4d5b4afcfa80dd" -ForegroundColor Gray
    }
    exit 1
}

Write-Host ""
Write-Host "[5] Getting service URL..." -ForegroundColor Yellow
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
Write-Host "    ✓ URL: $SERVICE_URL" -ForegroundColor Green
Write-Host ""

# Wait and test
Write-Host "[6] Waiting 20 seconds for service to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 20
Write-Host ""

Write-Host "[7] Testing health endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$SERVICE_URL/health" -Method Get -TimeoutSec 15
    
    Write-Host "    Status: $($response.status)" -ForegroundColor White
    
    if ($response.status -eq "ok") {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✓✓✓ SUCCESS!" -ForegroundColor Green  
        Write-Host "Service is deployed and running!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "⚠ Status mismatch!" -ForegroundColor Yellow
        Write-Host "  Expected: ok" -ForegroundColor Yellow
        Write-Host "  Got: $($response.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "    ✗ Failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Fetching logs..." -ForegroundColor Yellow
    $logs = gcloud run services logs read $SERVICE_NAME --region=$REGION --limit=100 2>&1 | Out-String
    
    if ($logs -match "SyntaxError") {
        Write-Host ""
        Write-Host "✗ SYNTAX ERROR IN RUNTIME LOGS!" -ForegroundColor Red
        Write-Host "Old code is still being deployed despite fresh build." -ForegroundColor Red
        $logs | Select-String -Pattern "SyntaxError" -Context 5
    } else {
        Write-Host $logs
    }
    exit 1
}

Write-Host ""
Write-Host "Service URL: $SERVICE_URL" -ForegroundColor Cyan
