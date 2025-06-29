Write-Host "🔍 Checking BERT Model Status..." -ForegroundColor Cyan

# Check GPU Model
$gpuModelPath = "tensorflow_models\bert_gpu_models\bert_gpu_model\saved_model.pb"
$gpuModelExists = Test-Path $gpuModelPath
Write-Host "GPU Model (SavedModel): $gpuModelExists" -ForegroundColor $(if($gpuModelExists){"Green"}else{"Red"})

# Check CPU Model
$cpuModelPath = "saved_model\model.h5"
$cpuModelExists = Test-Path $cpuModelPath
Write-Host "CPU Model (H5): $cpuModelExists" -ForegroundColor $(if($cpuModelExists){"Green"}else{"Red"})

# Check Tokenizer
$tokenizerPath = "tensorflow_models\bert_gpu_models\tokenizer"
$tokenizerExists = Test-Path $tokenizerPath
Write-Host "Tokenizer: $tokenizerExists" -ForegroundColor $(if($tokenizerExists){"Green"}else{"Red"})

# Check Label Encoder
$labelEncoderPath = "tensorflow_models\bert_gpu_models\label_encoder.pkl"
$labelEncoderExists = Test-Path $labelEncoderPath
Write-Host "Label Encoder: $labelEncoderExists" -ForegroundColor $(if($labelEncoderExists){"Green"}else{"Red"})

# Check Metadata
$metadataPath = "tensorflow_models\bert_gpu_models\metadata.json"
$metadataExists = Test-Path $metadataPath
Write-Host "Metadata: $metadataExists" -ForegroundColor $(if($metadataExists){"Green"}else{"Red"})

# Summary
Write-Host "`n📊 Summary:" -ForegroundColor Yellow
if ($gpuModelExists -and $tokenizerExists -and $labelEncoderExists) {
    Write-Host "✅ BERT GPU Model is TRAINED and READY!" -ForegroundColor Green
} elseif ($cpuModelExists) {
    Write-Host "⚠️  CPU Model exists, but GPU model incomplete" -ForegroundColor Yellow
} else {
    Write-Host "❌ No trained models found" -ForegroundColor Red
}

# File sizes
Write-Host "`n📁 Model File Sizes:" -ForegroundColor Cyan
if ($gpuModelExists) {
    $gpuSize = (Get-Item $gpuModelPath).Length / 1MB
    Write-Host "GPU Model: $([math]::Round($gpuSize, 2)) MB"
}
if ($cpuModelExists) {
    $cpuSize = (Get-Item $cpuModelPath).Length / 1MB
    Write-Host "CPU Model: $([math]::Round($cpuSize, 2)) MB"
}