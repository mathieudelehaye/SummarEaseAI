Write-Host "Checking BERT Model Status..." -ForegroundColor Cyan

# Check GPU Model
$gpuModelPath = "tensorflow_models\bert_gpu_models\bert_gpu_model\saved_model.pb"
$gpuModelExists = Test-Path $gpuModelPath
Write-Host "GPU Model (SavedModel): $gpuModelExists" -ForegroundColor Green

# Check CPU Model  
$cpuModelPath = "saved_model\model.h5"
$cpuModelExists = Test-Path $cpuModelPath
Write-Host "CPU Model (H5): $cpuModelExists" -ForegroundColor Green

# Check Tokenizer
$tokenizerPath = "tensorflow_models\bert_gpu_models\tokenizer"
$tokenizerExists = Test-Path $tokenizerPath
Write-Host "Tokenizer: $tokenizerExists" -ForegroundColor Green

# Check Label Encoder
$labelEncoderPath = "tensorflow_models\bert_gpu_models\label_encoder.pkl"
$labelEncoderExists = Test-Path $labelEncoderPath
Write-Host "Label Encoder: $labelEncoderExists" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "Summary:" -ForegroundColor Yellow
if ($gpuModelExists -and $tokenizerExists -and $labelEncoderExists) {
    Write-Host "BERT GPU Model is TRAINED and READY!" -ForegroundColor Green
} elseif ($cpuModelExists) {
    Write-Host "CPU Model exists, but GPU model incomplete" -ForegroundColor Yellow  
} else {
    Write-Host "No trained models found" -ForegroundColor Red
} 