param(
    [string]$DataRoot = "D:\project\pythonProject\Road_Identification\SAM2-UNet\deepglobe",
    [int]$Epochs = 80,
    [int]$BatchSize = 4,
    [int]$NumWorkers = 4,
    [int]$ImageSize = 512
)

python train.py `
    --data-root $DataRoot `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --num-workers $NumWorkers `
    --image-size $ImageSize
