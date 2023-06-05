# Запуск проекта
## Создай пустую виртуальную среду conda
Например, python 3.10
## Скачай torch и cuda toolkit
source - https://pytorch.org/get-started/locally/
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
## Скачай все остальные зависимости
Так как torch уже скачан, необходимо удалить его из requirements.txt
или рекурсивно скачивать библиотеки из подготовленного requirements_dev.txt
## Проверим
```python
import torch

torch.cuda.is_available()
```
Должно быть True

# Обучение на локалке
1. скачай датасет с kaggle (https://www.kaggle.com/datasets/tthien/shanghaitech) и положи в корень
2. скачай веса для vgg и расположи в папке models_storage (нужна только vgg16_bn)
```python
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
```
3. запусти скрипт parse_dataset.py (1 раз для train, 1 раз для test) для подготовки датасета
```python
data_source_root = r'C:\dpm\CrowdCounting-P2PNet-main\ShanghaiTech\part_B\train_data'
output_dir = r'C:\dpm\CrowdCounting-P2PNet-main\DATA_ROOT'
dataset_part = 'train'
```
и
```python
data_source_root = r'C:\dpm\CrowdCounting-P2PNet-main\ShanghaiTech\part_B\test_data'
output_dir = r'C:\dpm\CrowdCounting-P2PNet-main\DATA_ROOT'
dataset_part = 'test'
```

# Проблемы
```
RuntimeError: CUDA error:
out of memory CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```
Уменьшаем батч
```python
parser.add_argument('--batch_size', default=4, type=int)
```
или собираем мусор и чистим кеш
```python
import torch
import gc

gc.collect()
torch.cuda.empty_cache()
```

