# Steel Scrap Vision Baseline

MobileNetV3 image-classification baseline using PyTorch ImageFolder.

## Dataset layout
```
data/train/<class_name>/*.jpg
data/val/<class_name>/*.jpg
```

## Run
```bash
pip install -r requirements.txt
python train.py
```

This is an experimental baseline. Add dataset provenance, class balance checks, held-out test evaluation, and confusion-matrix reporting before using results in production.