# FaceYOLOMini – Project Layout & Setup Guide

このドキュメントでは face_yolomini.py を学習・推論させるための
ディレクトリ構成、環境構築手順、データセット配置ルールをまとめます。

## 1. 推奨ディレクトリ構造

```
face_ai/
├── env.yml                # conda 環境ファイル (任意)
├── requirements.txt       # pip 依存一覧
├── face_yolomini.py       # ⭐ モデル定義 & 推論デコーダ
├── train.py               # 学習スクリプト (シンプル版)
├── data/                  # 画像 & アノテーション
│   ├── images/
│   │   ├── train/*.jpg
│   │   └── val/*.jpg
│   └── labels/            # YOLO txt (class cx cy w h)
│       ├── train/*.txt
│       └── val/*.txt
├── runs/                  # 学習結果 (weights, log)
└── utils/
    ├── dataloader.py      # DataLoader (YOLO 形式)
    └── augment.py         # 画像拡張 (Mosaic 等)
```

> **注意**: images と labels のフォルダ名は**必ず同一階層**にしてください。
> 画像名 xxxx.jpg に対しラベル xxxx.txt が対応する YOLO 仕様です。

## 2. 依存ライブラリ

| パッケージ | バージョン (例) | 備考 |
|------------|----------------|------|
| python | 3.10+ | 3.8 以上で動作確認 |
| torch | 2.2.x | GPU 版推奨 (cu121 等) |
| torchvision | 0.17.x | 同 CUDA バージョン |
| numpy | ≥1.24 | – |
| opencv‑python | ≥4.9 | cv2 で可視化 |
| tqdm | – | 進捗バー |
| matplotlib | – | ログ可視化 (任意) |
| albumentations | – | augment を追加する場合 |

### インストール例

**pip を使用する場合:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt 例:**
```
torch==2.2.1+cu121
torchvision==0.17.1+cu121
numpy
opencv-python
matplotlib
tqdm
```

## 3. データセット準備

1. Kaggle など CC0 (public‑domain) 顔検出データセットをダウンロード。
2. 画像を `data/images/train/`, `data/images/val/` にコピー。
3. アノテーションを YOLO 形式 (1 行 = class cx cy w h) に変換し、対応するパスで `data/labels/...` へ保存。

**変換スクリプト例:**
```python
for img_path, boxes in your_dataset:
    txt_path = Path(img_path).with_suffix('.txt').replace('images','labels')
    with open(txt_path,'w') as f:
        for (x1,y1,x2,y2) in boxes:
            cx = (x1+x2)/2/W; cy=(y1+y2)/2/H; w=(x2-x1)/W; h=(y2-y1)/H
            f.write(f"0 {cx} {cy} {w} {h}\n")
```

## 4. 学習スクリプト (train.py 概要)

```python
from utils.dataloader import create_dataloader
from face_yolomini import FaceYOLOMini, Detect
import torch, torch.optim as optim

model = FaceYOLOMini().cuda()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
train_loader = create_dataloader('data/images/train', 'data/labels/train')

for epoch in range(50):
    for imgs, targets in train_loader:  # imgs: (bs,3,640,640)
        preds, grid, stride = model(imgs.cuda())
        loss = compute_loss(preds, targets.cuda())  # DFL+CIoU+BCE
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
    torch.save(model.state_dict(), f'runs/epoch{epoch}.pt')
```

> compute_loss は YOLOv8 論文のロジックを自前実装してください。

## 5. 推論テスト

```python
from face_yolomini import FaceYOLOMini, decode
import cv2, torch

model = FaceYOLOMini().eval()
model.load_state_dict(torch.load('runs/epoch49.pt', map_location='cpu'))
img = cv2.imread('test.jpg')
result = decode(model, img)
vis = result.plot(img)
cv2.imshow('det', vis); cv2.waitKey(0)
```

## 6. よくあるエラー

| 症状 | 原因 | 対策 |
|------|------|------|
| shape mismatch | アノテーション数ずれ | 画像に対応する txt の行数確認 |
| CUDA out of memory | GPU VRAM 足りない | batch=4→2 に減らす / 入力サイズ 512² |
| loss = nan | 学習率高すぎ | lr=1e-4 へ下げる |

## 7. ライセンス備考

- **コード・重み**: MIT で自由に再配布可
- **データ**: CC0 につき二次配布自由。ただし人物の肖像権に配慮し、プライバシー規約を整備してください

---

以上でセットアップ完了です。ご不明点やトラブルがあればお気軽にご相談ください！

