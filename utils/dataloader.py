import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        
        # 画像ファイルのリストを取得
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 対応するラベルファイルが存在することを確認
        self.label_files = []
        for img_file in self.img_files:
            base_name = os.path.basename(img_file)
            name_without_ext = os.path.splitext(base_name)[0]
            label_file = os.path.join(label_dir, name_without_ext + '.txt')
            if os.path.exists(label_file):
                self.label_files.append(label_file)
            else:
                # ラベルファイルが見つからない場合、対応する画像も除外
                self.img_files.remove(img_file)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 画像読み込み
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ラベル読み込み
        label_path = self.label_files[idx]
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) == 5:  # class, cx, cy, w, h
                        labels.append([float(v) for v in values])
        
        labels = np.array(labels) if len(labels) > 0 else np.zeros((0, 5))
        
        # リサイズ
        h, w = img.shape[:2]
        ratio = self.img_size / max(h, w)
        if ratio != 1:
            img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        
        # パディング
        new_h, new_w = img.shape[:2]
        pad_h, pad_w = self.img_size - new_h, self.img_size - new_w
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=114)
        
        # 画像拡張がある場合は適用
        if self.augment and len(labels) > 0:
            img, labels = self.augment(img, labels)
        
        # 正規化とテンソル変換
        img = img.transpose((2, 0, 1)) / 255.0
        img = torch.from_numpy(img).float()
        
        # ラベルをテンソルへ変換
        labels = torch.from_numpy(labels).float()
        
        return img, labels

def create_dataloader(img_dir, label_dir, batch_size=4, img_size=640, augment=None, shuffle=True, num_workers=4):
    """
    YOLOデータセット用のDataLoaderを作成
    
    Parameters:
    - img_dir: 画像ディレクトリのパス
    - label_dir: ラベルディレクトリのパス
    - batch_size: バッチサイズ
    - img_size: 画像サイズ（正方形）
    - augment: 画像拡張関数（Noneの場合は拡張なし）
    - shuffle: シャッフルするかどうか
    - num_workers: DataLoaderの並列ワーカー数
    
    Returns:
    - DataLoader: PyTorchのデータローダー
    """
    dataset = YOLODataset(img_dir, label_dir, img_size=img_size, augment=augment)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return loader

def collate_fn(batch):
    """バッチ内の可変長データを処理するための関数"""
    imgs, labels = zip(*batch)
    
    # 画像をスタック
    imgs = torch.stack(imgs)
    
    # ラベルはバッチインデックスを追加
    batch_labels = []
    for i, l in enumerate(labels):
        if l.shape[0] > 0:
            # バッチインデックスを先頭に追加
            batch_idx = torch.full((l.shape[0], 1), i, dtype=l.dtype, device=l.device)
            batch_labels.append(torch.cat([batch_idx, l], dim=1))
    
    # バッチラベルを結合
    if len(batch_labels) > 0:
        batch_labels = torch.cat(batch_labels, 0)
    else:
        # 空のラベルの場合
        batch_labels = torch.zeros((0, 6))
    
    return imgs, batch_labels
