import cv2
import numpy as np
import random

def mosaic_augment(imgs, labels, img_size=640):
    """
    4つの画像を組み合わせてモザイク拡張を行う
    
    Parameters:
    - imgs: 画像のリスト [img1, img2, img3, img4]
    - labels: 各画像のラベルのリスト [labels1, labels2, labels3, labels4]
    - img_size: 出力画像サイズ
    
    Returns:
    - mosaic_img: モザイク画像
    - mosaic_labels: 変換後のラベル
    """
    assert len(imgs) == 4, "モザイク拡張には4つの画像が必要です"
    
    # モザイク中心点
    xc, yc = [int(random.uniform(img_size * 0.25, img_size * 0.75)) for _ in range(2)]
    
    # 出力画像
    mosaic_img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    
    # 境界と変換後のラベルを保存
    mosaic_labels = []
    
    # 4つの位置に画像を配置
    for i, (img, label) in enumerate(zip(imgs, labels)):
        h, w = img.shape[:2]
        
        # モザイクの4つの位置を決定
        if i == 0:  # 左上
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # 右上
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_size, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # 右下
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size), min(img_size, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        
        # モザイク画像に配置
        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        
        # ラベルの座標を調整
        if len(label) > 0:
            # ラベルは [class, cx, cy, w, h] 形式
            # cx, cy, w, h を絶対座標に変換
            label_copy = label.copy()
            
            # 元画像内の相対座標から絶対座標に変換
            label_copy[:, 1] = w * (label[:, 1] - 0.5 * label[:, 3]) + x1b
            label_copy[:, 2] = h * (label[:, 2] - 0.5 * label[:, 4]) + y1b
            label_copy[:, 3] = w * label[:, 3]
            label_copy[:, 4] = h * label[:, 4]
            
            # モザイク画像内の座標に変換
            label_copy[:, 1] = (label_copy[:, 1] - x1b) + x1a
            label_copy[:, 2] = (label_copy[:, 2] - y1b) + y1a
            
            # モザイク画像内の相対座標に戻す
            label_copy[:, 1] = label_copy[:, 1] / img_size
            label_copy[:, 2] = label_copy[:, 2] / img_size
            label_copy[:, 3] = label_copy[:, 3] / img_size
            label_copy[:, 4] = label_copy[:, 4] / img_size
            
            # 画像内に収まるボックスのみ保持
            mask = (
                (label_copy[:, 1] > 0) & 
                (label_copy[:, 2] > 0) & 
                (label_copy[:, 1] < 1) & 
                (label_copy[:, 2] < 1) &
                (label_copy[:, 3] > 0) & 
                (label_copy[:, 4] > 0)
            )
            
            mosaic_labels.append(label_copy[mask])
    
    if len(mosaic_labels):
        mosaic_labels = np.concatenate(mosaic_labels, 0)
    else:
        mosaic_labels = np.zeros((0, 5))
    
    return mosaic_img, mosaic_labels

def random_perspective(img, labels, degrees=10, translate=0.1, scale=0.1, shear=10):
    """
    ランダムな透視変換を適用
    
    Parameters:
    - img: 入力画像
    - labels: YOLO形式のラベル [class, cx, cy, w, h]
    - degrees: 回転の最大角度
    - translate: 平行移動の最大比率
    - scale: スケーリングの最大比率
    - shear: せん断の最大角度
    
    Returns:
    - img: 変換後の画像
    - labels: 変換後のラベル
    """
    height, width = img.shape[:2]
    
    # 中心、平行移動、スケール
    C = np.eye(3)
    C[0, 2] = -width / 2  # x中心への平行移動
    C[1, 2] = -height / 2  # y中心への平行移動
    
    # 透視変換行列
    P = np.eye(3)
    
    # 回転と拡大縮小
    R = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    scale = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
    
    # せん断
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    # 平行移動
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # 変換行列の合成
    M = T @ S @ R @ C
    
    # 画像に変換を適用
    img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
    
    # ラベルにも変換を適用
    n = len(labels)
    if n:
        # YOLO形式からxyxyに変換
        box = np.zeros((n, 4))
        box[:, 0] = labels[:, 1] - labels[:, 3] / 2  # 左上x
        box[:, 1] = labels[:, 2] - labels[:, 4] / 2  # 左上y
        box[:, 2] = labels[:, 1] + labels[:, 3] / 2  # 右下x
        box[:, 3] = labels[:, 2] + labels[:, 4] / 2  # 右下y
        
        # 絶対座標に変換
        box[:, [0, 2]] *= width
        box[:, [1, 3]] *= height
        
        # 拡張ポイントを作成
        points = np.ones((n * 4, 3))
        points[:, :2] = box[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        
        # 透視変換を適用
        points = (points @ M.T)
        points = points[:, :2].reshape(n, 8)
        
        # 新しいボックスの座標を計算
        x1 = np.min(points[:, [0, 2, 4, 6]], axis=1)
        y1 = np.min(points[:, [1, 3, 5, 7]], axis=1)
        x2 = np.max(points[:, [0, 2, 4, 6]], axis=1)
        y2 = np.max(points[:, [1, 3, 5, 7]], axis=1)
        
        # YOLOフォーマットに戻す
        labels[:, 1] = ((x1 + x2) / 2) / width
        labels[:, 2] = ((y1 + y2) / 2) / height
        labels[:, 3] = (x2 - x1) / width
        labels[:, 4] = (y2 - y1) / height
        
        # 画像内に収まるボックスのみ保持
        mask = (
            (labels[:, 1] > 0) & 
            (labels[:, 2] > 0) & 
            (labels[:, 1] < 1) & 
            (labels[:, 2] < 1) &
            (labels[:, 3] > 0) & 
            (labels[:, 4] > 0)
        )
        labels = labels[mask]
    
    return img, labels

def augment_hsv(img, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """
    HSV色空間での色調増強
    
    Parameters:
    - img: 入力画像
    - h_gain: 色相の変化量
    - s_gain: 彩度の変化量
    - v_gain: 明度の変化量
    
    Returns:
    - img: 増強された画像
    """
    if h_gain or s_gain or v_gain:
        # HSVに変換
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # ランダムなゲインを生成
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        
        # HSV画像に適用
        img_hsv = img_hsv * r
        
        # 値のクリッピング
        img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 179)
        img_hsv[:, :, 1:] = np.clip(img_hsv[:, :, 1:], 0, 255)
        
        # RGBに戻す
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return img

def apply_augmentation(img, labels, p_mosaic=0.5, p_hsv=0.5, p_perspective=0.5):
    """
    複数の拡張処理を確率的に適用する
    
    Parameters:
    - img: 入力画像
    - labels: YOLO形式のラベル [class, cx, cy, w, h]
    - p_mosaic: モザイク拡張を適用する確率
    - p_hsv: HSV拡張を適用する確率
    - p_perspective: 透視変換を適用する確率
    
    Returns:
    - img: 増強された画像
    - labels: 変換後のラベル
    """
    # HSV拡張
    if random.random() < p_hsv:
        img = augment_hsv(img)
    
    # 透視変換
    if random.random() < p_perspective:
        img, labels = random_perspective(img, labels)
    
    return img, labels
