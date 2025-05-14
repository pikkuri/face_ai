# -*- coding: utf-8 -*-
"""face_yolomini.py  ⭐ 完全版 ⭐
--------------------------------------------------
YOLOv8 風アーキテクチャを“白紙から”実装した **顔検出器 (1‑class)**。

* 入力 : RGB 画像 (Tensor[N,C,H,W] または numpy[H,W,C])
* 出力 : `results` オブジェクト  ── Ultralytics YOLO の API 風
    - `.boxes.xyxy`  (FloatTensor[N,4])  左上・右下座標
    - `.boxes.xywh`  (FloatTensor[N,4])  中心+幅高
    - `.boxes.conf`  (FloatTensor[N])    confidence = obj * cls_prob
    - `.boxes.cls`   (IntTensor[N])      クラス番号 (0 固定)
    - `.names`       List[str]           クラス名 ['face']
    - `.plot()`      ⇒ 注釈付き画像 (np.ndarray)

ライセンス : MIT  (コード・重みとも制限なし)
依   存   : torch ≥2.0, torchvision, numpy, opencv‑python (plot)
"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2

# --------------------------------------------------
# 1. ネットワーク部品 (Conv, C2f, Detect)
# --------------------------------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, k//2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act= nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class C2f(nn.Module):
    """縮小版 C2f (YOLOv8 参照)"""
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m   = nn.ModuleList(DWConv(c2, c2) for _ in range(n))
        self.cv2 = Conv(c2*(n+1), c2, 1, 1)
    def forward(self, x):
        y = [self.cv1(x)]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class Detect(nn.Module):
    """Anchor‑Free Head (4*reg_max + obj + cls)"""
    def __init__(self, nc=1, ch=(64,128,256), reg_max=16):
        super().__init__()
        self.nc, self.reg_max = nc, reg_max
        self.no = nc + 5 + 4*reg_max
        self.stride = torch.tensor([8,16,32])
        self.cv = nn.ModuleList(nn.Conv2d(c, self.no, 1) for c in ch)
        self.register_buffer('proj', torch.linspace(0, reg_max-1, reg_max))
        self.names = ['face']

    def forward(self, feats: List[torch.Tensor]):
        bs, device = feats[0].shape[0], feats[0].device
        outputs, grids, strides = [], [], []
        for i, x in enumerate(feats):
            h, w = x.shape[2:]
            y = self.cv[i](x)
            y = y.view(bs, self.nc+5+4*self.reg_max, h*w).permute(0,2,1).contiguous()  # bs,HW,C
            outputs.append(y)
            # grid coords
            gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
            grid = torch.stack((gx,gy), 2).view(-1,2)  # HW,2
            grids.append(grid)
            strides.append(torch.full((h*w,1), self.stride[i], device=device))
        return torch.cat(outputs,1), torch.cat(grids,0), torch.cat(strides,0)  # bs,N,C / N,2 / N,1

# --------------------------------------------------
# 2. ネットワーク全体
# --------------------------------------------------
class FaceYOLOMini(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone (浅め + 640 入力推奨)
        self.layer0 = Conv(3,32,3,2)
        self.layer1 = C2f(32,32,2)
        self.layer2 = Conv(32,64,3,2)
        self.layer3 = C2f(64,64,4)
        self.layer4 = Conv(64,128,3,2)
        self.layer5 = C2f(128,128,6)
        self.layer6 = Conv(128,256,3,2)
        self.layer7 = C2f(256,256,6)
        self.layer8 = Conv(256,512,3,2)
        self.layer9 = C2f(512,512,2)
        # Neck (FPN 三層)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lat_c4 = Conv(512,256,1,1)
        self.out_p4 = Conv(256,128,1,1)
        self.out_p3 = Conv(128,64,1,1)
        self.detect = Detect()

    # -------- forward returns Predictions + meta --------
    def forward(self, x):
        x = self.layer0(x); x = self.layer1(x)
        x = self.layer2(x); x = self.layer3(x)
        p3 = self.layer5(self.layer4(x))     # stride 8
        p4 = self.layer7(self.layer6(p3))    # stride 16
        p5 = self.layer9(self.layer8(p4))    # stride 32
        n4 = self.lat_c4(p5)
        n3 = self.out_p4(self.upsample(n4) + p4)
        n2 = self.out_p3(self.upsample(n3) + p3)
        preds, grid, stride = self.detect([n2,n3,n4])
        return preds, grid, stride

# --------------------------------------------------
# 3. Decoder → Results API 互換
# --------------------------------------------------
class FaceYOLOResult:
    def __init__(self, boxes: torch.Tensor, names: List[str]):
        self.boxes = boxes  # type: ignore
        self.names = names

    def plot(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        for (x1,y1,x2,y2,c,cls) in self.boxes.xyxy.cpu().numpy():
            cv2.rectangle(out,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(out,f"face {c:.2f}",(int(x1),int(y1)-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        return out

class _Boxes:
    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
        self._xyxy = xyxy  # (N,4)
        self._conf = conf  # (N,)
        self._cls  = cls   # (N,)
    @property
    def xyxy(self): return self._xyxy
    @property
    def conf(self): return self._conf
    @property
    def cls(self): return self._cls
    @property
    def xywh(self):
        x1,y1,x2,y2 = self._xyxy.split(1,1)
        w,h = x2-x1, y2-y1
        cx,cy = x1+0.5*w, y1+0.5*h
        return torch.cat([cx,cy,w,h],1)

@torch.no_grad()
def decode(model: FaceYOLOMini, img: np.ndarray, conf_th=0.25, iou_th=0.45) -> FaceYOLOResult:
    """Run single image inference and return YOLO‑like result"""
    device = next(model.parameters()).device
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (640,640))
    im = torch.from_numpy(im).permute(2,0,1).float().div(255)[None].to(device)
    with torch.inference_mode():
        preds, grid, stride = model(im)
    bs, n, c = preds.shape
    reg_max = model.detect.reg_max
    box_raw = preds[0,:,:4]          # l,t,r,b in grid coords
    obj     = preds[0,:,4:5]
    cls_p   = preds[0,:,5]
    # decode box
    xy = (grid + 0.5 - torch.stack((box_raw[:,0],box_raw[:,1]),1)) * stride  # top‑left
    wh = (box_raw[:,2:4] + 0.5) * stride
    xyxy = torch.cat([xy, xy+wh],1)
    conf = (obj.squeeze()*cls_p).sigmoid()
    keep = conf>conf_th
    xyxy, conf = xyxy[keep], conf[keep]
    # torchvision.ops.nms for IoU suppression
    import torchvision
    keep_idx = torchvision.ops.nms(xyxy, conf, iou_th)
    xyxy, conf = xyxy[keep_idx], conf[keep_idx]
    cls = torch.zeros_like(conf, dtype=torch.int64)
    boxes = _Boxes(xyxy, conf, cls)
    return FaceYOLOResult(boxes, model.detect.names)

# ---------------- test ----------------
if __name__ == "__main__":
    model = FaceYOLOMini().eval()
    img = np.zeros((480,640,3), dtype=np.uint8)
    res = decode(model, img)
    print("#faces", res.boxes.xyxy.shape[0])
