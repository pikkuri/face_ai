# -*- coding: utf-8 -*-
"""
frontal_face_tracker.py
======================

リアルタイムで *正面顔* だけを対象に人物を追跡し、人物が画面から離脱したタイミングで
取得したメタデータをバッチ処理用キュー (archived) に送るサンプル実装です。

💡 **構成**

1. **人物検出**    : Ultralytics YOLOv8 (class == person)
2. **顔検出**      : OpenCV YuNet (軽量・正面顔のランドマークが取れる)
3. **顔特徴抽出**  : OpenCV FaceRecognizerSF (ArcFace)  
                     *512-dim* L2 Normalised Embeddings
4. **追跡管理**    : 独自 Track クラス + Faiss (オプション) による高速類似検索
5. **正面フィルタ**: ランドマークで yaw / roll を推定し閾値内だけ採用

⚠️ **ライセンス注意**
- YOLOv8 はデフォルト GPL‑3。閉源/商用の場合は Ultralytics 商用ライセンスを取得してください。
- OpenCV (YuNet / FaceRecognizerSF) は BSD + MIT でパーミッシブ。

 Tested with:
    Python 3.10 / OpenCV 4.9.0 / Ultralytics 8.2.8 / faiss‑cpu 1.7.4
"""
from __future__ import annotations

import cv2
import numpy as np
import time
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

# --------------------------------------------------
# Optional: use Faiss if available for ANN search
# --------------------------------------------------
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    from sklearn.neighbors import KDTree  # fallback

# --------------------------------------------------
# Configuration parameters (tweak as needed)
# --------------------------------------------------
CONF_PERSON = 0.25         # YOLO person detection score threshold
FACE_YAW_TH = 20.0         # |yaw| < 20° なら正面顔とみなす
FACE_ROLL_TH = 15.0        # |roll| < 15°
T_LOST_SEC = 2.0           # last_seen から Lost とみなす秒数
T_ARCHIVE_SEC = 6.0        # last_seen から Archive に移行する秒数
EMBED_TH = 0.45            # cos 距離でマッチとみなす閾値 (ArcFace 推奨)

# --------------------------------------------------
# Helper dataclass for each tracked person
# --------------------------------------------------
@dataclass
class Track:
    tid: int
    embedding: np.ndarray        # (512, ) unit vector
    bbox: Tuple[int, int, int, int]  # xywh (last)
    last_seen: float
    state: str = "Active"        # Active | Lost | Archived
    metadata: Dict = field(default_factory=dict)

    def update(self, embedding: np.ndarray, bbox: Tuple[int, int, int, int], ts: float):
        """Update embedding (EMA) & bounding box, refresh timestamp"""
        # simple exponential moving average to smooth embedding drift
        self.embedding = 0.5 * self.embedding + 0.5 * embedding
        self.embedding /= np.linalg.norm(self.embedding) + 1e-8
        self.bbox = bbox
        self.last_seen = ts
        self.state = "Active"

# --------------------------------------------------
# Tracker class that wraps detection, face extraction, matching & housekeeping
# --------------------------------------------------
class FrontalFaceTracker:
    def __init__(self):
        # ID generator
        self._id_gen = itertools.count(start=1)
        # Active / Lost tracks
        self.live: Dict[int, Track] = {}
        # Archived (finished) tracks queued for offline processing
        self.archived: deque[Track] = deque()
        # ANN index init
        self._init_index()

        # ----- Model initialisation -----
        # 1. Person detector (YOLOv8n by default)
        from ultralytics import YOLO  # lazy import
        self.detector = YOLO("yolov8n.pt")  # path or model name

        # 2. YuNet face detector & landmark predictor
        model_path = cv2.samples.findFile("face_detection_yunet_2023mar.onnx")
        self.face_det = cv2.FaceDetectorYN.create(
            model_path, "", (640, 640), score_threshold=0.9, nms_threshold=0.3, top_k=500
        )

        # 3. Face recognizer (ArcFace)
        rec_model = cv2.FaceRecognizerSF.create(
            cv2.samples.findFile("face_recognition_sface_2021dec.onnx"), ""  # weights
        )
        self.face_recognizer = rec_model

    # ----------------------------------------------
    # ANN index helpers
    # ----------------------------------------------
    def _init_index(self):
        self._dim = 512
        if _FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self._dim)  # inner prod == cosine on L2‑normed vect
        else:
            self.index = None  # will build KDTree on demand
        self._id2idx: Dict[int, int] = {}   # track id -> faiss index row
        self._embeddings: List[np.ndarray] = []  # fallback storage for KDTree

    def _add_to_index(self, track_id: int, emb: np.ndarray):
        if _FAISS_AVAILABLE:
            self.index.add(emb[np.newaxis, :].astype(np.float32))
            self._id2idx[track_id] = self.index.ntotal - 1
        else:
            self._embeddings.append(emb)

    def _remove_from_index(self, track_id: int):
        if _FAISS_AVAILABLE:
            idx = self._id2idx.pop(track_id, None)
            if idx is not None:
                # Faiss には行削除 API が無いので lazy delete: flag as NaN
                self.index.reconstruct(idx)[:] = np.nan
        else:
            # rebuild KDTree lazily on next search
            pass

    def _search(self, emb: np.ndarray, k: int = 1):
        if _FAISS_AVAILABLE and self.index.ntotal > 0:
            D, I = self.index.search(emb[np.newaxis, :].astype(np.float32), k)
            return I[0], 1 - D[0]  # cosine distance = 1 - inner product
        elif not _FAISS_AVAILABLE and self._embeddings:
            X = np.vstack(self._embeddings)
            tree = KDTree(X)
            dist, ind = tree.query(emb[np.newaxis, :], k)
            return ind[0], dist[0]
        return [], []

    # ----------------------------------------------
    # Geometry helper: frontal face filter
    # ----------------------------------------------
    @staticmethod
    def _is_frontal(landmarks: np.ndarray) -> bool:
        """Estimate yaw & roll from 5p landmarks and judge if frontal"""
        # landmarks order (YuNet): [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
        le, re = landmarks[0], landmarks[1]
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        roll = np.degrees(np.arctan2(dy, dx))
        # yaw proxy: nose x ‑ center of eyes line
        center_x = (le[0] + re[0]) / 2
        nose_x = landmarks[2][0]
        yaw = np.degrees(np.arctan2(nose_x - center_x, abs(dx))) * 2.0
        return abs(yaw) < FACE_YAW_TH and abs(roll) < FACE_ROLL_TH

    # ----------------------------------------------
    # Main update call per frame
    # ----------------------------------------------
    def update(self, frame: np.ndarray):
        ts = time.time()
        H, W = frame.shape[:2]

        # 1. Person detection (YOLO)
        detections = self.detector(frame, verbose=False)[0]
        persons = [b for b in detections.boxes if int(b.cls) == 0 and b.conf > CONF_PERSON]

        # 2. Iterate persons and apply YuNet on cropped region
        embeddings: List[Tuple[int, np.ndarray, Tuple[int, int, int, int]]] = []
        for det in persons:
            x1, y1, x2, y2 = map(int, det.xyxy)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # YuNet expects resized image size registered at create()
            self.face_det.setInputSize((crop.shape[1], crop.shape[0]))
            ok, faces = self.face_det.detect(crop)
            if not ok or faces is None:
                continue
            for f in faces:
                # f: [x, y, w, h, score, lms(10)]
                if f[4] < 0.9:
                    continue
                lx, ly, lw, lh = map(int, f[:4])
                lm = f[5:].reshape(-1, 2)
                if not self._is_frontal(lm):
                    continue
                face_img = crop[ly:ly+lh, lx:lx+lw]
                if face_img.size == 0:
                    continue
                face_img = cv2.resize(face_img, (112, 112))  # ArcFace input
                # embedding (L2 normalised)
                emb = self.face_recognizer.feature(face_img)
                emb = emb.flatten()
                emb /= np.linalg.norm(emb) + 1e-9
                # global bbox xywh
                gx, gy, gw, gh = x1+lx, y1+ly, lw, lh
                embeddings.append((0, emb, (gx, gy, gw, gh)))  # pid unused here

        # 3. Matching each embedding to existing tracks
        for _, emb, bbox in embeddings:
            match_ids, dists = self._search(emb)
            matched = False
            if match_ids:
                tid = list(self.live.keys())[match_ids[0]] if _FAISS_AVAILABLE else list(self.live.keys())[match_ids[0]]
                if tid in self.live and dists[0] < EMBED_TH:
                    self.live[tid].update(emb, bbox, ts)
                    matched = True
            if not matched:
                tid = next(self._id_gen)
                tr = Track(tid, emb, bbox, ts)
                self.live[tid] = tr
                self._add_to_index(tid, emb)

        # 4. Timeout handling
        for tid, tr in list(self.live.items()):
            dt = ts - tr.last_seen
            if dt > T_ARCHIVE_SEC:
                tr.state = "Archived"
                self.archived.append(tr)
                del self.live[tid]
                self._remove_from_index(tid)
            elif dt > T_LOST_SEC:
                tr.state = "Lost"

    # ----------------------------------------------
    # Utility: draw tracks for visual debug
    # ----------------------------------------------
    def draw(self, frame: np.ndarray):
        for tr in self.live.values():
            x, y, w, h = tr.bbox
            color = (0, 255, 0) if tr.state == "Active" else (0, 128, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID:{tr.tid}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --------------------------------------------------
# Entry point (camera demo)
# --------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = FrontalFaceTracker()
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tracker.update(frame)
        vis = tracker.draw(frame.copy())
        cv2.imshow("Frontal Face Tracker", vis)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    # Offline processing example
    print(f"Archived tracks: {len(tracker.archived)}")
    for tr in tracker.archived:
        print(f"Track {tr.tid}: last seen {time.ctime(tr.last_seen)}")
