from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from ultralytics import YOLO

@dataclass
class Detection: #container object to store detection
    cls_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]

class PPEDetector:
    """
    Wraps a YOLO model and returns clean detections for downstream rules + Grok.
    """

    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25) -> List[Detection]:
        results = self.model.predict(image_bgr, conf=conf, verbose=False)
        r0 = results[0]

        names = r0.names  # class id -> name
        dets: List[Detection] = []

        if r0.boxes is None or len(r0.boxes) == 0:
            return dets

        boxes = r0.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = str(names[cls_id])
            c = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            dets.append(Detection(cls_name=cls_name, conf=c, xyxy=(x1, y1, x2, y2)))

        return dets