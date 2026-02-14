from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
from ppe_detector import Detection


@dataclass
class PersonCompliance:
    person_box: Tuple[int, int, int, int]
    present: Dict[str, bool]          # e.g. {"hardhat": True, "vest": False}
    missing: Dict[str, bool]          # e.g. {"hardhat": False, "vest": True}
    score: int                        # 0-100
    severity: str                     # "LOW" | "MEDIUM" | "HIGH"
    evidence: List[str]               # short rule evidence strings


# def _iou(a, b) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1) #computing intersection rectangle
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1) #intersection widt/height
#     inter = iw * ih
#     if inter == 0:
#         return 0.0 #if they don't overlap width/height == 0
#     area_a = (ax2 - ax1) * (ay2 - ay1)
#     area_b = (bx2 - bx1) * (by2 - by1)
#     return inter / (area_a + area_b - inter + 1e-9)


# def assign_ppe_to_people(
#     detections: List[Detection],
#     person_label: str = "class6",
#     iou_threshold: float = 0.10,
# ) -> Dict[Tuple[int, int, int, int], List[Detection]]:
#     """
#     Groups PPE detections under the most overlapping person box.
#     """
#     people = [d for d in detections if d.cls_name.lower() == person_label]
#     others = [d for d in detections if d.cls_name.lower() != person_label]

#     mapping: Dict[Tuple[int, int, int, int], List[Detection]] = {p.xyxy: [] for p in people}

#     for det in others:
#         best_person = None
#         best_iou = 0.0
#         for p in people:
#             i = _iou(det.xyxy, p.xyxy)
#             if i > best_iou:
#                 best_iou = i
#                 best_person = p

#         if best_person is not None and best_iou >= iou_threshold:
#             mapping[best_person.xyxy].append(det)

#     return mapping

def _center_inside(child_box, parent_box, margin=15):
    x1, y1, x2, y2 = child_box
    px1, py1, px2, py2 = parent_box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (px1 - margin <= cx <= px2 + margin) and (py1 - margin <= cy <= py2 + margin)


def assign_ppe_to_people(detections, person_label="class6", margin=15):
    people = [d for d in detections if d.cls_name.lower() == person_label]
    others = [d for d in detections if d.cls_name.lower() != person_label]

    mapping = {p.xyxy: [] for p in people}

    for det in others:
        for p in people:
            if _center_inside(det.xyxy, p.xyxy, margin=margin):
                mapping[p.xyxy].append(det)
                break

    return mapping



def evaluate_compliance(
    detections: List[Detection],
    required_ppe: List[str],
    label_map: Dict[str, str] | None = None,
) -> List[PersonCompliance]:
    """
    required_ppe: list like ["hardhat", "vest", "goggles"]
    label_map: maps model's class names 
    """
    if label_map is None:
        label_map = {}

    grouped = assign_ppe_to_people(detections)
    results: List[PersonCompliance] = []

    for person_box, items in grouped.items():
        present = {k: False for k in required_ppe}
        evidence = []

        for it in items:
            raw = it.cls_name.lower().strip()
            canonical = label_map.get(raw, raw)  # if not mapped, use as-is
            if canonical in present:
                present[canonical] = True
                evidence.append(f"Detected {canonical} (conf={it.conf:.2f})")

        missing = {k: not v for k, v in present.items()}

        # scoring
        total = len(required_ppe)
        have = sum(1 for v in present.values() if v)
        score = int(round((have / max(1, total)) * 100))

        # severity: missing hardhat/vest tends to be higher
        high_impact = {"hardhat", "vest"}
        if any(missing.get(x, False) for x in high_impact):
            severity = "HIGH" if score < 80 else "MEDIUM"
        else:
            severity = "MEDIUM" if score < 100 else "LOW"

        if any(missing.values()):
            miss_list = [k for k, v in missing.items() if v]
            evidence.append(f"Missing: {', '.join(miss_list)}")

        results.append(
            PersonCompliance(
                person_box=person_box,
                present=present,
                missing=missing,
                score=score,
                severity=severity,
                evidence=evidence,
            )
        )

    return results
