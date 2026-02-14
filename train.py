import os
from ultralytics import YOLO

os.environ["ULTRALYTICS_DATASETS_DIR"] = r"C:\Users\ayans\VSCodeProjects\Construction-RIsk-Analyzer\hackathon\datasets"

model = YOLO("yolo11n.pt")

model.train(
    data="construction-ppe-local.yaml",
    epochs=8,
    imgsz=512,
    batch=4,
    workers=0,
    device="cpu",
    patience=3,
    project=r"C:\Users\ayans\VSCodeProjects\Construction-RIsk-Analyzer\hackathon\runs",
    name="ppe_fast",
    pretrained=True,
    freeze=10
)