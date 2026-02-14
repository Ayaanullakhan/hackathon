from __future__ import annotations
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ppe_detector import PPEDetector
from compliance_rules import evaluate_compliance
from grok_agent import generate_audit_report


st.set_page_config(page_title="Site risk Spotter", layout="wide")
st.title("Site risk Spotter")
st.caption("Upload a construction site image → detect PPE → audit compliance → generate an actionable checklist.")


@st.cache_resource
def load_detector():
    # trained weights are put here
    return PPEDetector(weights_path="best.pt")


detector = load_detector()

left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    required = st.multiselect(
        "Required PPE policy",
        options=["hardhat", "vest", "goggles", "gloves", "boots", "mask"],
        default=["hardhat", "vest"],
    )
    conf = st.slider("Detection confidence threshold", 0.1, 0.9, 0.25, 0.05)

if uploaded is None:
    st.stop()

img = Image.open(uploaded).convert("RGB")
img_np = np.array(img)  # RGB
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

dets = detector.predict(img_bgr, conf=conf)

label_map = {
    "helmet": "hardhat",
    "hardhat": "hardhat",
    "safety vest": "vest",
    "vest": "vest",
    "goggles": "goggles",
    "gloves": "gloves",
    "boots": "boots",
    "mask": "mask",
}

people = evaluate_compliance(dets, required_ppe=required, label_map=label_map)

# Draw boxes
draw = img_bgr.copy()
for d in dets:
    x1, y1, x2, y2 = d.xyxy
    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(draw, f"{d.cls_name} {d.conf:.2f}", (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

with right:
    st.image(draw_rgb, caption="Detections", use_container_width=True)

st.subheader("Per-person compliance")
if len(people) == 0:
    st.warning("No people detected. Try a different image or lower the confidence threshold.")
    st.stop()

for i, p in enumerate(people, start=1):
    miss = [k for k, v in p.missing.items() if v]
    st.write(f"Person {i}: **{p.score}%** | Risk: **{p.severity}** | Missing: {', '.join(miss) if miss else 'None'}")
    with st.expander("Evidence"):
        for e in p.evidence:
            st.write("•", e)

if st.button("Generate audit report (Grok)"):
    payload = {
        "site_policy": required,
        "per_person": [
            {
                "person_id": i,
                "compliance_percent": p.score,
                "risk_level": p.severity,
                "missing_ppe": [k for k, v in p.missing.items() if v],
                "evidence": p.evidence,
            }
            for i, p in enumerate(people, start=1)
        ],
        "global_notes": "This is an automated audit based on visual detections.",
    }

    report = generate_audit_report(payload)

    st.subheader("Audit summary")
    st.json(report)
