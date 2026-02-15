from __future__ import annotations
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ppe_detector import PPEDetector
from compliance_rules import evaluate_compliance
from openai_agent import generate_audit_report


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
    "class6": "person",
    "class0": "hardhat",
    "class2": "vest",
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

if st.button("Generate audit report (OpenAI)"):
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


per_person = report.get("per_person", [])
site_policy = report.get("site_policy", [])

if per_person:
    overall_percent = round(sum(p.get("compliance_percent", 0) for p in per_person) / len(per_person))
    risk_levels = [p.get("risk_level", "HIGH") for p in per_person]
    overall_risk = "HIGH" if "HIGH" in risk_levels else ("MEDIUM" if "MEDIUM" in risk_levels else "LOW")

    missing_items = set()
    for p in per_person:
        missing_items.update([m.lower() for m in p.get("missing_ppe", [])])

else:
    overall_percent = 0
    overall_risk = "HIGH"
    missing_items = set()

st.subheader("Risk Narrative")

if overall_risk == "LOW":
    narrative = (
        f"Low risk: observed PPE compliance is strong at {overall_percent}%. "
        f"All detected personnel meet the current site policy ({', '.join(site_policy)}). "
        "Recommendation: maintain spot-checks and re-audit at shift change."
    )
elif overall_risk == "MEDIUM":
    narrative = (
        f"Medium risk: overall compliance is {overall_percent}%. "
        f"Some PPE gaps were detected ({', '.join(sorted(missing_items)) if missing_items else 'unspecified'}). "
        "Recommendation: correct immediately and re-scan the area."
    )
else:
    narrative = (
        f"High risk: overall compliance is {overall_percent}%. "
        f"Critical PPE gaps were detected ({', '.join(sorted(missing_items)) if missing_items else 'unspecified'}). "
        "Recommendation: pause work in the active zone until compliance is restored."
    )

st.write(narrative)

