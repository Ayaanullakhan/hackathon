from __future__ import annotations 
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def build_client() -> OpenAI:
    base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing XAI_API_KEY in environment (.env).")
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_audit_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload should include:
      - site_policy: required PPE list
      - per_person: compliance results + evidence
      - global_notes: optional
    Returns structured JSON for UI rendering.
    """
    model = os.getenv("XAI_MODEL", "grok-beta")
    client = build_client()

    system = (
        "You are a construction safety compliance auditor.\n"
        "You MUST only use the provided detections, scores, and evidence.\n"
        "If something is not supported by evidence, write 'UNKNOWN'.\n"
        "Return ONLY valid JSON matching the schema."
    )

    schema = {
        "summary": {
            "overall_compliance_percent": "number 0-100",
            "overall_risk_level": "LOW|MEDIUM|HIGH",
            "top_issues": ["string (max 4)"],
        },
        "people": [
            {
                "person_id": "int starting at 1",
                "compliance_percent": "number 0-100",
                "risk_level": "LOW|MEDIUM|HIGH",
                "missing_ppe": ["string"],
                "actions": ["string (max 3)"],
                "evidence": ["string (short)"],
            }
        ],
        "checklist": ["string (max 8)"],
        "disclaimer": "string (short)"
    }

    user = {
        "input_payload": payload,
        "required_output_schema": schema,
        "scoring_rules": (
            "overall_compliance_percent = average of per-person compliance_percent.\n"
            "overall_risk_level: HIGH if any person risk_level is HIGH, else MEDIUM if any MEDIUM, else LOW."
        ),
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user)},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content
    # Grok should return JSON only, so we can parse it in the UI.
    import json
    return json.loads(text)
