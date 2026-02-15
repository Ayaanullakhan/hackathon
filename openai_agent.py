import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
        raise

def generate_audit_report(payload: dict) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    client = OpenAI(
        api_key=api_key
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are a construction safety compliance auditor. Return ONLY valid JSON. No extra text."
            },
            {
                "role": "user",
                "content": json.dumps(payload)
            }
        ],
        temperature=0
    )

    return _extract_json(resp.output_text)
