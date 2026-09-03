"""
Core capsule agent loop: retrieve relevant scars -> reason -> decide -> record.
"""
import os
from groq import Groq
from dotenv import load_dotenv
from memory_client import find_relevant_scars, record_decision

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-20b"  # confirmed live via client.models.list() - Sep 2026


def build_prompt(situation, scars):
    scar_block = ""
    for s in scars:
        b = s["body"]
        scar_block += (
            f"\n- SCAR [{b['id']}] (status: {b['status']}, confidence: {b['confidence']})\n"
            f"  Trigger: {b['trigger']}\n"
            f"  Failed action: {b['action']}\n"
            f"  Why it failed: {b['root_cause']}\n"
        )

    if not scar_block:
        scar_block = "\n(no relevant scars found - this is unfamiliar territory)\n"

    return f"""You are a troubleshooting assistant with persistent memory of past failures.

Current situation:
{situation}

Relevant memory from past incidents:{scar_block}

Instructions:
- If an ACTIVE scar's failed action matches what you'd naturally try first, state clearly that you're avoiding it and name the scar id. Do not restate the fix in your own words - it will be shown separately from verified memory.
- If a scar's status is "overridden", you may still consider that action, but mention the past conflicting evidence.
- If no scar applies, reason normally and recommend a diagnostic first step.
- Be concise: 2-3 sentences max.
"""


def handle_situation(situation):
    scars = find_relevant_scars(situation)
    prompt = build_prompt(situation, scars)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    reasoning = response.choices[0].message.content

    blocked_scar_ids = [
        s["body"]["id"] for s in scars if s["body"]["status"] == "active"
    ]

    record_decision(
        trigger=situation,
        action_chosen=reasoning,
        blocked_scar_id=blocked_scar_ids if blocked_scar_ids else None,
    )

    print(f"\n{'='*60}")
    print(f"SITUATION: {situation}")
    print(f"{'='*60}")
    print(f"Scars consulted: {[s['body']['id'] for s in scars] or 'none'}")
    print(f"\nAGENT REASONING:\n{reasoning}\n")

    if blocked_scar_ids:
        print("--- Verified fix, pulled directly from memory (not model-generated) ---")
        for sid in blocked_scar_ids:
            scar_body = next(s["body"] for s in scars if s["body"]["id"] == sid)
            print(f"  [{sid}] {scar_body['real_fix']}")

    return reasoning


if __name__ == "__main__":
    handle_situation(
        "My llama-cpp-python server won't accept --tensor_split no matter how I format it."
    )
