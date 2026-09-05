"""
Core capsule agent loop:
  decide:  retrieve relevant scars -> reason -> pick an action -> record
  learn:   observe an outcome -> reinforce/override an existing scar,
           or create a brand new scar/ability from experience
"""
import os
from groq import Groq
from dotenv import load_dotenv
from memory_client import (
    find_relevant_scars,
    record_decision,
    bump_scar_evidence,
    create_scar,
    create_ability,
    get_scar,
    extract_keywords,
)

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-20b"  # confirmed live via client.models.list() - Sep 2026


def build_prompt(situation, scars):
    scar_block = ""
    for s in scars:
        b = s["body"]
        if b["status"] == "active":
            directive = "STILL BLOCKING - do not recommend the failed action below."
        elif b["status"] == "overridden":
            directive = "OVERRIDDEN - later evidence contradicted this. You MAY consider this action, but mention the past conflict."
        else:
            directive = ""
        scar_block += (
            f"\n- SCAR [{b['id']}] status={b['status']} ({directive})\n"
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
- Only avoid an action if its scar is explicitly marked STILL BLOCKING above. Do not treat OVERRIDDEN scars as blocking - that would contradict the memory record itself.
- If you avoid an action, name the scar id. Do not restate the fix in your own words - it will be shown separately from verified memory.
- If no scar applies, reason normally and recommend a diagnostic first step.
- Be concise: 2-3 sentences max.
"""


def handle_situation(situation):
    """Step 1: decide. Retrieves scars, asks the model to reason, prints the
    verified fix pulled directly from memory (never from the model's own text)."""
    scars = find_relevant_scars(situation)
    prompt = build_prompt(situation, scars)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    reasoning = response.choices[0].message.content

    blocked_scar_ids = [
        s["body"]["id"] for s in scars if s["body"]["status"] == "active"
    ]
    overridden_ids = [
        s["body"]["id"] for s in scars if s["body"]["status"] == "overridden"
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

    if overridden_ids:
        print("--- Overridden scars (no longer blocking, evidence contradicted them) ---")
        for sid in overridden_ids:
            print(f"  [{sid}] status changed after new evidence - action may be reconsidered")

    return scars, reasoning


def _action_matches_scar(action_taken, scar_body):
    """Rough overlap check: does the attempted action look like the scar's
    known failed action? Used to decide reinforce/override vs brand new scar."""
    action_kw = set(extract_keywords(action_taken))
    scar_kw = set(extract_keywords(scar_body["action"]))
    if not scar_kw:
        return False
    overlap = action_kw & scar_kw
    return len(overlap) / len(scar_kw) >= 0.5


def try_action_and_learn(situation, action_taken, outcome, root_cause=None, real_fix=None):
    """
    Step 2: learn. Call this after actually attempting an action, to let
    the agent update its memory based on what really happened.

    outcome: "success" or "failure"
    root_cause / real_fix: only needed if this is a genuinely NEW failure
        (not matching an existing scar) - required to create a full scar.
    """
    scars = find_relevant_scars(situation)
    matched_scar = None
    for s in scars:
        b = s["body"]
        if b["status"] == "active" and _action_matches_scar(action_taken, b):
            matched_scar = b
            break

    print(f"\n--- LEARNING STEP ---")
    print(f"Action taken: {action_taken}")
    print(f"Outcome: {outcome}")

    if matched_scar:
        worked_anyway = (outcome == "success")
        bump_scar_evidence(matched_scar["id"], worked_anyway=worked_anyway)
        updated = get_scar(matched_scar["id"])["body"]
        print(f"Existing scar [{matched_scar['id']}] evidence updated.")
        print(f"  evidence_for={updated.get('evidence_for', 0)}  "
              f"evidence_against={updated.get('evidence_against', 0)}  "
              f"status={updated['status']}")
        if updated["status"] == "overridden":
            print(f"  >>> Scar [{matched_scar['id']}] just flipped to OVERRIDDEN "
                  f"- new evidence contradicts the old failure.")
    else:
        if outcome == "failure":
            if not root_cause or not real_fix:
                print("  (failure with no known root cause/fix yet - logging raw failure only)")
            else:
                scar_id = create_scar(
                    trigger=situation,
                    action=action_taken,
                    root_cause=root_cause,
                    real_fix=real_fix,
                )
                print(f"  >>> NEW SCAR CREATED: [{scar_id}]")
        else:
            ability_id = create_ability(
                trigger=situation,
                action=action_taken,
                note=root_cause or "",
            )
            print(f"  >>> NEW ABILITY RECORDED: [{ability_id}]")

    record_decision(situation, f"outcome={outcome}: {action_taken}")


if __name__ == "__main__":
    situation = "Getting 'CUDA out of memory' when merging a large LoRA adapter on limited RAM."

    handle_situation(situation)

    try_action_and_learn(
        situation=situation,
        action_taken="merge using load_in_8bit=True",
        outcome="failure",
        root_cause="BNB 8-bit merge path errors out at large parameter counts",
        real_fix="merge on CPU in bf16 with enough system RAM, or shard the merge",
    )

    handle_situation(situation)
