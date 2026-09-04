"""
Demonstrates the evidence-override mechanic: a scar isn't a permanent ban.
If the same action later succeeds under different conditions, contradicting
evidence accumulates, and after enough of it the scar flips to "overridden" -
the agent can reconsider the action, while still remembering why it was
once flagged.

Run with the real db (not the deletion-test one) since this builds on
scar-009 created in earlier demo runs.
"""
from agent import handle_situation, try_action_and_learn
from memory_client import get_scar

situation = "Getting 'CUDA out of memory' when merging a large LoRA adapter, but this time on a machine with much more available system RAM."

print("### BEFORE: current state of scar-009 ###")
before = get_scar("scar-009")["body"]
print(f"status={before['status']}  evidence_for={before.get('evidence_for', 0)}  "
      f"evidence_against={before.get('evidence_against', 0)}\n")

print("### STEP 1: agent still avoids it, same as before ###")
handle_situation(situation)

print("\n### STEP 2: this time, on a system with more RAM, the SAME action actually works ###")
try_action_and_learn(
    situation=situation,
    action_taken="merge using load_in_8bit=True",
    outcome="success",
)

print("\n### STEP 3: it works AGAIN under similar conditions - second piece of contradicting evidence ###")
try_action_and_learn(
    situation=situation,
    action_taken="merge using load_in_8bit=True",
    outcome="success",
)

print("\n### AFTER: scar-009 state should now be OVERRIDDEN ###")
after = get_scar("scar-009")["body"]
print(f"status={after['status']}  evidence_for={after.get('evidence_for', 0)}  "
      f"evidence_against={after.get('evidence_against', 0)}\n")

print("### STEP 4: fresh call, same situation - does the agent now reconsider? ###")
handle_situation(situation)
