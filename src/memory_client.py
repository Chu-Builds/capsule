"""
Thin wrapper around Sibyl Memory for the capsule agent.
Two entity kinds: "scar" (failure -> avoid) and "ability" (success -> prefer).
"""
import json
import os
import re

from sibyl_memory_client import MemoryClient

# Swappable via env var so the deletion-test demo can point at a throwaway
# db instead of ever touching the real one (file-move-based swapping proved
# unreliable on this system - real data got lost once already).
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "memory", "capsule.db")
DB_PATH = os.environ.get("CAPSULE_DB_PATH", DEFAULT_DB_PATH)

memory = MemoryClient.local(DB_PATH)

STOPWORDS = {
    "my", "the", "a", "an", "is", "it", "to", "of", "and", "or", "no", "how",
    "wont", "won't", "does", "do", "on", "in", "for", "with", "that", "this",
    "i", "me", "am", "be", "was", "were", "will", "would", "can", "could",
    "not", "at", "so", "but", "if", "matter", "any"
}


def extract_keywords(text):
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def load_seed_scars(path="seed_data/scars.json"):
    with open(path) as f:
        scars = json.load(f)
    for scar in scars:
        memory.set_entity("scar", scar["id"], scar)
    return len(scars)


def get_scar(scar_id):
    return memory.get_entity("scar", scar_id)


def find_relevant_scars(query_text):
    keywords = extract_keywords(query_text)
    seen = {}
    for kw in keywords:
        results = memory.search_entities(kw, category="scar")
        for r in results:
            seen[r["name"]] = r
    return list(seen.values())


def record_decision(trigger, action_chosen, blocked_scar_id=None):
    memory.write_event(acted=[{
        "trigger": trigger,
        "action_chosen": action_chosen,
        "blocked_by_scar": blocked_scar_id,
    }])


def bump_scar_evidence(scar_id, worked_anyway: bool):
    scar = get_scar(scar_id)
    body = scar["body"]
    if worked_anyway:
        body["evidence_against"] = body.get("evidence_against", 0) + 1
        if body["evidence_against"] >= 2:
            body["status"] = "overridden"
    else:
        body["evidence_for"] = body.get("evidence_for", 0) + 1
    memory.set_entity("scar", scar_id, body)


def _next_id(category):
    existing = memory.search_entities(category, category=category, limit=1000)
    max_n = 0
    for e in existing:
        name = e.get("name", "")
        if name.startswith(f"{category}-"):
            try:
                n = int(name.split("-")[-1])
                max_n = max(max_n, n)
            except ValueError:
                pass
    return f"{category}-{max_n + 1:03d}"


def create_scar(trigger, action, root_cause, real_fix, severity="medium",
                 confidence=0.6, source="learned:live"):
    scar_id = _next_id("scar")
    body = {
        "id": scar_id,
        "source": source,
        "trigger": trigger,
        "action": action,
        "outcome": "failure",
        "root_cause": root_cause,
        "real_fix": real_fix,
        "severity": severity,
        "confidence": confidence,
        "evidence_for": 1,
        "evidence_against": 0,
        "status": "active",
    }
    memory.set_entity("scar", scar_id, body)
    return scar_id


def create_ability(trigger, action, note="", confidence=0.6, source="learned:live"):
    ability_id = _next_id("ability")
    body = {
        "id": ability_id,
        "source": source,
        "trigger": trigger,
        "action": action,
        "outcome": "success",
        "note": note,
        "confidence": confidence,
        "evidence_for": 1,
        "status": "active",
    }
    memory.set_entity("ability", ability_id, body)
    return ability_id


if __name__ == "__main__":
    count = load_seed_scars()
    print(f"Loaded {count} scars into Sibyl.")
