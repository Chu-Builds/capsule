"""
Thin wrapper around Sibyl Memory for the capsule agent.
Two entity kinds: "scar" (failure -> avoid) and "ability" (success -> prefer).
"""
import json
import os
import re

from sibyl_memory_client import MemoryClient

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "memory", "capsule.db")

memory = MemoryClient.local(DB_PATH)

STOPWORDS = {
    "my", "the", "a", "an", "is", "it", "to", "of", "and", "or", "no", "how",
    "wont", "won't", "does", "do", "on", "in", "for", "with", "that", "this",
    "i", "me", "am", "be", "was", "were", "will", "would", "can", "could",
    "not", "at", "so", "but", "if", "matter", "any"
}


def extract_keywords(situation_text):
    """search_entities() ANDs every token together by default (confirmed
    from source: _sanitize_fts5_query, v0.4.2+). There's no OR mode. So we
    search one keyword at a time and merge results in Python instead."""
    words = re.findall(r"[a-zA-Z0-9_]+", situation_text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def load_seed_scars(path="seed_data/scars.json"):
    """One-time load: pushes every scar in the JSON file into Sibyl as an entity."""
    with open(path) as f:
        scars = json.load(f)

    for scar in scars:
        memory.set_entity("scar", scar["id"], scar)

    return len(scars)


def get_scar(scar_id):
    return memory.get_entity("scar", scar_id)


def find_relevant_scars(query_text):
    """Search across all stored scars for ones matching the current situation.
    Runs one FTS5 query per keyword and merges/dedupes in Python, since the
    SDK's search_entities has no built-in OR/any-match mode."""
    keywords = extract_keywords(query_text)

    seen = {}
    for kw in keywords:
        results = memory.search_entities(kw, category="scar")
        for r in results:
            seen[r["name"]] = r

    return list(seen.values())


def record_decision(trigger, action_chosen, blocked_scar_id=None):
    """Write every decision to the COLD journal, always, regardless of outcome."""
    memory.write_event(acted=[{
        "trigger": trigger,
        "action_chosen": action_chosen,
        "blocked_by_scar": blocked_scar_id,
    }])


def bump_scar_evidence(scar_id, worked_anyway: bool):
    """Called after an outcome is observed. If the 'dangerous' action actually
    succeeded under new conditions, count it as evidence against the scar."""
    scar = get_scar(scar_id)
    body = scar["body"]
    if worked_anyway:
        body["evidence_against"] = body.get("evidence_against", 0) + 1
        if body["evidence_against"] >= 2:
            body["status"] = "overridden"
    else:
        body["evidence_for"] = body.get("evidence_for", 0) + 1
    memory.set_entity("scar", scar_id, body)


if __name__ == "__main__":
    count = load_seed_scars()
    print(f"Loaded {count} scars into Sibyl.")
