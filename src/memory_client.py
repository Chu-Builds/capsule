"""
Thin wrapper around Sibyl Memory for the capsule agent.
Two entity kinds: "scar" (failure -> avoid) and "ability" (success -> prefer).
"""
import json
import os
import re

from sibyl_memory_client import MemoryClient

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
    """search_entities() ANDs every token together by default (confirmed
    from source: _sanitize_fts5_query, v0.4.2+). There's no OR mode. So we
    search one keyword at a time and merge results in Python instead."""
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
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


def find_relevant_abilities(query_text):
    """Same pattern as find_relevant_scars, but for abilities - the positive
    half of memory, so it can actually influence a decision instead of just
    accumulating unused evidence."""
    keywords = extract_keywords(query_text)
    seen = {}
    for kw in keywords:
        results = memory.search_entities(kw, category="ability")
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


def _next_id(category):
    """Derive the next id by scanning existing entities directly, instead of
    trusting a separate counter that can silently drift out of sync (this
    already bit us once - a stale counter collided with seeded scar-001)."""
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
    """A failed action becomes a brand new scar, created at runtime."""
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


def _find_matching_ability(trigger, action_taken):
    """Check if an ability already exists for a similar action, so success
    reinforces existing competence instead of spawning duplicates."""
    keywords = extract_keywords(action_taken)
    seen = {}
    for kw in keywords:
        for r in memory.search_entities(kw, category="ability"):
            seen[r["name"]] = r
    for r in seen.values():
        existing_kw = set(extract_keywords(r["body"]["action"]))
        action_kw = set(extract_keywords(action_taken))
        if existing_kw and len(action_kw & existing_kw) / len(existing_kw) >= 0.5:
            return r["body"]
    return None


def _find_ability_by_supersedes(scar_id):
    """Look up the ability that superseded a given scar, so further
    contradicting evidence can reinforce that ability too, not just
    increment the scar's own counter with no corresponding positive signal."""
    results = memory.search_entities(scar_id, category="ability")
    for r in results:
        if r["body"].get("supersedes_scar") == scar_id:
            return r["body"]["id"]
    return None


def reinforce_ability(ability_id):
    """Bumps confidence and evidence on an existing ability instead of
    creating a duplicate record for the same demonstrated competence."""
    ability = memory.get_entity("ability", ability_id)
    body = ability["body"]
    body["evidence_for"] = body.get("evidence_for", 0) + 1
    body["confidence"] = min(0.99, body.get("confidence", 0.6) + 0.1)
    memory.set_entity("ability", ability_id, body)
    return body


def create_ability(trigger, action, note="", confidence=0.6, source="learned:live",
                     supersedes_scar=None):
    """A successful action, not previously scarred, becomes a new ability.
    If supersedes_scar is set, this ability is explicitly linked to the scar
    it contradicted, so the lifecycle is visible in the data itself."""
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
    if supersedes_scar:
        body["supersedes_scar"] = supersedes_scar
    memory.set_entity("ability", ability_id, body)
    return ability_id


def bump_scar_evidence(scar_id, worked_anyway: bool):
    """Called after an outcome is observed. If the 'dangerous' action actually
    succeeded under new conditions, count it as evidence against the scar.
    Two pieces of contradicting evidence flips it to overridden and creates a
    linked ability. Further contradictions reinforce that linked ability too.
    Four pieces archives the scar out of WARM entirely."""
    scar = get_scar(scar_id)
    body = scar["body"]

    if worked_anyway:
        body["evidence_against"] = body.get("evidence_against", 0) + 1

        if body["evidence_against"] == 2 and body["status"] == "active":
            body["status"] = "overridden"
            memory.set_entity("scar", scar_id, body)
            create_ability(
                trigger=body["trigger"],
                action=body["action"],
                note=f"Supersedes scar {scar_id} - later evidence contradicted the original failure",
                supersedes_scar=scar_id,
            )
            return

        if body["status"] == "overridden":
            linked_ability_id = _find_ability_by_supersedes(scar_id)
            if linked_ability_id:
                reinforce_ability(linked_ability_id)

        if body["evidence_against"] >= 4 and body["status"] == "overridden":
            memory.set_entity("scar", scar_id, body)
            memory.archive_entity("scar", scar_id)
            return
    else:
        body["evidence_for"] = body.get("evidence_for", 0) + 1

    memory.set_entity("scar", scar_id, body)


if __name__ == "__main__":
    count = load_seed_scars()
    print(f"Loaded {count} scars into Sibyl.")
