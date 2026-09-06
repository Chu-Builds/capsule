"""
Wipes the real capsule.db entirely and reloads a clean seed set.
Use this before recording the demo, so old test noise (duplicate abilities,
leftover live-learned scars from debugging) doesn't show up on camera.
"""
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "memory", "capsule.db")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"Removed {DB_PATH}")
else:
    print("No existing db found - nothing to remove.")

from memory_client import load_seed_scars

count = load_seed_scars()
print(f"Reloaded {count} seed scars into a fresh database.")
