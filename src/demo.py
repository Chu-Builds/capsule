"""
Demo entry point with a real on/off memory switch.

--without-memory points the agent at a fresh, separate, throwaway database
instead of ever moving/renaming the real one. Safer than file-swapping,
which proved unreliable on this system.
"""
import argparse
import os
import tempfile


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--with-memory", action="store_true")
    group.add_argument("--without-memory", action="store_true")
    args = parser.parse_args()

    if args.without_memory:
        # point at a brand new empty db in a temp dir - real capsule.db
        # is never touched, so nothing to restore, nothing to lose.
        temp_dir = tempfile.mkdtemp(prefix="capsule_deletion_test_")
        os.environ["CAPSULE_DB_PATH"] = os.path.join(temp_dir, "empty.db")
        print(f">>> Memory DISABLED: agent pointed at a fresh empty db "
              f"at {os.environ['CAPSULE_DB_PATH']} - real capsule.db untouched.\n")

    # imported here, AFTER CAPSULE_DB_PATH is possibly set, so
    # memory_client.py picks up the right path at import time
    from agent import handle_situation, try_action_and_learn

    situation = "Getting 'CUDA out of memory' when merging a large LoRA adapter on limited RAM."

    print("### STEP 1: agent encounters the situation ###")
    handle_situation(situation)

    print("\n### STEP 2: the obvious first move is tried and fails ###")
    try_action_and_learn(
        situation=situation,
        action_taken="merge using load_in_8bit=True",
        outcome="failure",
        root_cause="BNB 8-bit merge path errors out at large parameter counts",
        real_fix="merge on CPU in bf16 with enough system RAM, or shard the merge",
    )

    print("\n### STEP 3: fresh call, same situation - does it repeat the mistake? ###")
    handle_situation(situation)


if __name__ == "__main__":
    main()
