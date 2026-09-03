from sibyl_memory_client import MemoryClient

memory = MemoryClient.local("memory/capsule.db")

memory.set_entity("scar", "test-scar", {
    "trigger": "test",
    "action": "test_action",
    "outcome": "failure",
    "severity": "low",
    "confidence": 0.5,
    "status": "active"
})

result = memory.get_entity("scar", "test-scar")
print(result)
