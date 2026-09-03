from memory_client import memory

queries = [
    "My llama-cpp-python server won't accept --tensor_split no matter how I format it.",
    "tensor_split",
    "llama-cpp-python server",
    "invalid optional value",
]

for q in queries:
    print(f"\n--- query: {q!r} ---")
    results = memory.search_entities(q)
    print(f"raw result count: {len(results)}")
    for r in results:
        print(f"  category={r.get('category')}  name={r.get('name')}")
