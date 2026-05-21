from semantic_kv.workloads.generators import build_workload_generator


def test_workload_generation_is_deterministic():
    a = build_workload_generator("rag_shared_corpus", 42).generate(3, 2)
    b = build_workload_generator("rag_shared_corpus", 42).generate(3, 2)
    assert [r.model_dump() for s in a for r in s.requests] == [r.model_dump() for s in b for r in s.requests]


def test_all_required_workloads_generate_requests():
    names = [
        "repeated_system",
        "rag_shared_corpus",
        "support_agent",
        "coding_assistant",
        "legal_evidence",
        "tool_using_agent",
    ]
    for name in names:
        sessions = build_workload_generator(name, 1).generate(2, 2)
        assert len(sessions) == 2
        assert sum(len(s.requests) for s in sessions) == 4

