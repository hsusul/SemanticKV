from __future__ import annotations

from semantic_kv.cache.models import PromptRequest, WorkloadSession
from semantic_kv.workloads.base import WorkloadGenerator


SYSTEM_PROMPTS = [
    "You are a careful enterprise assistant. Use concise answers and preserve citations.",
    "You are a coding assistant. Prefer minimal diffs and explain failing tests.",
    "You are a legal review assistant. Cite evidence IDs and avoid unsupported claims.",
]

DOCS = [f"Document: shared policy section {i}. " + "This context describes controls, exceptions, and definitions. " * 12 for i in range(12)]
LARGE_DOCS = [
    f"Document: hot regulatory corpus chunk {i}. "
    + "This long retrieved context is reused by many requests and should be valuable under memory pressure. " * 42
    for i in range(6)
]
TOOLS = [f"TOOL: schema search_{i}(query: string) -> results" for i in range(5)]
CODE_FILES = [f"```python\ndef function_{i}(value):\n    return value + {i}\n```" for i in range(8)]
LEGAL_EVIDENCE = [f"EVIDENCE: exhibit {i}. Contract clause, timeline, and deposition excerpt. " * 18 for i in range(8)]
LONG_LEGAL_EVIDENCE = [
    f"EVIDENCE: recurring exhibit bundle {i}. "
    + "Contract clause timeline deposition citation privilege note damages analysis. " * 55
    for i in range(5)
]
UNIQUE_USER_CHURN = " ".join(f"unique_request_detail_{i}" for i in range(90))


class RepeatedSystemPromptGenerator(WorkloadGenerator):
    workload_type = "repeated_system"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        system = SYSTEM_PROMPTS[0]
        for s in range(num_sessions):
            requests = []
            for r in range(requests_per_session):
                requests.append(self._request(s, r, [("system", system), ("user", f"USER: unique question {s}-{r} about account {self.rng.randint(1, 999)}")]))
            sessions.append(self._session(s, requests, ["system_shared"]))
        return sessions

    def _request(self, s: int, r: int, pairs: list[tuple[str, str]]) -> PromptRequest:
        return PromptRequest(request_id=f"{self.workload_type}_{s}_{r}", session_id=f"sess_{self.workload_type}_{s}", timestamp_ms=s * 1000 + r, messages=[{"role": role, "content": content} for role, content in pairs], metadata={"workload": self.workload_type})

    def _session(self, s: int, requests: list[PromptRequest], artifacts: list[str]) -> WorkloadSession:
        return WorkloadSession(session_id=f"sess_{self.workload_type}_{s}", workload_type=self.workload_type, requests=requests, shared_artifact_ids=artifacts)


class RAGSharedCorpusGenerator(RepeatedSystemPromptGenerator):
    workload_type = "rag_shared_corpus"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        for s in range(num_sessions):
            hot_doc = self.rng.choice(DOCS[:4])
            requests = []
            for r in range(requests_per_session):
                doc = hot_doc if self.rng.random() < 0.7 else self.rng.choice(DOCS)
                pairs = [("system", SYSTEM_PROMPTS[0]), ("user", f"CONTEXT:\n{doc}\n\nUSER: answer corpus question {s}-{r}")]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["docs"]))
        return sessions


class SupportAgentGenerator(RepeatedSystemPromptGenerator):
    workload_type = "support_agent"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        template = "TEMPLATE: classify ticket {{ticket}} and cite policy."
        for s in range(num_sessions):
            customer = f"CONTEXT: customer history tier={self.rng.choice(['gold','silver'])} incidents={s % 5}"
            requests = []
            history = ""
            for r in range(requests_per_session):
                history = f"ASSISTANT: previous resolution step {r - 1}" if r else ""
                pairs = [("system", SYSTEM_PROMPTS[0]), ("user", f"{template}\n\n{customer}\n\n{history}\n\nUSER: troubleshoot ticket {s}-{r}")]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["support_template"]))
        return sessions


class CodingAssistantGenerator(RepeatedSystemPromptGenerator):
    workload_type = "coding_assistant"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        for s in range(num_sessions):
            file_context = self.rng.choice(CODE_FILES)
            requests = []
            for r in range(requests_per_session):
                test_output = "Traceback: assertion failed in tests/test_cache.py" if r % 3 == 0 else ""
                pairs = [("system", SYSTEM_PROMPTS[1]), ("user", f"CODE:\n{file_context}\n\n{test_output}\nUSER: implement change {s}-{r}")]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["repo_file"]))
        return sessions


class LegalEvidenceGenerator(RepeatedSystemPromptGenerator):
    workload_type = "legal_evidence"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        for s in range(num_sessions):
            evidence = self.rng.choice(LEGAL_EVIDENCE)
            requests = []
            for r in range(requests_per_session):
                pairs = [("system", SYSTEM_PROMPTS[2]), ("user", f"CONTEXT:\n{evidence}\n\nUSER: assess issue {s}-{r}")]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["evidence"]))
        return sessions


class ToolUsingAgentGenerator(RepeatedSystemPromptGenerator):
    workload_type = "tool_using_agent"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        schema = self.rng.choice(TOOLS)
        for s in range(num_sessions):
            requests = []
            for r in range(requests_per_session):
                output = f"TOOL: result for query {s}-{r}: status={self.rng.choice(['ok','pending'])}"
                pairs = [("system", SYSTEM_PROMPTS[0]), ("tool", schema), ("user", f"{output}\nUSER: continue task {s}-{r}")]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["tool_schema"]))
        return sessions


class LowMemoryRAGStressGenerator(RepeatedSystemPromptGenerator):
    workload_type = "low_memory_rag_stress"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        hot_docs = LARGE_DOCS[:3]
        for s in range(num_sessions):
            requests = []
            for r in range(requests_per_session):
                doc = hot_docs[(s + r) % len(hot_docs)] if self.rng.random() < 0.85 else self.rng.choice(LARGE_DOCS)
                query_noise = self.rng.randint(10_000, 99_999)
                pairs = [
                    ("system", SYSTEM_PROMPTS[0]),
                    ("user", f"CONTEXT:\n{doc}\n\nUSER: question variant {s}-{r}-{query_noise} {UNIQUE_USER_CHURN}"),
                ]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["hot_large_docs"]))
        return sessions


class SharedSystemPromptHighReuseGenerator(RepeatedSystemPromptGenerator):
    workload_type = "shared_system_prompt_high_reuse"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        system = SYSTEM_PROMPTS[0] + " Follow the enterprise response contract exactly."
        template = "TEMPLATE: response_sections={{summary,decision,citations,next_steps}} tone=concise"
        for s in range(num_sessions):
            requests = []
            for r in range(requests_per_session):
                pairs = [
                    ("system", system),
                    ("user", f"{template}\n\nUSER: unique operational request {s}-{r}-{self.rng.randint(1, 1_000_000)} {UNIQUE_USER_CHURN}"),
                ]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["shared_system", "shared_template"]))
        return sessions


class ToolAgentSessionReuseGenerator(RepeatedSystemPromptGenerator):
    workload_type = "tool_agent_session_reuse"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        schema = TOOLS[0]
        for s in range(num_sessions):
            session_tool_outputs = [
                f"TOOL: session {s} lookup result {i}: account_state={self.rng.choice(['active', 'blocked', 'review'])}"
                for i in range(2)
            ]
            requests = []
            for r in range(requests_per_session):
                reused_output = session_tool_outputs[r % len(session_tool_outputs)]
                unique_tail = f"TOOL: ephemeral result turn={r} nonce={self.rng.randint(1, 99999)} {UNIQUE_USER_CHURN}"
                pairs = [
                    ("system", SYSTEM_PROMPTS[0]),
                    ("tool", schema),
                    ("user", f"{reused_output}\n\n{unique_tail}\n\nUSER: continue tool workflow {s}-{r}"),
                ]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["tool_schema", "session_tool_outputs"]))
        return sessions


class LegalReviewLongContextGenerator(RepeatedSystemPromptGenerator):
    workload_type = "legal_review_long_context"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        template = "TEMPLATE: issue -> evidence -> legal standard -> risk rating -> citation list"
        for s in range(num_sessions):
            evidence = LONG_LEGAL_EVIDENCE[s % len(LONG_LEGAL_EVIDENCE)]
            requests = []
            for r in range(requests_per_session):
                pairs = [
                    ("system", SYSTEM_PROMPTS[2]),
                    ("user", f"{template}\n\nCONTEXT:\n{evidence}\n\nUSER: review matter {s}-{r}-{self.rng.randint(1, 9999)} {UNIQUE_USER_CHURN}"),
                ]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["legal_template", "recurring_evidence"]))
        return sessions


class WorkloadShiftStaticVsAdaptiveGenerator(RepeatedSystemPromptGenerator):
    workload_type = "workload_shift_static_vs_adaptive"

    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        sessions = []
        template = "TEMPLATE: stable enterprise answer format {{answer,citations}}"
        midpoint = max(1, num_sessions // 2)
        for s in range(num_sessions):
            requests = []
            for r in range(requests_per_session):
                if s < midpoint:
                    pairs = [
                        ("system", SYSTEM_PROMPTS[0]),
                        ("user", f"{template}\n\nUSER: unique first-phase prompt {s}-{r}-{self.rng.randint(1, 99999)} {UNIQUE_USER_CHURN}"),
                    ]
                else:
                    doc = LARGE_DOCS[(s + r) % 3]
                    pairs = [
                        ("system", "You are a corpus-grounded assistant. Prefer retrieved evidence."),
                        ("user", f"CONTEXT:\n{doc}\n\nUSER: second-phase RAG query {s}-{r}-{self.rng.randint(1, 99999)} {UNIQUE_USER_CHURN}"),
                    ]
                requests.append(self._request(s, r, pairs))
            sessions.append(self._session(s, requests, ["phase_shift"]))
        return sessions


def build_workload_generator(name: str, seed: int) -> WorkloadGenerator:
    generators: dict[str, type[WorkloadGenerator]] = {
        "repeated_system": RepeatedSystemPromptGenerator,
        "rag_shared_corpus": RAGSharedCorpusGenerator,
        "support_agent": SupportAgentGenerator,
        "coding_assistant": CodingAssistantGenerator,
        "legal_evidence": LegalEvidenceGenerator,
        "tool_using_agent": ToolUsingAgentGenerator,
        "low_memory_rag_stress": LowMemoryRAGStressGenerator,
        "shared_system_prompt_high_reuse": SharedSystemPromptHighReuseGenerator,
        "tool_agent_session_reuse": ToolAgentSessionReuseGenerator,
        "legal_review_long_context": LegalReviewLongContextGenerator,
        "workload_shift_static_vs_adaptive": WorkloadShiftStaticVsAdaptiveGenerator,
    }
    try:
        return generators[name](seed)
    except KeyError as exc:
        raise ValueError(f"Unknown workload family: {name}") from exc
