from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random

from semantic_kv.cache.models import PromptRequest, WorkloadSession


class WorkloadGenerator(ABC):
    workload_type: str

    def __init__(self, seed: int = 0) -> None:
        self.rng = Random(seed)

    @abstractmethod
    def generate(self, num_sessions: int, requests_per_session: int) -> list[WorkloadSession]:
        raise NotImplementedError


def flatten_sessions(sessions: list[WorkloadSession]) -> list[PromptRequest]:
    requests: list[PromptRequest] = []
    for session in sessions:
        requests.extend(session.requests)
    return sorted(requests, key=lambda r: (r.timestamp_ms, r.request_id))

