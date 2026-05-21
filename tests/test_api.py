from fastapi.testclient import TestClient

from apps.api.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_simulate_endpoint():
    client = TestClient(app)
    response = client.post(
        "/v1/simulate/request",
        json={
            "request_id": "api_req_1",
            "messages": [
                {"role": "system", "content": "You are a support assistant."},
                {"role": "user", "content": "USER: hello"},
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["policy_name"] == "adaptive_semantic"
    assert payload["misses"] >= 1

