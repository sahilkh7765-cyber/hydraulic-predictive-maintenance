from __future__ import annotations

import time
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


def wait_for(client: TestClient, statuses: "tuple[str, ...]" = ("ready", "unavailable"), timeout: float = 240) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get("/api/health").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.25)
    raise TimeoutError("API did not finish warming up")


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as c:
        health = wait_for(c)
        assert health["status"] == "ready", health
        yield c
