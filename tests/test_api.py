"""API smoke tests with everything stubbed: no binary, no model, no network."""

import pytest
from fastapi.testclient import TestClient

from opening_theory_engine.serving.app import ServeSettings, create_app

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture()
def client():
    app = create_app(ServeSettings(backend="stub", engine="stub"))
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["explainer"] == "stub-explainer"
    assert body["engine"] == "stub-analyser-1"


def test_root_points_at_docs(client):
    assert client.get("/").json()["docs"] == "/docs"


def test_explain_returns_verified_explanation(client):
    response = client.post("/explain", json={"fen": START_FEN, "move": "e4"})
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"].startswith("Verdict:")
    assert body["verification"]["passed"] is True
    assert body["verification"]["checks"], "expected at least one claim check"
    assert body["engine"]["cp_loss"] >= 0
    assert isinstance(body["engine"]["best_move_san"], str)
    assert body["attempts"] == 1
    assert body["latency_ms"] >= 0


def test_explain_accepts_uci(client):
    response = client.post("/explain", json={"fen": START_FEN, "move": "g1f3"})
    assert response.status_code == 200


def test_bad_fen_is_422(client):
    response = client.post("/explain", json={"fen": "not a fen", "move": "e4"})
    assert response.status_code == 422


def test_illegal_move_is_422(client):
    response = client.post("/explain", json={"fen": START_FEN, "move": "e5"})
    assert response.status_code == 422
    assert "not legal" in response.json()["detail"]


def test_missing_fields_are_422(client):
    assert client.post("/explain", json={"move": "e4"}).status_code == 422


def test_oversized_fields_are_rejected_before_the_engine_sees_them(client):
    """The length bounds, not python-chess, should turn these away."""
    assert (
        client.post(
            "/explain", json={"fen": "8/8/8/8/8/8/8/8 w - - 0 1" * 40, "move": "e4"}
        ).status_code
        == 422
    )
    assert client.post("/explain", json={"fen": START_FEN, "move": "e4" * 50}).status_code == 422
    assert client.post("/explain", json={"fen": "", "move": "e4"}).status_code == 422
