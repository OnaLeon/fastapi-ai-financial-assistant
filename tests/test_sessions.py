import pytest
from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


class FakeChat:
    pass


@pytest.fixture(autouse=True)
def clear_sessions():
    main.sessions.clear()


def test_create_session(monkeypatch):
    monkeypatch.setattr(
        main,
        "create_chat",
        lambda: FakeChat()
    )

    response = client.post("/sessions")

    assert response.status_code == 200

    data = response.json()

    assert "session_id" in data
    assert data["session_id"] in main.sessions


def test_unknown_session_returns_404():
    response = client.post(
        "/sessions/not-a-real-session/reset"
    )

    assert response.status_code == 404


def test_reset_session(monkeypatch):
    monkeypatch.setattr(
        main,
        "create_chat",
        lambda: FakeChat()
    )

    create_response = client.post("/sessions")
    session_id = create_response.json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/reset"
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Conversation memory cleared."


def test_delete_session(monkeypatch):
    monkeypatch.setattr(
        main,
        "create_chat",
        lambda: FakeChat()
    )

    create_response = client.post("/sessions")
    session_id = create_response.json()["session_id"]

    response = client.delete(
        f"/sessions/{session_id}"
    )

    assert response.status_code == 200
    assert session_id not in main.sessions