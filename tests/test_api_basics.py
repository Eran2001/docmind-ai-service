import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.deps import verify_internal_key
from core.config import Settings, get_settings
from core.errors import register_error_handlers
from tests.conftest import KEY


def test_health_needs_no_key(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
    assert res.headers["x-request-id"]


def test_unknown_route_uses_error_shape(client: TestClient) -> None:
    res = client.get("/nope")
    assert res.status_code == 404
    assert res.json() == {"error": {"code": "NOT_FOUND", "message": "Not found."}}


@pytest.fixture
def guarded() -> TestClient:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/secret", dependencies=[Depends(verify_internal_key)])
    async def secret() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app)


def test_missing_key_is_401(guarded: TestClient) -> None:
    res = guarded.get("/secret")
    assert res.status_code == 401
    assert res.json()["error"]["code"] == "UNAUTHORIZED"


def test_wrong_key_is_401(guarded: TestClient) -> None:
    assert guarded.get("/secret", headers={"X-Internal-Key": "nope"}).status_code == 401


def test_right_key_passes(guarded: TestClient) -> None:
    assert guarded.get("/secret", headers={"X-Internal-Key": KEY}).json() == {"ok": True}


def test_missing_env_exits_with_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERNAL_API_KEY")
    monkeypatch.setitem(Settings.model_config, "env_file", None)  # ignore the real config/.env
    get_settings.cache_clear()
    with pytest.raises(SystemExit, match="INTERNAL_API_KEY"):
        get_settings()
