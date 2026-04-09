"""Tests for mnemon.auth middleware."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mnemon.auth import AuthMiddleware, log_auth_startup


def _make_app(auth_mode: str, auth_token: str = "secret-token") -> Starlette:
    """Build a minimal Starlette app with AuthMiddleware."""

    async def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    return Starlette(
        routes=[Route("/", homepage)],
        middleware=[
            Middleware(AuthMiddleware, auth_mode=auth_mode, auth_token=auth_token),
        ],
    )


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------


class TestDisabledMode:
    def test_allows_no_auth(self) -> None:
        client = TestClient(_make_app("disabled"))
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.text == "ok"

    def test_allows_any_token(self) -> None:
        client = TestClient(_make_app("disabled"))
        resp = client.get("/", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Permissive mode
# ---------------------------------------------------------------------------


class TestPermissiveMode:
    def test_allows_valid_token(self) -> None:
        client = TestClient(_make_app("permissive"))
        resp = client.get("/", headers={"Authorization": "Bearer secret-token"})
        assert resp.status_code == 200

    def test_allows_missing_token_with_warning(self) -> None:
        client = TestClient(_make_app("permissive"))
        resp = client.get("/")
        assert resp.status_code == 200

    def test_allows_invalid_token_with_warning(self) -> None:
        client = TestClient(_make_app("permissive"))
        resp = client.get("/", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Enforcing mode
# ---------------------------------------------------------------------------


class TestEnforcingMode:
    def test_allows_valid_token(self) -> None:
        client = TestClient(_make_app("enforcing"))
        resp = client.get("/", headers={"Authorization": "Bearer secret-token"})
        assert resp.status_code == 200
        assert resp.text == "ok"

    def test_rejects_missing_token(self) -> None:
        client = TestClient(_make_app("enforcing"))
        resp = client.get("/")
        assert resp.status_code == 401
        assert "missing_token" in resp.json()["error"]
        assert "WWW-Authenticate" in resp.headers

    def test_rejects_invalid_token(self) -> None:
        client = TestClient(_make_app("enforcing"))
        resp = client.get("/", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401
        assert "invalid_token" in resp.json()["error"]

    def test_rejects_malformed_auth_header(self) -> None:
        client = TestClient(_make_app("enforcing"))
        resp = client.get("/", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_auth_mode_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            app = _make_app("bogus")
            client = TestClient(app, raise_server_exceptions=True)
            client.get("/")

    def test_empty_token_config_rejects_in_enforcing(self) -> None:
        client = TestClient(_make_app("enforcing", auth_token=""))
        resp = client.get("/", headers={"Authorization": "Bearer anything"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Startup logging
# ---------------------------------------------------------------------------


class TestStartupLogging:
    def test_disabled_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="mnemon.auth"):
            log_auth_startup("disabled")
        assert "DISABLED" in caplog.text

    def test_permissive_emits_info(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="mnemon.auth"):
            log_auth_startup("permissive")
        assert "PERMISSIVE" in caplog.text

    def test_enforcing_emits_info(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="mnemon.auth"):
            log_auth_startup("enforcing")
        assert "ENFORCING" in caplog.text
