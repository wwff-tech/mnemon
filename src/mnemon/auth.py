"""Authentication middleware for the Mnemon MCP server.

Three modes controlled by ``auth_mode`` in config:

- **disabled**: No auth checks. Emits a startup warning.
- **permissive**: Accepts all requests but logs warnings for missing/invalid
  tokens. Use this while migrating clients to support auth.
- **enforcing**: Rejects requests without a valid bearer token (401).
"""

from __future__ import annotations

import hmac
import logging
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("mnemon.auth")

VALID_AUTH_MODES = {"disabled", "permissive", "enforcing"}


def _extract_bearer(request: Request) -> str | None:
    """Return the bearer token from the Authorization header, or None."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


class AuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware implementing Mnemon's auth modes."""

    def __init__(self, app: Any, auth_mode: str, auth_token: str) -> None:
        super().__init__(app)
        if auth_mode not in VALID_AUTH_MODES:
            raise ValueError(
                f"Invalid auth_mode {auth_mode!r}. "
                f"Must be one of: {', '.join(sorted(VALID_AUTH_MODES))}"
            )
        self.auth_mode = auth_mode
        self.auth_token = auth_token

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if self.auth_mode == "disabled":
            return await call_next(request)  # type: ignore[no-any-return]

        token = _extract_bearer(request)
        client = request.client.host if request.client else "unknown"
        path = request.url.path

        if token is None:
            if self.auth_mode == "permissive":
                logger.warning(
                    "Unauthenticated request from %s to %s "
                    "(auth_mode=permissive, allowing)",
                    client, path,
                )
                return await call_next(request)  # type: ignore[no-any-return]
            # enforcing
            logger.warning(
                "Rejected unauthenticated request from %s to %s", client, path,
            )
            return JSONResponse(
                {"error": "missing_token", "message": "Authorization header required"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mnemon"'},
            )

        if not self._token_valid(token):
            if self.auth_mode == "permissive":
                logger.warning(
                    "Invalid token from %s to %s "
                    "(auth_mode=permissive, allowing)",
                    client, path,
                )
                return await call_next(request)  # type: ignore[no-any-return]
            # enforcing
            logger.warning(
                "Rejected invalid token from %s to %s", client, path,
            )
            return JSONResponse(
                {"error": "invalid_token", "message": "Invalid bearer token"},
                status_code=401,
                headers={
                    "WWW-Authenticate": (
                        'Bearer realm="mnemon", error="invalid_token"'
                    )
                },
            )

        return await call_next(request)  # type: ignore[no-any-return]

    def _token_valid(self, token: str) -> bool:
        if not self.auth_token:
            return False
        return hmac.compare_digest(token, self.auth_token)


def log_auth_startup(auth_mode: str) -> None:
    """Log the auth configuration at server startup."""
    if auth_mode == "disabled":
        logger.warning(
            "Auth is DISABLED — the MCP server accepts unauthenticated requests. "
            "Set auth_mode to 'permissive' or 'enforcing' in config to enable."
        )
    elif auth_mode == "permissive":
        logger.info(
            "Auth is PERMISSIVE — unauthenticated requests are allowed but logged. "
            "Switch to 'enforcing' once all clients support auth."
        )
    elif auth_mode == "enforcing":
        logger.info("Auth is ENFORCING — all requests require a valid bearer token.")
