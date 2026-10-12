"""Typed API errors: every failure reaches the client as {"error": code, "detail": message}."""

from __future__ import annotations

from typing import Dict, Optional


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, detail: str, headers: Optional[Dict[str, str]] = None
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.headers = headers
