"""Link verification: a URL only enters the kit if it resolves right now."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

_HEADERS = {"User-Agent": "HubScout/0.1 (link checker; +https://github.com/2910Abhishek/HubScout)"}


@dataclass(frozen=True)
class LinkStatus:
    url: str
    ok: bool
    status: int | None
    reason: str


class LinkChecker(Protocol):
    async def check(self, url: str) -> LinkStatus: ...


class HttpLinkChecker:
    def __init__(self, timeout_s: float) -> None:
        self._timeout = timeout_s

    async def check(self, url: str) -> LinkStatus:
        if not url.startswith(("https://", "http://")):
            return LinkStatus(url, False, None, "not an http(s) URL")
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True, headers=_HEADERS
            ) as client:
                resp = await client.head(url)
                if resp.status_code in (403, 405) or resp.status_code >= 500:
                    # Many sites reject HEAD; confirm with a GET before giving up.
                    resp = await client.get(url)
        except httpx.HTTPError as exc:
            return LinkStatus(url, False, None, f"unreachable ({type(exc).__name__})")
        if resp.status_code < 400:
            return LinkStatus(url, True, resp.status_code, "ok")
        return LinkStatus(url, False, resp.status_code, f"HTTP {resp.status_code}")
