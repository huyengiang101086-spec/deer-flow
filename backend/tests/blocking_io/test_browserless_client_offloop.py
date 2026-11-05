"""Regression: the Browserless client must build its HTTP client off the event loop.

``BrowserlessClient.fetch_html_with_status()`` and ``capture_screenshot()`` built
``httpx.AsyncClient`` inside their coroutine. Client construction loads the TLS trust
store synchronously (``ssl.SSLContext.load_verify_locations`` reads the CA bundle), so
every web fetch and screenshot stalled all other tasks sharing the Gateway event loop
for the duration of that load — reported in #7910.

These tests keep the production client and replace only ``httpx.AsyncClient.send``, so
client construction, TLS initialisation, request building and cleanup stay real. They
pin the two halves of the contract:

* the trust store is loaded off the event-loop thread, and
* the request itself still goes out on the event loop, with the client closed afterwards.
"""

from __future__ import annotations

import ssl
import threading

import httpx
import pytest

from deerflow.community.browserless.browserless_client import (
    BrowserlessClient,
    BrowserlessFetchResult,
    BrowserlessScreenshotResult,
)

pytestmark = pytest.mark.asyncio


def _record_trust_store_loads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the thread of every TLS trust-store load, keeping the real load."""
    threads: list[int] = []
    original = ssl.SSLContext.load_verify_locations

    def recording_load_verify_locations(self, cafile=None, capath=None, cadata=None):  # noqa: ANN001, ANN202
        threads.append(threading.get_ident())
        return original(self, cafile, capath, cadata)

    monkeypatch.setattr(ssl.SSLContext, "load_verify_locations", recording_load_verify_locations)
    return threads


def _stub_send(monkeypatch: pytest.MonkeyPatch, response: httpx.Response, send_threads: list[int]) -> None:
    async def fake_send(self, request, **kwargs):  # noqa: ANN001, ANN202
        send_threads.append(threading.get_ident())
        return response

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)


async def test_fetch_html_loads_tls_trust_store_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    trust_store_threads = _record_trust_store_loads(monkeypatch)
    send_threads: list[int] = []
    response = httpx.Response(
        200,
        text="<html><body>rendered</body></html>",
        headers={"X-Response-Code": "200", "X-Response-Status": "OK"},
        request=httpx.Request("POST", "http://browserless:3000/content"),
    )
    _stub_send(monkeypatch, response, send_threads)

    loop_thread = threading.get_ident()
    client = BrowserlessClient(base_url="http://browserless:3000", timeout_s=5)
    result = await client.fetch_html_with_status("https://example.com/page")

    assert isinstance(result, BrowserlessFetchResult)
    assert result.html == "<html><body>rendered</body></html>"
    assert result.target_status_code == "200"
    # The trust store was loaded at least once, never on the loop thread.
    assert trust_store_threads
    assert loop_thread not in trust_store_threads
    # ... while the request itself still goes out on the loop.
    assert send_threads == [loop_thread]


async def test_capture_screenshot_loads_tls_trust_store_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    trust_store_threads = _record_trust_store_loads(monkeypatch)
    send_threads: list[int] = []
    response = httpx.Response(
        200,
        content=b"\x89PNG\r\n\x1a\n",
        headers={"Content-Type": "image/png"},
        request=httpx.Request("POST", "http://browserless:3000/screenshot"),
    )
    _stub_send(monkeypatch, response, send_threads)

    loop_thread = threading.get_ident()
    client = BrowserlessClient(base_url="http://browserless:3000", timeout_s=5)
    result = await client.capture_screenshot("https://example.com/page")

    assert isinstance(result, BrowserlessScreenshotResult)
    assert result.content == b"\x89PNG\r\n\x1a\n"
    assert trust_store_threads
    assert loop_thread not in trust_store_threads
    assert send_threads == [loop_thread]


async def test_fetch_html_closes_the_offloaded_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Moving construction off the loop must not leak the client."""
    created: list[httpx.AsyncClient] = []
    response = httpx.Response(
        200,
        text="<html>ok</html>",
        headers={"X-Response-Code": "200", "X-Response-Status": "OK"},
        request=httpx.Request("POST", "http://browserless:3000/content"),
    )

    class RecordingClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            super().__init__(*args, **kwargs)
            created.append(self)

        async def send(self, request, **kwargs):  # noqa: ANN001, ANN003, ANN201
            return response

    monkeypatch.setattr("deerflow.community.browserless.browserless_client.httpx.AsyncClient", RecordingClient)

    client = BrowserlessClient(base_url="http://browserless:3000", timeout_s=5)
    await client.fetch_html_with_status("https://example.com/page")

    assert created, "the production path must construct exactly one client per request"
    assert all(instance.is_closed for instance in created)
