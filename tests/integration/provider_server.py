"""Shared lifetime for local provider servers used on the host and inside Docker."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from http.server import ThreadingHTTPServer
from threading import Thread
from typing import Any

from tests.integration.constants import EPHEMERAL_PORT, LOOPBACK_ADDRESS
from tests.integration.local_provider import LocalProvider


@contextmanager
def serve_provider(handler: type[LocalProvider]) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Start the selected response handler and release its server and thread on every exit."""
    requests: list[dict[str, Any]] = []
    server = ThreadingHTTPServer((LOOPBACK_ADDRESS, EPHEMERAL_PORT), partial(handler, requests=requests))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{LOOPBACK_ADDRESS}:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
