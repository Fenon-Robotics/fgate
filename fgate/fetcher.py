from __future__ import annotations

import hashlib
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from .interfaces import VideoFetcher
from .models import VideoIdentity


class FetchError(RuntimeError):
    pass


class HttpsVideoFetcher(VideoFetcher):
    def __init__(self, *, block_private_networks: bool, timeout_seconds: float = 120):
        self._block_private_networks = block_private_networks
        self._timeout = httpx.Timeout(timeout_seconds, connect=10)

    def fetch(self, source_url: str, destination: Path) -> VideoIdentity:
        current = source_url
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        digest = hashlib.sha256()
        size = 0
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=False) as client:
                for _ in range(6):
                    self.validate(current)
                    with client.stream("GET", current) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                raise FetchError("redirect response has no location")
                            current = urljoin(current, location)
                            continue
                        response.raise_for_status()
                        with temporary.open("wb") as stream:
                            for block in response.iter_bytes():
                                stream.write(block)
                                digest.update(block)
                                size += len(block)
                    break
                else:
                    raise FetchError("too many video URL redirects")
            if size == 0:
                raise FetchError("video URL returned an empty body")
            temporary.replace(destination)
            return VideoIdentity.model_validate(
                {
                    "source_url": source_url,
                    "path": destination,
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                }
            )
        except (httpx.HTTPError, OSError) as error:
            raise FetchError(str(error)) from error
        finally:
            if temporary.exists():
                temporary.unlink()

    def validate(self, source_url: str) -> None:
        parsed = urlsplit(source_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise FetchError("video URL must use HTTPS")
        if not self._block_private_networks:
            return
        try:
            addresses = socket.getaddrinfo(
                parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM
            )
        except socket.gaierror as error:
            raise FetchError(f"cannot resolve video host {parsed.hostname!r}") from error
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise FetchError("cloud video URL resolves to a non-public address")
