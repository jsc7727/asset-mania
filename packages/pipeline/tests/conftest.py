"""Pipeline tests are pure: no socket, no subprocess, no clock, no randomness."""

import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def deny_sockets(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("pipeline code must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture(autouse=True)
def deny_subprocesses(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("pipeline code must not start a subprocess")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)


@pytest.fixture
def now():
    return datetime(2026, 8, 19, 9, 35, tzinfo=UTC)


@pytest.fixture
def source_scene(tmp_path: Path) -> Path:
    """A synthetic .blend header stand-in; pipeline code never parses it."""
    path = tmp_path / "private-character.blend"
    path.write_bytes(b"BLENDER-v502" + bytes(64))
    return path
