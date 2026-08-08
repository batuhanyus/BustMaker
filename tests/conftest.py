"""Shared pytest fixtures for the whole test suite."""

import pytest

from core.config import Config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.load()
