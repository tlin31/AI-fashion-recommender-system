# Shared pytest fixtures for the test suite. Provides a mock OpenAI client so tests
# run without any external service dependencies.

from __future__ import annotations

import pytest


@pytest.fixture
def mock_openai(mocker):
    return mocker.patch("pipeline.utils._client")
