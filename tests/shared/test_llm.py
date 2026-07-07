"""
Unit tests for the LLM backend factory (shared/llm.py) -- focused on the
litellm-routing branch of the "standard" backend (SMART_ASSIGNMENT_MODEL as
a "<provider>/<model>" string). The plain-Gemini branch is exercised
implicitly by every other test in the suite (the sandbox/CI default is
SMART_ASSIGNMENT_LLM_BACKEND=standard -- see conftest.py); "sage" requires
the internal Sage SDK and enterprise credentials, out of scope for this
offline suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from smart_assignment.shared.config import Config
from smart_assignment.shared.llm import generate_text, get_llm


def _litellm_config(**overrides) -> Config:
    return Config(llm_backend="standard", model="openai/gpt-4o-mini", **overrides)


@patch("google.adk.models.lite_llm.LiteLlm")
def test_get_llm_wraps_a_provider_prefixed_model_with_litellm(mock_lite_llm):
    get_llm(_litellm_config())
    mock_lite_llm.assert_called_once_with(model="openai/gpt-4o-mini")


def test_get_llm_returns_bare_gemini_model_string_as_is():
    config = Config(llm_backend="standard", model="gemini-2.5-flash")
    assert get_llm(config) == "gemini-2.5-flash"


@patch("litellm.completion")
def test_generate_text_routes_provider_prefixed_model_through_litellm(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="a fluent rewrite"))]
    )
    result = generate_text(_litellm_config(), "some deterministic trace")
    assert result == "a fluent rewrite"
    mock_completion.assert_called_once_with(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "some deterministic trace"}],
    )


@patch("litellm.completion")
def test_generate_text_strips_whitespace(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="  padded  \n"))]
    )
    assert generate_text(_litellm_config(), "prompt") == "padded"
