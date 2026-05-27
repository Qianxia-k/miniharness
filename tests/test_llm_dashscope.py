from miniharness.llm import LLMClient
from miniharness.providers import get_profile


def test_qwen_profile_has_extra_body():
    """DashScope profile includes enable_thinking=False for non-streaming compat."""
    profile = get_profile("qwen")
    client = LLMClient(profile=profile, model=profile.default_model)

    assert client.profile.extra_body == {"enable_thinking": False}


def test_openai_profile_has_no_extra_body():
    """OpenAI profile does not need extra_body."""
    profile = get_profile("openai")
    client = LLMClient(profile=profile, model=profile.default_model)

    assert client.profile.extra_body == {}
