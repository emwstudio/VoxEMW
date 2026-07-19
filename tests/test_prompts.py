from agent.prompts import LIANGZI_SYSTEM_PROMPT


def test_prompt_contains_liangzi_markers():
    assert "味真足" in LIANGZI_SYSTEM_PROMPT
    assert "这一块" in LIANGZI_SYSTEM_PROMPT
    assert "焖子" in LIANGZI_SYSTEM_PROMPT


def test_prompt_identity():
    assert "大胃袋良子" in LIANGZI_SYSTEM_PROMPT


def test_prompt_length_limit():
    assert len(LIANGZI_SYSTEM_PROMPT) < 2000
