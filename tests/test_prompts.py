from agent.prompts import FENGGE_SYSTEM_PROMPT, LIANGZI_SYSTEM_PROMPT


def test_prompt_contains_liangzi_markers():
    assert "味真足" in LIANGZI_SYSTEM_PROMPT
    assert "这一块" in LIANGZI_SYSTEM_PROMPT
    assert "焖子" in LIANGZI_SYSTEM_PROMPT


def test_prompt_identity():
    assert "大胃袋良子" in LIANGZI_SYSTEM_PROMPT


def test_prompt_length_limit():
    assert len(LIANGZI_SYSTEM_PROMPT) < 2000


def test_fengge_prompt_contains_markers():
    assert "这是个好事啊" in FENGGE_SYSTEM_PROMPT
    assert "恰恰相反" in FENGGE_SYSTEM_PROMPT
    assert "有枣没枣搂一杆子" in FENGGE_SYSTEM_PROMPT
    assert "连接" in FENGGE_SYSTEM_PROMPT


def test_fengge_prompt_identity():
    assert "峰哥亡命天涯" in FENGGE_SYSTEM_PROMPT


def test_fengge_prompt_length_limit():
    assert len(FENGGE_SYSTEM_PROMPT) < 2000


def test_fengge_prompt_anti_liangzi_guard():
    # 同屏防串腔：必须显式禁止良子口头禅
    assert "良子" in FENGGE_SYSTEM_PROMPT
    assert "绝不串腔" in FENGGE_SYSTEM_PROMPT or "禁止模仿" in FENGGE_SYSTEM_PROMPT
