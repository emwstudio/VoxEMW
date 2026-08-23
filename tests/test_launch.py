"""launch 文本过滤纯逻辑单测：括号舞台指示剥离（无需 GPU/依赖）。"""

from voxemw.pipeline.launch import strip_stage_directions


def test_strip_fullwidth_paren():
    assert strip_stage_directions("哎妈呀（乐）今天真高兴") == "哎妈呀 今天真高兴"


def test_strip_halfwidth_paren():
    assert strip_stage_directions("这事儿(笑)我跟你说") == "这事儿 我跟你说"


def test_strip_multiple_and_long():
    assert strip_stage_directions("（拍大腿）嘎嘎香（满足）！") == " 嘎嘎香 ！"
    # 超过 20 字的不当动作标注处理（保守不剥，防误伤长括号引用）
    long_inner = "（" + "很长" * 12 + "）"
    assert strip_stage_directions(long_inner) == long_inner


def test_normal_text_untouched():
    assert strip_stage_directions("今天吃十六包泡面，味真足！") == "今天吃十六包泡面，味真足！"
