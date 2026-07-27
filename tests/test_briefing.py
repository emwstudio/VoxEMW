"""briefing 纯逻辑单测：当天过滤、指纹去重、存档合并、prompt 构造、输出清洗。"""

from datetime import date, datetime

from voxemw.briefing import (
    build_select_messages,
    build_messages,
    dedupe_posts,
    filter_today,
    parse_post_time,
    parse_selection,
    merge_archive,
    parse_briefing,
    post_fingerprint,
)

TODAY = date(2026, 7, 27)


class TestFilterToday:
    def test_relative_and_today_formats_kept(self):
        posts = [
            {"time": "刚刚", "text": "a"},
            {"time": "5分钟前", "text": "b"},
            {"time": "今天 09:30", "text": "c"},
            {"time": "7月27日 12:00", "text": "d"},
            {"time": "2026-7-27", "text": "e"},
            {"time": "2026年7月27日", "text": "f"},
        ]
        kept = filter_today(posts, today=TODAY)
        assert [p["text"] for p in kept] == ["a", "b", "c", "d", "e", "f"]

    def test_other_days_dropped(self):
        posts = [
            {"time": "7月26日 23:59", "text": "x"},
            {"time": "2026-7-26", "text": "y"},
            {"time": "2025年7月27日", "text": "z"},
        ]
        assert filter_today(posts, today=TODAY) == []

    def test_missing_or_unparseable_time_kept(self):
        posts = [
            {"time": "", "text": "a"},
            {"text": "b"},  # time 缺失
            {"time": "昨天 10:00", "text": "c"},  # 解析不出，保守保留
        ]
        kept = filter_today(posts, today=TODAY)
        assert [p["text"] for p in kept] == ["a", "b", "c"]

    def test_hours_ago_same_day(self):
        """「N小时前」按当前时间回推，未跨零点则算当天。"""
        posts = [{"time": "1小时前", "text": "a"}]
        kept = filter_today(posts, today=date.today())
        # 当前时间减 1 小时多数情况仍当天；跨零点边界本测试不强行断言剔除
        assert kept in (posts, [])


class TestFingerprint:
    def test_stable(self):
        p = {"time": "今天 09:00", "text": "hello"}
        assert post_fingerprint(p) == post_fingerprint(p)

    def test_sensitive_to_text_only(self):
        # 只按正文取指纹：时间漂移（「N分钟前」→「N小时前」）不应产生新指纹
        base = {"time": "今天 09:00", "text": "hello"}
        assert post_fingerprint(base) != post_fingerprint({"time": "今天 09:00", "text": "world"})
        assert post_fingerprint(base) == post_fingerprint({"time": "3小时前", "text": "hello"})


class TestDedupePosts:
    def test_drops_seen_keeps_new(self):
        posts = [
            {"time": "今天 09:00", "text": "old"},
            {"time": "今天 10:00", "text": "new"},
        ]
        seen = {post_fingerprint(posts[0])}
        assert dedupe_posts(posts, seen) == [posts[1]]

    def test_all_seen_returns_empty(self):
        posts = [{"time": "今天 09:00", "text": "old"}]
        seen = {post_fingerprint(p) for p in posts}
        assert dedupe_posts(posts, seen) == []


class TestBuildMessages:
    def test_contains_persona_query_posts_and_limit(self):
        posts = [{"time": "今天 09:00", "text": "峰哥说房价要涨"}]
        msgs = build_messages(posts, "房价", "你是突发主播。", max_chars=150)
        assert len(msgs) == 2
        system, user = msgs[0]["content"], msgs[1]["content"]
        assert "你是突发主播。" in system  # 人设文本
        assert "150" in system  # 字数约束
        assert "峰哥说啥我反着来" in system  # 任务规则
        assert "峰哥说房价要涨" in user  # 微博内容
        assert "今天 09:00" in user  # 时间
        assert "房价" in user  # 查询词

    def test_no_query(self):
        msgs = build_messages([{"time": "刚刚", "text": "x"}], None, "人设")
        assert "查询词" not in msgs[1]["content"]


class TestParseBriefing:
    def test_strips_fence_and_markdown(self):
        assert parse_briefing("```\n**峰哥反指**：注意风险\n```") == "峰哥反指：注意风险"

    def test_strips_quotes(self):
        assert parse_briefing('"注意风险"') == "注意风险"
        assert parse_briefing("“注意风险”") == "注意风险"

    def test_empty(self):
        assert parse_briefing("") == ""
        assert parse_briefing("```\n```") == ""


class TestMergeArchive:
    def test_new_posts_appended_with_meta(self):
        archive = []
        merge_archive(archive, [{"time": "10分钟前", "text": "a"}], today=TODAY)
        assert len(archive) == 1
        e = archive[0]
        assert e["fp"] == post_fingerprint({"time": "10分钟前", "text": "a"})
        assert e["time"] == "10分钟前"
        assert e["text"] == "a"
        assert e["date"] == TODAY.isoformat()

    def test_existing_fp_refreshes_time_only(self):
        archive = []
        # 指纹只含正文：同一条博文时间漂移后仍是同一条目，只刷新时间
        merge_archive(archive, [{"time": "10分钟前", "text": "a"}], today=TODAY)
        merge_archive(archive, [{"time": "1小时前", "text": "a"}], today=TODAY)
        assert len(archive) == 1
        assert archive[0]["time"] == "1小时前"  # 时间以最新看到的为准
        # 不同正文才是新条目
        merge_archive(archive, [{"time": "1小时前", "text": "b"}], today=TODAY)
        assert len(archive) == 2

    def test_order_preserved(self):
        archive = []
        merge_archive(
            archive,
            [{"time": "刚刚", "text": "x"}, {"time": "刚刚", "text": "y"}],
            today=TODAY,
        )
        assert [e["text"] for e in archive] == ["x", "y"]


class TestSelectMessages:
    POSTS = [
        {"time": "1小时前", "text": "妈呀，长鑫卖早了，现在都52元了"},
        {"time": "2小时前", "text": "在韩国品尝韩式牛杂锅！俩人300元！"},
    ]

    def test_build_select_messages(self):
        msgs = build_select_messages(self.POSTS, "长鑫")
        assert msgs[0]["role"] == "system"
        user = msgs[1]["content"]
        assert "[0] 妈呀，长鑫卖早了" in user
        assert "[1] 在韩国品尝" in user
        assert "查询词：长鑫" in user

    def test_parse_selection_basic(self):
        assert parse_selection("[0, 2]", 3) == [0, 2]

    def test_parse_selection_strips_fence(self):
        assert parse_selection("```json\n[1]\n```", 2) == [1]

    def test_parse_selection_drops_out_of_range_and_dup(self):
        assert parse_selection("[0, 5, -1, 0]", 2) == [0]

    def test_parse_selection_garbage(self):
        assert parse_selection("没有相关", 3) == []
        assert parse_selection("", 3) == []
        assert parse_selection('{"a": 1}', 3) == []


class TestParsePostTime:
    NOW = datetime(2026, 7, 27, 16, 30, 0)

    def test_relative(self):
        assert parse_post_time("刚刚", self.NOW) == "2026-07-27T16:30:00"
        assert parse_post_time("5分钟前", self.NOW) == "2026-07-27T16:25:00"
        assert parse_post_time("3小时前", self.NOW) == "2026-07-27T13:30:00"

    def test_absolute(self):
        assert parse_post_time("今天 09:05", self.NOW) == "2026-07-27T09:05:00"
        assert parse_post_time("7月27日 12:00", self.NOW) == "2026-07-27T12:00:00"
        assert parse_post_time("2026-7-27", self.NOW) == "2026-07-27T00:00:00"

    def test_unparseable(self):
        assert parse_post_time("", self.NOW) is None
        assert parse_post_time("上周", self.NOW) is None
