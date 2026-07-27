---
name: fengge-alerter
description: |
  峰哥反指提示器的 Agent 侧采集 playbook：用 Kimi WebBridge 操作浏览器看
  「峰哥亡命天涯」的当天微博，POST 给本地提示器服务生成反指短报并语音播报。
  两条链路：定时短报（cron 触发）、语音指令检索（页面麦克风 → /api/ask 队列）。
  当 cron 提示词提到「峰哥短报」「反指」「看峰哥微博」「处理语音指令」时使用。
---

# 峰哥反指提示器 · 采集 playbook

提示器服务（LLM 短报 + TTS + 警报页）跑在 AutoDL，经 SSH 隧道映射到
`http://localhost:8000`。本 skill 只负责**感知层**：看微博、组装 posts、回调服务。

## 前置检查（每次执行先做，任一不满足就停下报告用户，不要硬闯）

1. WebBridge 守护进程：`kimi-webbridge status`，或读
   `~/.kimi-code/skills/kimi-webbridge/SKILL.md` 确认工具可用（本 skill 的
   浏览器操作全部走 WebBridge）
2. 提示器服务活着：`curl -s http://localhost:8000/api/blocks` 能返回 JSON。
   不通说明隧道/服务没起，报告用户（隧道命令见 README）
3. 浏览器里微博处于登录态；遇到登录墙/验证码/风控页，停下报告用户，不要尝试
   绕过，不要触碰密码和验证码

**已验证的坑**：微博里 click 链接默认开新标签页，不在 WebBridge 会话内——
要用 `evaluate` 取链接 href 再 `navigate`，不要直接 click 跳转。

## 流程 A · 定时短报（cron 触发）

1. WebBridge `navigate` 直接打开峰哥主页 `https://weibo.com/u/2397417584`
   （UID 已确认；失效再搜索「峰哥亡命天涯」重新解析）
2. `evaluate` 遍历 `document.querySelectorAll("article")` 读博文流，
   **只收集当天博文**（「刚刚/N分钟前/N小时前/今天 HH:MM」或当天日期）
3. **⚠️ 防漏（微博限流）**：峰哥的部分动态会被微博从主页限流隐藏（搜索
   可见、主页不可见，已实测发生）。若主页只抓到 ≤2 条，必须补搜**实时搜索**
   （综合搜索会被转发抽奖的路人刷屏）：
   `https://s.weibo.com/realtime?q=峰哥亡命天涯&timescope=custom:<今天-0>:<今天-23>&Refer=g`，
   筛选作者为「峰哥亡命天涯」本人的卡片，与主页结果按正文去重合并
4. 组装 `posts: [{"time": "<微博原始时间字符串>", "text": "<正文>"}]`：
   保留原始时间字符串（服务端靠它过滤）；正文清洗——只剥开头
   「峰哥亡命天涯 <时间> 来自 <设备>」和结尾的转发/评论/赞计数等 UI 杂质；
   被折叠的长文先点「展开全文」再取。
   **⚠️ text 必须稳定**：去重指纹按正文计算，任何变化都会被当成新博文重复播报。
   - 只剥开头「峰哥亡命天涯 <时间> 来自 <设备>」前缀
   - 结尾的互动计数**循环剥到剥不动为止**：「转发 评论 赞」三组数字（可能带万），
     以及转发微博里**内嵌的原博统计行**（形如「7-20 17:23 9.6万 1.1万 2.2万」
     的日期+三组数字）——这些数字会涨，是重复播报的头号来源
   - 除此之外正文一字不改：不加注释、不概括、不写括号说明
   已验证的抽取方式：
   `evaluate` 遍历 `document.querySelectorAll("article")`，时间取
   `a.querySelector("time, [class*=time]")`，正文取 `a.innerText` 再清洗
4. 回调：
   ```bash
   curl -s -X POST http://localhost:8000/api/briefing \
     -H 'Content-Type: application/json' -d '{"posts": [...]}'
   ```
   - 返回 `{"skipped":"no_new_posts"}`：没有新微博或都已播报过，正常结束
   - 返回 `{"alert_id":..., "briefing":...}`：短报已生成并合成语音，警报页会
     自动播，无需再做任何事
   - 无论哪种结果，服务端都会把收到的当天博文自动并入本地存档
     （有新的存、没有不存），存档是语音查询的数据源
   - 今天一条微博都没有：`posts` 传空数组，由服务端决定（当前逻辑是生成
     「今日暂无动态」类简报——见 server 行为，不要自己编短报）

## 流程 B · 语音指令检索（cron 轮询 /api/tasks/pending 触发）

**数据源是服务端本地存档（data/posts_archive.json），不开浏览器、不用 WebBridge。**
存档由定时短报流程自动累积（有新的存、没有不存）。

1. `curl -s --noproxy '*' http://localhost:8000/api/tasks/pending`；`tasks` 为空就结束
2. 对每个 `{task_id, query}`：
   a. `curl -s --noproxy '*' http://localhost:8000/api/posts/today` 拿当天存档
   b. 用语义匹配筛出与 query 相关的条目（注意语音识别错字，如「长心」=「长鑫」；
      主页当天内容明显不够时，才回退去走流程 A 的 WebBridge 采集补档）
   c. 回调（**必须带 query 和 task_id**，task_id 用于核销队列）：
      ```bash
      curl -s --noproxy '*' -X POST http://localhost:8000/api/briefing \
        -H 'Content-Type: application/json' \
        -d '{"posts": [...], "query": "<原query>", "task_id": "<task_id>"}'
      ```
   d. 存档里确实没有相关内容：`posts` 传空数组也要回调，让服务端生成
      「今天还没发过相关动态」的播报，否则用户的指令会石沉大海
3. 全部处理完再结束；某个任务失败了保留在队列里（不要手动核销），下轮重试

## 红线

- 只看、只读，不点赞/评论/转发/私信，不动用户账号任何设置
- 反指短报是娱乐内容，不给用户追加投资建议；用户追问时也只播报不荐股
- 微博内容版权归原博主；posts 仅用于生成短报，不另存副本
