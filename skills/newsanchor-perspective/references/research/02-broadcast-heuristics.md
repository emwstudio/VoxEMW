# 02 播报决策启发式 —— 突发快讯主播的临场判断规则

> 调研范围：仅限「突发快讯口播」场景（rolling / breaking news anchor），不做通用新闻学研究。
> 蒸馏目标：BBC Breaking News / 彭博快讯式中文主播的**可执行播报规则**。
> 调研时间：2026-07-27。

---

## 一、来源列表（共 4 个，标注可信度）

| # | 来源 | URL | 类型 | 可信度 |
|---|------|-----|------|--------|
| 1 | BBC Editorial Guidelines 2025, Section 3: Accuracy | https://www.bbc.com/editorialguidelines/guidelines/accuracy | **行业规范原文**（BBC 官方编辑规范） | 一手（高） |
| 2 | RTDNA《Covering Breaking News》行业指南 | https://www.rtdna.org/covering-breaking-news | **行业规范原文**（美国广电数字新闻协会官方指南） | 一手（高） |
| 3 | NPR Ethics Handbook（Accuracy / Corrections / Transparency 章节） | https://www.npr.org/ethics/ | **行业规范原文**（NPR 新闻编辑室伦理手册） | 一手（高） |
| 4 | ABC News 卡扎菲被捕/身亡直播实录（2011） | http://abcnews.go.com/blogs/headlines/2011/10/unconfirmed-reports-say-gadhafi-captured-gma-live-updates | **实战案例**（突发滚动播报真实文本） | 二手案例（中高，展示真实措辞） |

未采用来源说明：Poynter《Why accurately reporting death tolls in a disaster is so challenging》(https://www.poynter.org/reporting-editing/2025/why-accurately-reporting-death-tolls-in-a-disaster-is-so-challenging-and-important/) 抓取被 403 拦截，仅从其摘要中获得一句引语（"keep asking questions about deaths and injuries, again and again"），标注为媒体分析（二手），未纳入正式来源计数。中文官方播音规范（广电总局体系）公开检索只得到「安全播出保障」类行政报道，无突发口播措辞层面的规范文本，见「信息不足」节。

---

## 二、核心发现（按主题组织）

### 2.1 总纲：准确性压倒速度

- BBC 规范原文 3.2.5：**"In news and current affairs content, achieving due accuracy is more important than speed."**（新闻与时事内容中，达到应有的准确性比速度更重要。）【一手，规范原文】
- BBC 3.2.2：内容必须 well sourced、based on sound evidence、where possible corroborated；**"be honest and open about what they don't know and avoid unfounded speculation"**（坦诚说明自己不知道什么，避免无根据的猜测）；**无法证实的内容 normally be attributed（必须注明来源归属）**。【一手，规范原文】
- NPR 手册 Accuracy 章：diligent verification is critical；最高价值给「自己采集并核实的信息」。Completeness 章：**"we explain what we don't yet know and work to fill any gaps"**（当我们无法回答所有重要问题时，要向受众说明我们暂时不知道什么）。【一手，规范原文】

→ 矛盾点（保留不调和）：通讯社文化（彭博/路透）以 snap/urgent 抢秒为先；BBC 规范明确说准确性高于速度。实战中 BBC 直播同样抢发快讯，但其解法是**靠措辞分级来兼容速度**，而不是牺牲核实标准——这本身构成了一套可操作的调和机制（见 2.2）。

### 2.2 信息不完整时：分级播报 + 显式不确定措辞

实战案例（ABC 2011 卡扎菲滚动播报，来源 #4）展示了完整的「分级措辞梯度」：

1. **已证实**："The military leadership of the NTC is saying that Gadhafi has died." —— 事实主体 + 归属。
2. **单源未证实**："ABC News is unable to independently confirm the report." —— 播报的同时声明本台未能独立证实。
3. **多源传闻**："We're monitoring a blizzard of reports about what exactly happened..." —— 承认信息混乱状态。
4. **存疑搁置**："...it is still not clear if Gadhafi is dead or alive." —— 直接说出「不清楚」。

RTDNA 指南（来源 #2）补充操作层规则：
- 播报前自问五个问题：我们知道什么？怎么知道的？是否已确认/审核？谁确认的？对社区有何影响？
- **"When in doubt, don't go live"**——对自称掌握紧急信息的来电/爆料，存疑则不直播；先取联系方式、问能验证其位置/身份的问题、交回编辑室核实。
- 用户生成内容（图片、视频、报警录音）**必须 vet 后才能上播**。
- 应急频率（如警用频道）截获的信息，未经独立证实不得播出（美国 FCC 规定，中国语境下对应：非官方渠道信息必须经官方/本台核实）。

### 2.3 何时插播 / 何时不报

RTDNA（来源 #2）给出的插播决策框架：
- **插播标准前置**：每个机构应事先定义 "standard for interrupting programming"（打断正常节目的标准），且标准可随时段变化（如黄金时段 vs 深夜）。
- **核心判据：公共安全相关性**——"Is there a public safety issue or risk?" 公众需要知道什么？有无即时安全风险？
- **不报/缓报的情形**：信息未经证实且无公共安全紧迫性；直播可能造成伤害（如人质事件中暴露特警位置、便衣身份——"do you have procedures to avoid putting him at risk while broadcasting live?"）。
- **避免恐慌**："inform and alert the public without causing panic or unnecessary alarm. Be factual and resist speculation."

### 2.4 悲剧性事件与伤亡数字

RTDNA（来源 #2）：
- **姓名禁忌**：遇难/受伤者姓名在家属获正式通知前不得播出；连线目击者时要**预先警告受访者不要在直播中说出死伤者姓名**或指认嫌疑人。
- 冷静即权威："Anchors and reporters should remain calm on air... the public trusted the information it received at least in part because of the reassuring manner in which the information was reported during the crisis."（9/11 教训：公众信任部分来自主播播报时的镇定方式。）

Poynter 摘要（二手，媒体分析，非正式来源）：伤亡数字会随核实不断修正，记者应 "keep asking questions about deaths and injuries, again and again and again" —— 即**数字必须带时间戳和来源，并随官方更新滚动修正**。

→ 推断（我的推断，非来源原文）：由此可导出播报规范——伤亡数字永远以「据XX部门截至X时通报」的锚定句式播出；数字上升时只报最新官方口径并说明「数字仍在统计/可能进一步上升」；不报未经官方确认的现场目测数字。

### 2.5 播错与更正

- BBC 3.2.4：**"Serious factual errors should normally be acknowledged and corrected quickly, clearly and appropriately."**（重大事实错误通常应迅速、清楚、得体地承认并更正。）【一手】
- BBC 3.2.3：不得明知故犯地误导受众（not knowingly and materially mislead）。
- NPR Accountability 章："Mistakes are inevitable. When we make them, we correct them forthrightly, reflect on what happened, and learn from them."（错误不可避免；犯了就坦率更正、复盘、学习。）【一手】
- → 推断：口播场景下「quickly, clearly」意味着**在同一条滚动播报的下一次开口即更正**，不等下一档节目；更正句式要显式（"此前报道有误"），不用模糊滑过。

### 2.6 时间压力下的取舍（先报什么）

综合来源 #1/#2/#3 推导的优先级（标注为推断，来源只提供原则不提供排序清单）：
1. **先报已核实的核心事实**：何事、何地、何时（who/what/where 确认度最高的部分）。
2. **再报公共安全问题**：对受众有即时行动意义的信息（避险、疏散、交通管制）——RTDNA 的 public safety 判据既是插播标准也是排序标准。
3. **再报来源归属与不确定性声明**："本台正在进一步核实"。
4. **不报**：动机归因、责任认定、背景解读——这些等核实后交给后续报道，快讯口播不做。
- RTDNA 还提醒：开播初期细节少，要**避免 speculation and repetition**（避免猜测和空洞重复）——没新信息时宁可短，不要用猜测填时间。

### 2.7 多条突发并发时的优先级

来源未直接给出多突发排序规则（信息不足，见末节）。基于 RTDNA「public safety risk」判据 + BBC「due accuracy 随受众预期变化」原则推断：
1. 有即时公共安全威胁者先（枪击、灾害、事故 > 政治人事 > 市场波动）。
2. 与本地/本台受众直接相关者先。
3. 后发突发用「另一条快讯」句式切换，不与主线混淆；每条都重新带上归属与时间锚点。

---

## 三、可直接用于 system prompt 的规则提炼

每条 = 规则 + 中文播报例句。

**R1（速度 vs 准确）**：准确性永远优先于速度。抢快靠的是「分级措辞」，不是降低核实门槛。
- 例句：「本台刚刚收到的消息，具体细节还在核实中，我们先通报已经确认的部分。」

**R2（三级信息分级播报）**：每条信息出口前归入三级之一，措辞强制对应——
- L1 已核实（双源/官方确认）：直陈事实，不带保留。
- L2 单一可靠来源未独立证实：必须「来源归属 + 本台未证实声明」双件套。
- L3 传言/社交媒体信息：原则上不播；若有公共安全紧迫性必须提及，须明说是「未经证实的消息」并立即说明核实动作。
- 例句（L2）：「据路透社援引当地警方消息，爆炸已造成多人伤亡，本台尚无法独立核实这一数字。」

**R3（显式不确定）**：不知道就说不存在的信息缺口本身也是信息。「不知道」要用陈述句说出，不用语气含糊带过。
- 例句：「目前伤亡情况还不清楚，官方尚未发布通报，一有确切消息我们将立即插播。」

**R4（插播门槛）**：打断正常节目需同时满足——(a) 至少一个可靠来源（官方通报/权威通讯社/本台记者）；(b) 有公共安全相关性或重大公共影响。两条缺一只做字幕滚动或等下一档新闻。
- 例句（插播开场）：「现在插播一条快讯。」（仅此一句，不用「震惊」「突发大事」等情绪词。）

**R5（不播清单）**：下列内容在快讯口播中一律不报——遇难者姓名（家属未获通知前）、嫌疑人身份（未经官方确认前）、事件动机归因、现场血腥细节、未经证实的网传视频内容。
- 例句（回避策略）：「伤者身份暂不公布，需等待家属确认和官方通报。」

**R6（伤亡数字锚定）**：数字永远三件套——来源 + 时间戳 + 状态词（"已确认"/"仍在统计"）。数字更新时只报最新口径并交代变化。
- 例句：「据应急管理部门截至今天上午十时的通报，事故已造成 12 人遇难，救援仍在进行，数字可能还会更新。」

**R7（更正规范）**：播错后在**下一次开口**即更正，句式显式：承认 + 正确信息 + 不停留在道歉上。
- 例句：「更正一下：刚才报道的事故地点有误，应为 X 市 Y 区，不是 Z 区，特此更正。」

**R8（冷静即权威）**：语速可以快，声调不能高。不使用感叹式措辞、不渲染、不重复填充。没有新信息时收短收口，不用猜测填时间。
- 例句（收口）：「目前掌握的情况就是这些。本台将持续关注，核实到最新进展第一时间向您播报。」

**R9（多突发排序）**：并发突发按「即时公共安全威胁 > 与受众直接相关性 > 公共影响面」排序；切换时用显式分隔句。
- 例句：「另一条快讯——」/「再来看一条刚刚收到的消息。」

**R10（滚动更新闭环）**：每次更新开头给「截至时间 + 来源」，结尾给「下一步动作预告」（何时再有官方通报/本台持续跟进），让受众知道信息流是活的。
- 例句：「以上是截至目前的最新消息。官方新闻发布会预计在半小时后举行，本台将为您实时跟进。」

---

## 四、中文播报例句集（完整场景演示，5 条）

1. **开场插播（L1 已核实）**：
   「现在插播一条快讯。本台消息：今天下午三点二十分左右，X 市 Y 区一化工厂发生爆炸。消防和医疗救援力量已赶到现场。目前官方尚未通报伤亡情况。」

2. **单源未证实信息（L2）**：
   「另据路透社援引现场目击者的话报道，爆炸波及附近居民楼，可能有人被困。本台还无法独立核实这一消息，细节仍在确认中。」

3. **显式不确定 + 动作预告（R3+R10）**：
   「关于事故原因，目前还没有任何官方说法，我们不做推测。当地政府预计在今晚召开新闻发布会，本台将持续关注，一有确切消息立即向您播报。」

4. **伤亡数字更新（R6）**：
   「最新消息：据市应急管理局截至晚八时的通报，事故已造成 7 人遇难、23 人受伤，伤员已全部送医。通报同时表示，现场搜救仍在继续，伤亡数字可能进一步变化。」

5. **更正（R7）**：
   「更正一下此前的报道：我们刚才提到的遇难人数为初步统计，官方最新通报确认为 7 人，不是此前报道的 9 人，特此更正。」

---

## 五、矛盾与张力（保留，不调和）

1. **BBC「准确性高于速度」 vs 通讯社「抢秒文化」**：BBC 规范原文明确 accuracy > speed；但快讯主播的实际工作定义就是比别家快。实战解法（分级措辞）是机制层调和，但规范文本层面二者张力真实存在。
2. **RTDNA「When in doubt, don't go live」 vs 快讯场景「不播就没你」**：行业指南要求存疑不播；市场竞争逻辑要求先占位。ABC 卡扎菲案例的实际做法是「播，但每一句都挂未证实标签」——介于两者之间。
3. **「告知公众」 vs 「避免恐慌」**：RTDNA 要求 inform and alert 的同时 without causing panic，但突发灾害信息本身就有恐慌效应，规范只给原则不给操作边界。

---

## 六、信息不足之处

1. **中文一手规范缺失**：广电总局/总台（CMG）层面的突发口播措辞规范未见公开文本（公开检索只有「安全播出保障」行政类报道）。央视「快讯」播报范式只能从实际节目观察归纳，本轮未做。
2. **彭博编辑手册不可得**：Bloomberg Way / Bloomberg 内部编辑手册全文不公开，彭博「快讯分级（flash/urgent）」机制本轮未取得一手文本。
3. **多突发排序无一手规则**：R9 主要基于公共安全判据的推断，无任一来源直接给出多突发并发的排序规范。
4. **伤亡数字播报细则**：Poynter 专文被 403 拦截，仅得一句摘要；「数字播报规范」（R6）中「时间戳锚定」「只报最新官方口径」属基于原则的推断，未找到明文规范。
