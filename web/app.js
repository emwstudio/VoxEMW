/* 峰哥反指提示器 · 警报页逻辑
 *
 * 结构：
 *   时间线（主区）：GET /api/posts/today → 今天微博逐条卡片 + 播放按钮（/api/tts）
 *   新动态播报：轮询 /api/alerts，定时巡检产生的新告警 → 刷新时间线 + 自动播
 *   语音答复（弹窗提醒栏）：带 query 的告警 → 弹出 answer-bar + 自动播，不进时间线
 */

const $ = (id) => document.getElementById(id);

const els = {
  timeline: $("timeline"),
  timelineEmpty: $("timeline-empty"),
  micBtn: $("mic-btn"),
  askInput: $("ask-input"),
  askSendBtn: $("ask-send-btn"),
  askStatus: $("ask-status"),
  answerBar: $("answer-bar"),
  answerQuery: $("answer-query"),
  answerText: $("answer-text"),
  answerReplay: $("answer-replay"),
  answerClose: $("answer-close"),
  voiceSelect: $("voice-select"),
  ttsText: $("tts-text"),
  ttsBtn: $("tts-btn"),
  ttsAudioArea: $("tts-audio-area"),
  status: $("status"),
};

let lastAlertId = 0; // 已处理的最大告警 id
let firstPoll = true; // 首次轮询（含刷新）：不自动播
let statusTimer = null;

function setStatus(text, kind = "") {
  els.status.textContent = text;
  els.status.className = `status ${kind}`;
  if (statusTimer) clearTimeout(statusTimer);
  if (kind !== "busy" && text) {
    statusTimer = setTimeout(() => {
      els.status.textContent = "";
      els.status.className = "status";
    }, 6000);
  }
}

async function apiPost(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

// ---------------------------------------------------------------- 提示音

let audioCtx = null;
function beep() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const t0 = audioCtx.currentTime;
    [0, 0.18].forEach((dt) => {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "square";
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.08, t0 + dt);
      gain.gain.exponentialRampToValueAtTime(0.001, t0 + dt + 0.15);
      osc.connect(gain).connect(audioCtx.destination);
      osc.start(t0 + dt);
      osc.stop(t0 + dt + 0.15);
    });
  } catch (e) { /* 浏览器不让响就算了 */ }
}

// ---------------------------------------------------------------- 音频播放

function playUrl(url) {
  const audio = new Audio(url);
  audio.play().catch(() => setStatus("自动播放被浏览器拦截，点卡片上的按钮手动播", "error"));
  return audio;
}

function playAlertAudio(id) {
  return playUrl(`/api/alerts/${id}/audio`);
}

// ---------------------------------------------------------------- 微博时间线

async function synthInto(btn, text, card) {
  btn.disabled = true;
  btn.textContent = "⏳";
  try {
    const resp = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    const blob = await resp.blob();
    // 播放期间：按钮变 🚨 闪烁且不可再按（防叠播），卡片警报样式，播完复原
    btn.textContent = "🚨";
    if (card) card.classList.add("playing");
    const audio = playUrl(URL.createObjectURL(blob));
    const done = () => {
      if (card) card.classList.remove("playing");
      btn.disabled = false;
      btn.textContent = "▶";
    };
    audio.onended = done;
    audio.onerror = done;
  } catch (e) {
    setStatus(`语音合成失败：${e.message}`, "error");
    btn.disabled = false;
    btn.textContent = "▶";
  }
}

function renderPost(post, isNew) {
  const card = document.createElement("div");
  card.className = "post-card" + (isNew ? " flash" : "");
  const time = document.createElement("span");
  time.className = "alert-time";
  // 有绝对时间戳显示 HH:MM，没有才退回原始相对时间
  time.textContent = post.ts ? post.ts.slice(11, 16) : post.time || "时间未知";
  const btn = document.createElement("button");
  btn.className = "play-btn";
  btn.textContent = "▶";
  btn.title = "语音播报这条";
  // 播的是反指播报稿（没有才播原文）
  const spoken = post.briefing || post.text;
  btn.addEventListener("click", () => synthInto(btn, spoken, card));
  const body = document.createElement("p");
  body.className = "alert-text";
  body.textContent = post.briefing || post.text;
  const head = document.createElement("div");
  head.className = "alert-head";
  head.append(time, btn);
  card.append(head, body);
  if (post.briefing) {
    // 有播报稿时，原文降级为小字引文
    const quote = document.createElement("p");
    quote.className = "post-quote";
    quote.textContent = `原文：${post.text}`;
    card.appendChild(quote);
  }
  return card;
}

async function loadTimeline(flashLatest = false) {
  try {
    const resp = await fetch("/api/posts/today");
    const data = await resp.json();
    const posts = data.posts || []; // 服务端已按发布时间倒序
    els.timeline.innerHTML = "";
    if (!posts.length) {
      els.timeline.innerHTML = '<p class="alerts-empty">今天还没有采集到动态。</p>';
      return;
    }
    posts.forEach((p, i) => els.timeline.appendChild(renderPost(p, flashLatest && i === 0)));
  } catch (e) {
    setStatus(`时间线加载失败：${e.message}`, "error");
  }
}

// ---------------------------------------------------------------- 语音答复弹窗

function showAnswer(alert) {
  els.answerQuery.textContent = alert.query ? `「${alert.query}」` : "";
  els.answerText.textContent = alert.briefing;
  els.answerBar.classList.remove("hidden");
  els.answerReplay.onclick = () => playAlertAudio(alert.id);
  playAlertAudio(alert.id);
}

els.answerClose.addEventListener("click", () => els.answerBar.classList.add("hidden"));

// ---------------------------------------------------------------- 告警轮询

async function pollAlerts() {
  try {
    // 拉全量再本地过滤：服务端重启后告警 id 会清零重排，用 since 会永久错过
    const resp = await fetch("/api/alerts");
    const data = await resp.json();
    const all = data.alerts || [];
    if (all.length && Math.max(...all.map((a) => a.id)) < lastAlertId) {
      lastAlertId = 0; // 检测到服务端重启（最大 id 反而变小），重置游标
    }
    const fresh = all.filter((a) => a.id > lastAlertId);
    if (fresh.length) {
      lastAlertId = Math.max(...fresh.map((a) => a.id));
      if (!firstPoll) {
        let timelineDirty = false;
        let lastScheduled = null;
        for (const a of fresh) {
          if (a.query) {
            // 语音答复：弹窗提醒栏，不进时间线
            beep();
            showAnswer(a);
            els.askStatus.textContent = ""; // 答复到了，清掉「正在检索」
          } else {
            // 定时巡检新动态：刷新时间线 + 自动播报（多条只播最新一条）
            timelineDirty = true;
            lastScheduled = a;
          }
        }
        if (timelineDirty) {
          beep();
          await loadTimeline(true);
          playAlertAudio(lastScheduled.id);
        }
      }
    }
    firstPoll = false;
  } catch (e) {
    setStatus(`告警轮询失败：${e.message}`, "error");
  }
}

// ---------------------------------------------------------------- 语音/文字查询

async function submitQuery(query) {
  query = (query || "").trim();
  if (!query) return;
  // 先给反馈再走请求（快路径服务端同步生成约 8 秒，不能让用户干等）
  els.askStatus.textContent = "指令已接收，正在检索峰哥动态…";
  try {
    await apiPost("/api/ask", { query });
    els.askInput.value = "";
  } catch (e) {
    els.askStatus.textContent = "";
    setStatus(`查询提交失败：${e.message}`, "error");
  }
}

function setupSpeech(sttCfg) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    // 不支持：麦克风置灰，文字输入框兜底
    els.micBtn.disabled = true;
    els.micBtn.title = "当前浏览器不支持语音识别，请用下方文字输入";
    els.micBtn.classList.add("disabled");
    return;
  }
  els.askInput.placeholder = "点麦克风说话，或直接输入查询词";
  els.micBtn.disabled = false;
  els.micBtn.title = "点击开始/停止语音识别";

  let recog = null;
  let listening = false;
  els.micBtn.addEventListener("click", () => {
    if (listening && recog) {
      recog.stop();
      return;
    }
    recog = new SR();
    recog.lang = sttCfg.lang || "zh-CN";
    recog.interimResults = false;
    recog.maxAlternatives = 1;
    recog.onresult = (e) => {
      const text = e.results[0][0].transcript;
      if (text) submitQuery(text);
    };
    recog.onend = () => {
      listening = false;
      els.micBtn.classList.remove("listening");
    };
    recog.onerror = (e) => {
      listening = false;
      els.micBtn.classList.remove("listening");
      if (e.error !== "aborted") setStatus(`语音识别失败：${e.error}`, "error");
    };
    recog.start();
    listening = true;
    els.micBtn.classList.add("listening");
  });
}

// ---------------------------------------------------------------- 初始化

async function init() {
  try {
    const resp = await fetch("/api/blocks");
    const data = await resp.json();
    const blocks = data.blocks || {};
    if (blocks.vad?.impl === "webspeech" && blocks.stt?.impl === "webspeech") {
      setupSpeech(blocks.stt);
    } else {
      els.micBtn.disabled = true;
      els.micBtn.title = "服务端未启用 webspeech 积木";
    }
    els.voiceSelect.innerHTML = "";
    const vresp = await fetch("/api/voices");
    const vdata = await vresp.json();
    for (const v of vdata.voices || []) {
      const opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = v.name;
      els.voiceSelect.appendChild(opt);
    }
  } catch (e) {
    setStatus(`初始化失败：${e.message}`, "error");
  }
  loadTimeline();
  pollAlerts();
  setInterval(pollAlerts, 3000);
}

els.askSendBtn.addEventListener("click", () => submitQuery(els.askInput.value));
els.askInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitQuery(els.askInput.value);
});
els.ttsBtn.addEventListener("click", async () => {
  const text = els.ttsText.value.trim();
  if (!text) {
    setStatus("先输入要试听的文本", "error");
    return;
  }
  const btn = { disabled: false, textContent: "" };
  els.ttsBtn.disabled = true;
  try {
    const resp = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice: els.voiceSelect.value }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    els.ttsAudioArea.innerHTML = "";
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = url;
    els.ttsAudioArea.appendChild(audio);
    audio.play().catch(() => {});
  } catch (e) {
    setStatus(`语音合成失败：${e.message}`, "error");
  } finally {
    els.ttsBtn.disabled = false;
  }
});

init();
