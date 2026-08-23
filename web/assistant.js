/* 良子语音助手前端（纯语音模式，星空背景即全部画面）。
 *
 * 下行音频：WebRTC——POST /rtc/offer 建连，音频（Opus）走 RTP 轨，
 *           挂隐藏 <audio> 播放（Chrome 对 RTC 音轨的解码只在媒体元素上才启动）。
 * WS /ws：上行麦克风/控制事件；下行转写/状态事件（音频体已剥离，走音轨）。
 * 星空：全屏 canvas 跟随对话状态——idle 无序漂移 / listening 向中心收拢的
 *       专注波动（随麦克风能量）/ speaking 随 RTC 音频能量的径向声波。
 */

"use strict";

const VOX_JS_VERSION = "20260823e";  // 排障用：Console 里 VOX_JS_VERSION 可验版本
console.log("VOXEMW JS", VOX_JS_VERSION);

const SAMPLE_RATE = 16000;

const els = {
  status: document.getElementById("status"),
  personaBar: document.getElementById("persona-bar"),
  transcript: document.getElementById("transcript"),
  micBtn: document.getElementById("mic-btn"),
};

let ws = null;
let mic = null;
let personas = [];
let currentPersona = null;
let assistantLine = null; // 正在流式累积的助手文本行
let lineGotDeltas = false; // 当前行已收到逐字 delta（新上游 delta+done 双发，done 只收尾不重复上屏）
let rtcEnabled = false;   // vox.status 下发：下行音频走 WebRTC
let pc = null;            // RTCPeerConnection


// ---------------------------------------------------------------------------
// PCM 编解码（麦克风上行）
// ---------------------------------------------------------------------------

function floatTo16BitPCM(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

function base64FromInt16(int16) {
  const bytes = new Uint8Array(int16.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

// RTC 音频输出：挂隐藏 <audio> 元素——Chrome 对 WebRTC 远端音轨只有挂到
// 媒体元素上才开始解码（WebAudio 直通在 Chrome 里是静音，Safari 正常；
// MediaRecorder 能录到是因为录制强制解码。2026-08-23 实测定案）。
// 元素 play() 需用户手势，各按钮点击处兜底补一次。
let rtcAudioEl = null;

function ensureRtcAudio() {
  if (!rtcAudioEl) {
    rtcAudioEl = document.createElement("audio");
    rtcAudioEl.id = "rtc-audio";
    rtcAudioEl.autoplay = true;
    rtcAudioEl.playsInline = true;
    rtcAudioEl.muted = false;
    rtcAudioEl.volume = 1.0;
    rtcAudioEl.style.display = "none";
    document.body.appendChild(rtcAudioEl);
  }
  rtcAudioEl.play().catch(() => {});
  return rtcAudioEl;
}

// ---------------------------------------------------------------------------
// WebRTC 音频：RTP 轨下行，隐藏 <audio> 播放
// ---------------------------------------------------------------------------

async function startRTC() {
  if (pc) return;  // 一次 WS 会话只建一路
  // ICE 配置现取：服务端下发本地 coturn 凭证
  let iceServers = [];
  try {
    iceServers = (await (await fetch("/rtc/ice")).json()).ice_servers || [];
  } catch (_) { /* 取不到就裸 host candidate，LAN 还能用 */ }
  // SSH 隧道场景（页面在 localhost 且有服务端 TURN 下发）：强制 relay——host
  // candidate 是双方各自的私网/环回地址，互指必败；媒体走 coturn TCP 中继。
  // 本机直连（localhost 但无 TURN 下发，如 Mac 本地版）回退 all：loopback
  // host candidate 直连即可，强制 relay 会零候选哑连（2026-08-22 实测踩坑）。
  const isTunnel = ["localhost", "127.0.0.1"].includes(location.hostname);
  const conn = new RTCPeerConnection({
    iceServers,
    iceTransportPolicy: isTunnel && iceServers.length > 0 ? "relay" : "all",
  });
  pc = conn;
  conn.ontrack = (e) => {
    if (e.track.kind === "audio") {
      // 音频挂独立隐藏 <audio>（Chrome 对 RTC 音轨的解码只在媒体元素上才启动）
      const el = ensureRtcAudio();
      el.srcObject = new MediaStream([e.track]);
      el.play().catch(() => {});
      attachTtsAnalyser(e.streams[0] || new MediaStream([e.track]));
    }
  };
  conn.onconnectionstatechange = () => {
    if (["failed", "closed"].includes(conn.connectionState)) {
      // 媒体链路断了（隧道抖动等）：2s 后自动重建，跟 WS 重连一个思路。
      // 用局部变量 conn 判定/操作——全局 pc 可能被重入的新连接占用
      if (pc === conn) {
        pc = null;
      }
      conn.close();
      setTimeout(() => {
        if (ws && ws.readyState === WebSocket.OPEN && rtcEnabled && !pc) {
          startRTC().catch(() => {});
        }
      }, 2000);
    }
  };
  // 显式 recvonly transceiver：Safari 对 createOffer({offerToReceive*}) 老语法
  // 会产出零 m-section 的空 offer（iPad 实测 500），addTransceiver 全浏览器一致
  conn.addTransceiver("audio", { direction: "recvonly" });
  const offer = await conn.createOffer();
  await conn.setLocalDescription(offer);
  // aiortc 不支持 trickle：offer 必须带齐候选。TURN 走 TCP 隧道可能较慢，给足 15s
  await Promise.race([
    new Promise((resolve) => {
      if (conn.iceGatheringState === "complete") return resolve();
      conn.addEventListener("icegatheringstatechange", () => {
        if (conn.iceGatheringState === "complete") resolve();
      });
    }),
    new Promise((resolve) => setTimeout(resolve, 15000)),
  ]);
  if (pc !== conn) { conn.close(); return; }  // 建连期间被断开回收，直接放弃
  const res = await fetch("/rtc/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sdp: conn.localDescription.sdp,
      type: conn.localDescription.type,
      vbr: parseInt(new URLSearchParams(location.search).get("vbr") || "0", 10) || undefined,
    }),
  });
  if (!res.ok) {
    addLine("sys", "", `⚠ WebRTC 建连失败（HTTP ${res.status}），看静态图`);
    if (pc === conn) pc = null;
    conn.close();
    return;
  }
  if (pc !== conn) { conn.close(); return; }
  await conn.setRemoteDescription(await res.json());
  // 排障：12s 后把 ICE 状态+候选回传服务端日志（隧道/TURN 链路黑盒开灯）
  setTimeout(() => {
    if (pc !== conn) return;
    const cands = (conn.localDescription ? conn.localDescription.sdp : "")
      .split("\n")
      .filter((l) => l.includes("candidate:"))
      .map((l) => l.trim().slice(0, 120));
    fetch("/rtc/debug", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        js: VOX_JS_VERSION,
        tunnel: isTunnel,
        iceGatheringState: conn.iceGatheringState,
        iceConnectionState: conn.iceConnectionState,
        connectionState: conn.connectionState,
        candidates: cands,
      }),
    }).catch(() => {});
  }, 12000);
}

function stopRTC() {
  if (!pc) return;
  pc.close();
  pc = null;
}

// ---------------------------------------------------------------------------
// 麦克风（AudioWorklet 采集，16kHz mono）
// ---------------------------------------------------------------------------

const WORKLET_SRC = `
class PCMCapture extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}
registerProcessor("pcm-capture", PCMCapture);
`;

async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  await ctx.audioWorklet.addModule(
    URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }))
  );
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  node.port.onmessage = (e) => {
    // 麦克风能量（星空 listening 模式的专注波动输入）：抽样 RMS，快攻慢放
    const f = e.data;
    let sum = 0;
    for (let i = 0; i < f.length; i += 4) sum += f[i] * f[i];
    const rms = Math.sqrt(sum / Math.max(1, f.length / 4));
    SPACE.micLevel = Math.max(Math.min(rms * 5, 1), SPACE.micLevel * 0.9);
    if (ws && ws.readyState === WebSocket.OPEN) {
      const int16 = floatTo16BitPCM(f);
      ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio: base64FromInt16(int16) }));
    }
  };
  source.connect(node);
  const gain = ctx.createGain();
  gain.gain.value = 0;
  node.connect(gain);
  gain.connect(ctx.destination);
  mic = { ctx, stream, node };
}

function stopMic() {
  if (!mic) return;
  mic.node.disconnect();
  mic.stream.getTracks().forEach((t) => t.stop());
  mic.ctx.close();
  mic = null;
}

// ---------------------------------------------------------------------------
// 转写区
// ---------------------------------------------------------------------------

function addLine(cls, who, text) {
  const div = document.createElement("div");
  div.className = `line ${cls}`;
  if (who) {
    const span = document.createElement("span");
    span.className = "who";
    span.textContent = who;
    div.appendChild(span);
  }
  div.appendChild(document.createTextNode(text));
  els.transcript.appendChild(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return div;
}

function appendAssistantDelta(delta) {
  if (!assistantLine) {
    const name = (personas.find((p) => p.id === currentPersona) || {}).name || "助手";
    assistantLine = addLine("assistant", `${name}:`, "");
  }
  assistantLine.appendChild(document.createTextNode(delta));
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

// ---------------------------------------------------------------------------
// 单一状态指示器（顶栏 #status）：连接状态 + 对话状态合一
//   未连接/已断开 → 灰/琥珀；已连接（未开麦）→ 灰
//   开麦后：idle=🎙 聆听中 / listening=👂 倾听中 / thinking=🤔 思考中 / speaking=🔊 说话中
// avatarState 同时驱动星空模式（见 tickSpace）
// ---------------------------------------------------------------------------

let avatarState = "idle"; // idle | listening | thinking | speaking
let wsConnected = false;

function setIndicator() {
  const el = els.status;
  if (!wsConnected) {
    el.textContent = "已断开";
    el.className = "status warn";
    return;
  }
  if (!mic) {
    el.textContent = "已连接";
    el.className = "status";
    return;
  }
  if (avatarState === "listening") {
    el.textContent = "👂 倾听中…";
    el.className = "status state-listening";
  } else if (avatarState === "thinking") {
    el.textContent = "🤔 思考中…";
    el.className = "status state-thinking";
  } else if (avatarState === "speaking") {
    el.textContent = "🔊 说话中";
    el.className = "status state-speaking";
  } else {
    el.textContent = "🎙 聆听中";
    el.className = "status live";
  }
}

function setAvatarState(state) {
  avatarState = state;
  setIndicator();
}

// ---------------------------------------------------------------------------
// Realtime 事件处理
// ---------------------------------------------------------------------------

const realtimeHandlers = {
  "input_audio_buffer.speech_started"() {
    // 用户开口（打断）：服务端 flush 音频队列，本地只需更新状态
    assistantLine = null;
    setAvatarState("listening");
  },
  "input_audio_buffer.speech_stopped"() {
    // 用户说完：到助手首个音频 delta 之前是「思考」窗口
    if (avatarState === "listening") setAvatarState("thinking");
  },
  "conversation.item.input_audio_transcription.completed"(event) {
    const text = (event.transcript || "").trim();
    if (text) addLine("user", "你:", text);
  },
  "response.output_audio_transcript.delta"(event) {
    if (event.delta) {
      lineGotDeltas = true;
      appendAssistantDelta(event.delta);
    }
  },
  "response.output_text.delta"(event) {
    if (event.delta) {
      lineGotDeltas = true;
      appendAssistantDelta(event.delta);
    }
  },
  "response.output_audio_transcript.done"(event) {
    // delta 流过的行 done 只收尾；没 delta 的老协议才在 done 里补整段
    if (event.transcript && !lineGotDeltas) {
      appendAssistantDelta(event.transcript);
    }
    assistantLine = null;
    lineGotDeltas = false;
  },
  "response.output_text.done"() {
    assistantLine = null;
    lineGotDeltas = false;
  },
  "response.output_audio.delta"(event) {
    // 事件被服务端剥了音频体（音频走 RTC 音轨），但事件本身仍标志「开口」
    if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
  },
  "response.audio.delta"(event) {
    if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
  },
  "response.done"() {
    assistantLine = null;
    if (avatarState === "thinking") setAvatarState("idle");  // 无音频回复的兜底
  },
  error(event) {
    addLine("sys", "", `⚠ ${(event.error && event.error.message) || "未知错误"}`);
  },
};

function handleTextMessage(data) {
  let event;
  try {
    event = JSON.parse(data);
  } catch {
    return;
  }
  if (event.type === "vox.status") {
    currentPersona = event.persona;
    updatePersonaBar();
    rtcEnabled = !!(event.rtc && event.rtc.enabled);
    if (rtcEnabled) {
      startRTC().catch((e) =>
        addLine("sys", "", `⚠ WebRTC 建连异常: ${e.message}`)
      );
    }
    return;
  }
  const handler = realtimeHandlers[event.type];
  if (handler) handler(event);
}

// ---------------------------------------------------------------------------
// 星空背景：全屏 canvas，跟随对话状态（avatarState）的三种动态
//   idle      无人说话：无序漂移 + 闪烁
//   listening 你在说话：减速收拢向中心，随你的音量呼吸（专注倾听感）
//   thinking  良子在想：持续缓慢内流 + 深呼吸
//   speaking  良子说话：从中心向外的径向波动（能量取自 RTC 音频 RMS）
// ---------------------------------------------------------------------------

const SPACE = {
  stars: [],
  w: 0,
  h: 0,
  micLevel: 0,      // 麦克风 RMS（0..1，快攻慢放）
  ttsLevel: 0,      // RTC 音频 RMS（0..1，快攻慢放）
  ttsAnalyser: null,
  ttsData: null,
  last: 0,
};

function attachTtsAnalyser(stream) {
  if (SPACE.ttsAnalyser) return;
  try {
    const actx = new AudioContext();
    const src = actx.createMediaStreamSource(stream);
    const an = actx.createAnalyser();
    an.fftSize = 1024;
    src.connect(an);  // 只分析不回放（声音走隐藏 <audio>），无需接 destination
    SPACE.ttsAnalyser = an;
    SPACE.ttsData = new Uint8Array(an.fftSize);
  } catch (_) { /* 分析失败只是星空不波动，不影响通话 */ }
}

function initSpace() {
  const canvas = document.getElementById("space");
  SPACE.canvas = canvas;
  SPACE.ctx2d = canvas.getContext("2d");

  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    SPACE.w = window.innerWidth;
    SPACE.h = window.innerHeight;
    canvas.width = SPACE.w * dpr;
    canvas.height = SPACE.h * dpr;
    SPACE.ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.min(340, Math.floor((SPACE.w * SPACE.h) / 4200));
    SPACE.stars = Array.from({ length: n }, () => ({
      x: Math.random() * SPACE.w,
      y: Math.random() * SPACE.h,
      vx: (Math.random() - 0.5) * 0.18,   // 无序漂移速度
      vy: (Math.random() - 0.5) * 0.18,
      r: 0.5 + Math.random() * 1.3,       // 基础半径
      p: Math.random() * Math.PI * 2,     // 闪烁相位
      s: 0.4 + Math.random() * 1.2,       // 闪烁速率
    }));
  };
  window.addEventListener("resize", resize);
  resize();
  requestAnimationFrame(tickSpace);
}

function tickSpace(t) {
  const { ctx2d: g, stars, w, h } = SPACE;
  const dt = Math.min(50, t - (SPACE.last || t)) / 16.7;  // 以 60fps 为 1 的步长
  SPACE.last = t;
  const cx = w / 2;
  const cy = h / 2;
  const mode = avatarState;  // idle | listening | thinking | speaking

  // RTC 音频能量（speaking 波动输入）
  if (SPACE.ttsAnalyser) {
    SPACE.ttsAnalyser.getByteTimeDomainData(SPACE.ttsData);
    let sum = 0;
    const d = SPACE.ttsData;
    for (let i = 0; i < d.length; i += 4) {
      const v = (d[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / Math.max(1, d.length / 4));
    SPACE.ttsLevel = Math.max(Math.min(rms * 6, 1), SPACE.ttsLevel * 0.88);
  } else {
    SPACE.ttsLevel *= 0.9;
  }
  SPACE.micLevel *= mode === "listening" ? 1 : 0.94;  // 非倾听期麦克风能量淡出

  g.clearRect(0, 0, w, h);
  const breath = 0.5 + 0.5 * Math.sin(t * 0.0011);  // thinking 的深呼吸

  for (const st of stars) {
    let boost = 0;   // 额外亮度（0..1）
    if (mode === "listening") {
      // 专注：减速 + 向中心收拢，你声音越大收得越紧
      st.vx *= 0.985; st.vy *= 0.985;
      const pull = (0.0004 + SPACE.micLevel * 0.0035) * dt;
      st.x += (cx - st.x) * pull;
      st.y += (cy - st.y) * pull;
      boost = SPACE.micLevel * 0.5;
    } else if (mode === "thinking") {
      st.vx *= 0.99; st.vy *= 0.99;
      const pull = (0.0002 + breath * 0.0005) * dt;
      st.x += (cx - st.x) * pull;
      st.y += (cy - st.y) * pull;
      boost = breath * 0.2;
    } else if (mode === "speaking") {
      // 声波：以中心为源的径向正弦波，幅度随音频能量
      const dx = st.x - cx;
      const dy = st.y - cy;
      const dist = Math.hypot(dx, dy) || 1;
      const wave = Math.sin(dist * 0.014 - t * 0.007);
      const amp = SPACE.ttsLevel * 7 * dt;
      st.x += (dx / dist) * wave * amp;
      st.y += (dy / dist) * wave * amp;
      boost = SPACE.ttsLevel * (0.35 + 0.4 * wave);
    } else {
      // idle：无序漂移，偶尔轻拐个弯
      if (Math.random() < 0.002) {
        st.vx += (Math.random() - 0.5) * 0.06;
        st.vy += (Math.random() - 0.5) * 0.06;
      }
      st.vx = Math.max(-0.35, Math.min(0.35, st.vx));
      st.vy = Math.max(-0.35, Math.min(0.35, st.vy));
    }
    st.x += st.vx * dt;
    st.y += st.vy * dt;
    // 出界回卷
    if (st.x < -4) st.x = w + 4; else if (st.x > w + 4) st.x = -4;
    if (st.y < -4) st.y = h + 4; else if (st.y > h + 4) st.y = -4;

    const tw = 0.55 + 0.45 * Math.sin(t * 0.001 * st.s + st.p);
    const alpha = Math.min(1, tw * (0.45 + boost) + boost * 0.3);
    const rad = st.r * (1 + boost * 0.9);
    g.beginPath();
    g.arc(st.x, st.y, rad, 0, Math.PI * 2);
    g.fillStyle = `rgba(190, 214, 255, ${alpha.toFixed(3)})`;
    g.fill();
  }
  requestAnimationFrame(tickSpace);
}

// ---------------------------------------------------------------------------
// 调试角标（URL 加 ?debug=1 开启）：WebRTC 原生统计（抖动/丢包/码率）
// ---------------------------------------------------------------------------

if (new URLSearchParams(location.search).has("debug")) {
  const dbg = document.createElement("div");
  dbg.style.cssText =
    "position:fixed;right:8px;bottom:8px;background:#000c;color:#0f0;" +
    "font:12px monospace;padding:6px 10px;border-radius:6px;z-index:99;white-space:pre";
  document.body.appendChild(dbg);
  setInterval(async () => {
    if (!pc) { dbg.textContent = "RTC 未建连"; return; }
    let text = "";
    const stats = await pc.getStats();
    stats.forEach((r) => {
      if (r.type === "inbound-rtp" && r.kind === "audio") {
        text =
          `音频: ${((r.bytesReceived || 0) / 131072).toFixed(1)}Mb 丢包:${r.packetsLost || 0}\n` +
          `抖动: ${((r.jitter || 0) * 1000).toFixed(0)}ms 连接: ${pc.connectionState}`;
      }
    });
    dbg.textContent = text || "RTC 统计等待中";
  }, 1000);
}

// ---------------------------------------------------------------------------
// 连接与人设
// ---------------------------------------------------------------------------

function updatePersonaBar() {
  // 只有一个人设时隐藏切换条（chip 标签没意义）；多个人设自动恢复
  els.personaBar.style.display = personas.length > 1 ? "" : "none";
  els.personaBar.innerHTML = "";
  for (const p of personas) {
    const chip = document.createElement("button");
    chip.className = "persona-chip" + (p.id === currentPersona ? " active" : "");
    chip.textContent = p.name;
    chip.onclick = () => switchPersona(p.id);
    els.personaBar.appendChild(chip);
  }
}

function switchPersona(id) {
  if (!ws || ws.readyState !== WebSocket.OPEN || id === currentPersona) return;
  ws.send(JSON.stringify({ type: "vox.persona", id }));
  currentPersona = id;
  assistantLine = null;
  updatePersonaBar();
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  // ?alead=毫秒：新回复音频压后量（云时代等数字人渲染的遗产，默认 0，调试用）
  const q = new URLSearchParams(location.search);
  const qs = q.get("alead") ? `?alead=${encodeURIComponent(q.get("alead"))}` : "";
  ws = new WebSocket(`${proto}://${location.host}/ws${qs}`);
  ws.onopen = () => {
    wsConnected = true;
    setIndicator();
    els.micBtn.disabled = false;
    // RTC 建连等 vox.status 到了启动
  };
  ws.onclose = () => {
    wsConnected = false;
    setIndicator();
    els.micBtn.disabled = true;
    stopRTC();
    // 简单重连：对话中断后 3s 重试
    setTimeout(() => {
      if (mic) connect();
    }, 3000);
  };
  ws.onerror = () => {
    wsConnected = false;
    setIndicator();
  };
  ws.onmessage = (msg) => {
    if (typeof msg.data === "string") {
      handleTextMessage(msg.data);
    }
    // 服务端只发文本事件（音频全走 RTC），忽略任何二进制消息
  };
}

async function init() {
  initSpace();
  const res = await fetch("/api/personas");
  const data = await res.json();
  personas = data.list;
  currentPersona = data.default;
  updatePersonaBar();
  setAvatarState("idle");
  connect();
}

els.micBtn.onclick = async () => {
  if (mic) {
    stopMic();
    setAvatarState("idle");
    els.micBtn.textContent = "🎙 开始对话";
    els.micBtn.classList.remove("live");
    return;
  }
  try {
    await startMic();
    els.micBtn.textContent = "■ 结束对话";
    els.micBtn.classList.add("live");
    setIndicator();  // 开麦：显示 🎙 聆听中
    // 用户手势已发生：解锁/补播 RTC 音频元素（autoplay 策略要手势）
    ensureRtcAudio();
  } catch (e) {
    addLine("sys", "", `⚠ 麦克风不可用: ${e.message}`);
    return;
  }
};

init();
