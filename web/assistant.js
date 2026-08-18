/* VoxEMW 数字人语音助手前端。
 *
 * 下行音画：WebRTC——POST /rtc/offer 建连，音频（Opus）+ 视频（H264）走 RTP 轨，
 *           浏览器按时间戳原生音画同步，<video> 直挂远程流，零补偿参数。
 *           音画对齐在服务端 AVSyncScheduler（?alead=毫秒 调音频压后量）。
 * WS /ws：上行麦克风/控制事件；下行转写/状态事件（音频体已剥离，走音轨）。
 * 数字人缺席时显示 persona 静态肖像（纯语音模式）。
 */

"use strict";

const VOX_JS_VERSION = "20260816b";  // 排障用：Console 里 VOX_JS_VERSION 可验版本
console.log("VOXEMW JS", VOX_JS_VERSION);

const SAMPLE_RATE = 16000;

const els = {
  status: document.getElementById("status"),
  avatarVideo: document.getElementById("avatar-video"),
  still: document.getElementById("avatar-still"),
  avatarWrap: document.querySelector(".avatar-wrap"),
  fallback: document.getElementById("avatar-fallback"),
  avatarLabel: document.getElementById("avatar-label"),
  avatarState: document.getElementById("avatar-state"),
  camWrap: document.getElementById("cam-wrap"),
  userCam: document.getElementById("user-cam"),
  personaBar: document.getElementById("persona-bar"),
  transcript: document.getElementById("transcript"),
  micBtn: document.getElementById("mic-btn"),
  imgUploadBtn: document.getElementById("img-upload-btn"),
  imgUpload: document.getElementById("img-upload"),
};

let ws = null;
let mic = null;
let camStream = null;
let personas = [];
let currentPersona = null;
let avatarOn = false;
let assistantLine = null; // 正在流式累积的助手文本行
let lineGotDeltas = false; // 当前行已收到逐字 delta（新上游 delta+done 双发，done 只收尾不重复上屏）
let rtcEnabled = false;   // vox.status 下发：下行音画走 WebRTC
let pc = null;            // RTCPeerConnection
// solo 模式（?solo=1）：demo 录制用，隐藏用户画面、数字人单栏居中、不开摄像头
const SOLO_MODE = new URLSearchParams(location.search).has("solo");
if (SOLO_MODE) document.body.classList.add("solo");


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

// ---------------------------------------------------------------------------
// WebRTC 音画：RTP 轨原生音画同步，<video> 直挂远程流
// ---------------------------------------------------------------------------

async function startRTC() {
  if (pc) return;  // 一次 WS 会话只建一路
  // ICE 配置现取：服务端下发本地 coturn 凭证
  let iceServers = [];
  try {
    iceServers = (await (await fetch("/rtc/ice")).json()).ice_servers || [];
  } catch (_) { /* 取不到就裸 host candidate，LAN 还能用 */ }
  // SSH 隧道场景（页面在 localhost）：强制 relay——host candidate 是双方各自的
  // 私网/环回地址，互指必败；媒体走 coturn TCP 中继（隧道转发 3478）。
  // LAN 直连场景用默认 all：host candidate 直配，无需 TURN。
  const isTunnel = ["localhost", "127.0.0.1"].includes(location.hostname);
  const conn = new RTCPeerConnection({
    iceServers,
    iceTransportPolicy: isTunnel ? "relay" : "all",
  });
  pc = conn;
  const rtcStream = new MediaStream();  // aiortc 音/视分两个 stream 发，收进同一个
  const trackKinds = [];
  conn.ontrack = (e) => {
    rtcStream.addTrack(e.track);
    trackKinds.push(e.track.kind);
    els.avatarVideo.srcObject = rtcStream;
    els.avatarWrap.classList.add("webrtc", "streaming");
    els.still.classList.add("hidden");
    // Chrome 自动播放策略：带音轨的 <video> 无用户手势不许出声自动播。
    // 元素先 muted 自动播（画面能出），用户点了开始对话（有手势）再开声音
    els.avatarVideo.play().catch(() => {});
  };
  conn.onconnectionstatechange = () => {
    if (["failed", "closed"].includes(conn.connectionState)) {
      // 媒体链路断了（隧道抖动等）：2s 后自动重建，跟 WS 重连一个思路。
      // 用局部变量 conn 判定/操作——全局 pc 可能被重入的新连接占用
      if (pc === conn) {
        pc = null;
        els.avatarVideo.srcObject = null;
        els.avatarWrap.classList.remove("webrtc", "streaming");
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
  conn.addTransceiver("video", { direction: "recvonly" });
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
        tracks: trackKinds,
        video: {
          readyState: els.avatarVideo.readyState,
          paused: els.avatarVideo.paused,
          muted: els.avatarVideo.muted,
          size: `${els.avatarVideo.videoWidth}x${els.avatarVideo.videoHeight}`,
        },
        candidates: cands,
      }),
    }).catch(() => {});
  }, 12000);
}

function stopRTC() {
  if (!pc) return;
  pc.close();
  pc = null;
  els.avatarVideo.srcObject = null;
  els.avatarWrap.classList.remove("webrtc", "streaming");
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
    if (ws && ws.readyState === WebSocket.OPEN) {
      const int16 = floatTo16BitPCM(e.data);
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
// 摄像头（左侧画面，仅展示）
// ---------------------------------------------------------------------------

async function startCamera() {
  camStream = await navigator.mediaDevices.getUserMedia({ video: true });
  els.userCam.srcObject = camStream;
  els.camWrap.classList.add("live");
}

function stopCamera() {
  if (!camStream) return;
  camStream.getTracks().forEach((t) => t.stop());
  camStream = null;
  els.userCam.srcObject = null;
  els.camWrap.classList.remove("live");
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
// 数字人画面（RTC 模式：<video> 挂远程流；未连接时显示静态肖像）
// ---------------------------------------------------------------------------

function showStill(personaId) {
  // 注意：不动 streaming 类——RTC 流的生命周期由 ontrack/断连事件管理。
  // 这里若移除 streaming，persona 切换/换图后 video 会被永久定格成静态图
  //（ontrack 只在建连时触发一次，streaming 再也加不回来）——2026-08-18 踩坑
  const persona = personas.find((p) => p.id === personaId);
  if (persona && persona.has_image) {
    els.still.src = `/api/personas/${personaId}/image`;
    els.still.classList.remove("hidden");
  } else {
    els.still.removeAttribute("src");
  }
}

// ---------------------------------------------------------------------------
// 对话状态角标：listening（用户说话中）/ thinking（说完到开口前）显示角标，
// speaking / idle 隐藏。画面动感由 avatar 服务驱动：listening 时用户麦克风
// 音频经 listen 轨喂给模型产生点头/注视等倾听反应，thinking/calm 纯静音
// ---------------------------------------------------------------------------

let avatarState = "idle"; // idle | listening | thinking | speaking

function setAvatarState(state) {
  avatarState = state;
  const el = els.avatarState;
  if (state === "listening") {
    el.textContent = "👂 倾听中…";
    el.className = "state-listening";
  } else if (state === "thinking") {
    el.textContent = "🤔 思考中…";
    el.className = "state-thinking";
  } else {
    el.className = "hidden";
  }
}

// ---------------------------------------------------------------------------
// Realtime 事件处理
// ---------------------------------------------------------------------------

const realtimeHandlers = {
  "input_audio_buffer.speech_started"() {
    // 用户开口（打断）：服务端 flush 音画队列，本地只需更新状态
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
    avatarOn = event.avatar === "on";
    currentPersona = event.persona;
    els.fallback.classList.toggle("hidden", avatarOn);
    updatePersonaBar();
    showStill(currentPersona);
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
// 调试角标（URL 加 ?debug=1 开启）：WebRTC 原生统计（到帧率/抖动/丢包/码率）
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
      if (r.type === "inbound-rtp" && r.kind === "video") {
        text =
          `视频: ${(r.framesPerSecond || 0).toFixed(1)}fps 解码:${r.framesDecoded || 0} 丢包:${r.packetsLost || 0}\n` +
          `抖动: ${((r.jitter || 0) * 1000).toFixed(0)}ms 累计: ${((r.bytesReceived || 0) / 131072).toFixed(1)}Mb\n` +
          `连接: ${pc.connectionState}`;
      }
    });
    dbg.textContent = text || "RTC 统计等待中";
  }, 1000);
}

// ---------------------------------------------------------------------------
// 连接与人设
// ---------------------------------------------------------------------------

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = `status ${cls || ""}`;
}

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
  const cur = personas.find((p) => p.id === currentPersona);
  if (cur) els.avatarLabel.textContent = cur.label || cur.name;
}

function switchPersona(id) {
  if (!ws || ws.readyState !== WebSocket.OPEN || id === currentPersona) return;
  ws.send(JSON.stringify({ type: "vox.persona", id }));
  currentPersona = id;
  assistantLine = null;
  updatePersonaBar();
  showStill(id);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  // ?alead=毫秒：音频压后量（音画对齐补偿，服务端调度器执行），默认 250
  const q = new URLSearchParams(location.search);
  const qs = q.get("alead") ? `?alead=${encodeURIComponent(q.get("alead"))}` : "";
  ws = new WebSocket(`${proto}://${location.host}/ws${qs}`);
  ws.onopen = () => {
    setStatus("已连接", "live");
    els.micBtn.disabled = false;
    // RTC 建连等 vox.status 到了启动
  };
  ws.onclose = () => {
    setStatus("已断开", "warn");
    els.micBtn.disabled = true;
    stopRTC();
    els.avatarWrap.classList.remove("streaming");
    // 简单重连：对话中断后 3s 重试
    setTimeout(() => {
      if (mic) connect();
    }, 3000);
  };
  ws.onerror = () => setStatus("连接错误", "warn");
  ws.onmessage = (msg) => {
    if (typeof msg.data === "string") {
      handleTextMessage(msg.data);
    }
    // 服务端不再发二进制帧（音画全走 RTC），忽略任何二进制消息
  };
}

async function init() {
  const res = await fetch("/api/personas");
  const data = await res.json();
  personas = data.list;
  currentPersona = data.default;
  avatarOn = data.avatar === "on";
  els.fallback.classList.toggle("hidden", avatarOn);
  updatePersonaBar();
  showStill(currentPersona);
  setAvatarState("idle");
  connect();
}

// 换图免重启：上传新肖像 → 服务端覆盖文件并热推 avatar 服务（引擎 set_image）
els.imgUploadBtn.onclick = () => els.imgUpload.click();
els.imgUpload.onchange = async () => {
  const file = els.imgUpload.files[0];
  els.imgUpload.value = "";
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file, "ref.png");
  try {
    const r = await (await fetch(`/api/personas/${currentPersona}/image`, {
      method: "POST", body: fd,
    })).json();
    if (r.ok) {
      addLine("sys", "", r.hot ? "🖼 新形象已生效" : "🖼 已保存，下次连接生效");
      // 刷新静态图（服务端 no-cache + 查询参数双保险）
      els.still.src = `/api/personas/${currentPersona}/image?t=${Date.now()}`;
    } else {
      addLine("sys", "", `⚠ 换图失败: ${r.error || "未知错误"}`);
    }
  } catch (e) {
    addLine("sys", "", `⚠ 换图失败: ${e.message}`);
  }
};

els.micBtn.onclick = async () => {
  if (mic) {
    stopMic();
    stopCamera();
    setAvatarState("idle");
    els.micBtn.textContent = "🎙 开始对话";
    els.micBtn.classList.remove("live");
    setStatus("已连接（麦克风关）", "");
    return;
  }
  try {
    await startMic();
    els.micBtn.textContent = "■ 结束对话";
    els.micBtn.classList.add("live");
    setStatus("聆听中", "live");
    // 用户手势已发生：给数字人视频开声音（页面加载时只能 muted 自动播）
    els.avatarVideo.muted = false;
    els.avatarVideo.play().catch(() => {});
  } catch (e) {
    addLine("sys", "", `⚠ 麦克风不可用: ${e.message}`);
    return;
  }
  if (SOLO_MODE) return;  // solo 模式不开摄像头
  try {
    await startCamera();
  } catch (e) {
    // 摄像头被拒只降级：左侧显示占位，语音对话不受影响
    addLine("sys", "", `⚠ 摄像头不可用（纯语音继续）: ${e.message}`);
  }
};

init();
