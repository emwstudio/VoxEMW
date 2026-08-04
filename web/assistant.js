/* VoxEMW 数字人语音助手前端。
 *
 * 一路 ws（/ws，orchestrator）承载三种流量：
 *   上行 JSON：OpenAI Realtime 事件（input_audio_buffer.append / response.cancel）
 *              + 自定义 {"type":"vox.persona","id":...} 人设切换
 *   下行 JSON：Realtime 事件透传（转写/音频 delta/打断）+ {"type":"vox.status",...}
 *   下行二进制：0x01 + tag(1B) + JPEG 数字人视频帧
 *               （tag 0x00=idle 待机微动，直接画；0x01=speech 说话帧，进口型同步队列）
 *
 * 音频：麦克风 AudioWorklet 16kHz int16 上行；TTS PCM16 delta 在 AudioContext
 * 时间轴上无缝拼接播放；speech_started（打断）时清空播放队列。
 * 视频：JPEG 帧到就画；数字人缺席时显示 persona 静态肖像（纯语音模式）。
 */

"use strict";

const SAMPLE_RATE = 16000;
const FRAME_TYPE_JPEG = 0x01;       // 下行：数字人视频帧
const FRAME_TYPE_USER_JPEG = 0x02;  // 上行：用户摄像头截帧（打分环节）
const FRAME_TAG_IDLE = 0x00;        // 视频帧 tag：静音驱动的待机微动
const FRAME_TAG_SPEECH = 0x01;      // 视频帧 tag：真实音频驱动

const els = {
  status: document.getElementById("status"),
  canvas: document.getElementById("avatar-canvas"),
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
};

let ws = null;
let mic = null;
let camStream = null;
let player = null;
let personas = [];
let currentPersona = null;
let avatarOn = false;
let assistantLine = null; // 正在流式累积的助手文本行
// solo 模式（?solo=1）：demo 录制用，隐藏用户画面、数字人单栏居中、不开摄像头
const SOLO_MODE = new URLSearchParams(location.search).has("solo");
if (SOLO_MODE) document.body.classList.add("solo");

// 截帧打分：persona 说出暗号（vision.trigger）→ 截一帧摄像头发给 orchestrator
let visionOn = false;
let visionTrigger = "让我好好看看你";
let visionCooldownUntil = 0;
let responseText = ""; // 当前 response 已累积的助手文本（暗号在这里面找）

// ---------------------------------------------------------------------------
// PCM 编解码
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

function int16FromBase64(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

// ---------------------------------------------------------------------------
// 播放（无缝拼接 + 打断清空）
// ---------------------------------------------------------------------------

function ensurePlayer() {
  if (!player) {
    player = { ctx: new AudioContext({ sampleRate: SAMPLE_RATE }), nextStartTime: 0 };
  }
  if (player.ctx.state === "suspended") player.ctx.resume();
  return player;
}

// 口型同步模型:TTS 以 ~2x 实时速度流式送音频,播放链会积压(实测 3-4s),
// 各自计时必然漂移 → 视频帧节奏从属于音频播放时钟:音频播到第几秒就放第几帧。
// AVATAR_AUDIO_DELAY 保留基础延迟,保证音频播放位置始终落后于视频供帧点,
// 视频始终有帧可放。默认值按 avatar 后端固有延迟选择（vox.status 下发 avatar_backend）：
// flashhead 攒 0.96s chunk + 生成 ≈ 实测 0.8s 最佳（flash_attn 后）；
// avtr1 0.2s chunk + 0.2s 前瞻 + 生成 ≈ 0.35s。
// 可用 ?adelay=N 覆盖(?debug=1 看帧队列深度,经常见底就调回去)
const ADELAY_OVERRIDE = (() => {
  const v = parseFloat(new URLSearchParams(location.search).get("adelay"));
  return Number.isFinite(v) && v >= 0 ? v : null;
})();
const BACKEND_ADELAY = { avtr1: 0.35, flashhead: 0.8 };
let avatarAudioDelay = ADELAY_OVERRIDE ?? 0.8;

function playPCM(int16) {
  const p = ensurePlayer();
  const buf = p.ctx.createBuffer(1, int16.length, SAMPLE_RATE);
  const data = buf.getChannelData(0);
  for (let i = 0; i < int16.length; i++) data[i] = int16[i] / 0x8000;
  const src = p.ctx.createBufferSource();
  src.buffer = buf;
  src.connect(p.ctx.destination);
  const prevEnd = p.nextStartTime;  // 本 delta 排程前的音频链尾（= 上一段回复的播放结束点）
  const start = Math.max(p.ctx.currentTime + (avatarOn ? avatarAudioDelay : 0.02), prevEnd);
  src.start(start);
  p.nextStartTime = start + buf.duration;
  if (needVideoBase) {
    if (prevEnd - p.ctx.currentTime < 0.3) {
      // 常规：上一段回复已播完。本 response 首个音频 delta:记录它在 ctx 时间轴上的
      // 起点作为视频对齐基准。同时清空队列:里面滞留的是上一回复的"闭嘴尾帧"
      // (句尾零填充生成),不清掉会被当作本回复的开头播出,嘴型整体慢 ~1s
      responseAudioBase = start;
      videoFrameIdx = 0;
      frameQueue.length = 0;
    } else {
      // 注入式连续回复（垫场→打分）：生成远快于播放，新回复 delta 到达时上一段
      // 还在播。此时绝不能重锚+清队——上一段的真帧被扔掉、数字人又不会补发，
      // 视频就会半路定格（音频还在放）。音频链是连续的，帧按到达顺序从属同一
      // 时钟即可；只砍掉旧回复的"闭嘴尾帧"（零填充生成，对应播放中不存在的静音段）
      const oldTotalFrames = Math.floor((prevEnd - responseAudioBase) * 25);
      const keep = Math.max(0, oldTotalFrames - videoFrameIdx);
      if (frameQueue.length > keep) frameQueue.length = keep;
    }
    needVideoBase = false;
  }
}

function flushPlayback() {
  // 打断：整个 AudioContext 关掉重建，已排程的音频全部作废
  if (player) {
    player.ctx.close();
    player = null;
  }
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
// 摄像头（左侧画面 + 打分截帧）
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

// 截一帧（与显示一致做镜像），缩放到最长边 640，0x02 + JPEG 发给 orchestrator
function sendCameraSnapshot() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !camStream) return;
  const video = els.userCam;
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  visionBusy = true;  // 直到 vox.vision scored/error/off 才恢复暗号检测
  const scale = Math.min(1, 640 / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.round(w * scale);
  c.height = Math.round(h * scale);
  const cx = c.getContext("2d");
  cx.translate(c.width, 0);
  cx.scale(-1, 1);
  cx.drawImage(video, 0, 0, c.width, c.height);
  const b64 = c.toDataURL("image/jpeg", 0.8).split(",")[1];
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length + 1);
  bytes[0] = FRAME_TYPE_USER_JPEG;
  for (let i = 0; i < bin.length; i++) bytes[i + 1] = bin.charCodeAt(i);
  ws.send(bytes.buffer);
}

// 助手本轮文本里出现暗号 → 延迟 1.2s（留摆姿势时间）截帧；10s 冷却防连发。
// visionBusy 期间（垫场+打分回复）不再检测——打分回复里可能又提到暗号句式，
// 不挡住会无限循环打分
let visionBusy = false;

function maybeSnapshot() {
  if (!visionOn || !camStream || visionBusy) return;
  if (!responseText.includes(visionTrigger)) return;
  const now = performance.now();
  if (now < visionCooldownUntil) return;
  visionCooldownUntil = now + 10000;
  setTimeout(sendCameraSnapshot, 1200);
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
// 数字人画面
// ---------------------------------------------------------------------------

// 画布上下文一次性创建：alpha:false 跳过上屏混合、desynchronized 降撕裂
//（浏览器不支持会自动忽略）；平滑质量拉满——移动端糊就是默认平滑质量太低
const avatarCtx = els.canvas.getContext("2d", { alpha: false, desynchronized: true });
avatarCtx.imageSmoothingEnabled = true;
avatarCtx.imageSmoothingQuality = "high";

let frameDecodeMs = 0;
let frameDecodeCount = 0;

async function drawFrame(jpegBytes) {
  try {
    const t0 = performance.now();
    const bitmap = await createImageBitmap(new Blob([jpegBytes], { type: "image/jpeg" }));
    frameDecodeMs += performance.now() - t0;
    frameDecodeCount++;
    avatarCtx.drawImage(bitmap, 0, 0, els.canvas.width, els.canvas.height);
    bitmap.close();
    els.avatarWrap.classList.add("streaming");
  } catch {
    /* 坏帧丢弃 */
  }
}

// 视频帧队列 + 音画对齐：帧按到达顺序编号（每个 response 从 0 起），
// 播放计时器按"音频已播秒数 × 25fps"放帧；视频落后音频 >1s 时跳帧追赶
const FRAME_QUEUE_MAX = 1000;  // ~40s 内容。供帧天然快于播放(TTS 1.5x 流式),
                               // 队列会持续增长,必须给足深度;丢帧绝不递增序号
                               // (否则嘴型整体超前,越丢越乱)
const frameQueue = [];
let frameTimer = null;
let responseAudioBase = 0;   // 当前 response 音频在 ctx 时间轴上的起点
let videoFrameIdx = 0;       // 当前 response 已消费（播放或丢弃）的帧序号
let needVideoBase = true;    // 下一个音频 delta 是 response 起点（response.done/打断后置位）

function enqueueFrame(jpegBytes) {
  frameRecvCount++;
  if (frameQueue.length >= FRAME_QUEUE_MAX) {
    frameQueue.shift();  // 极端情况丢最旧帧:嘴型最多滞后,绝不超前(滞后比超前自然)
    return;
  }
  frameQueue.push(jpegBytes);
}

function startFramePlayback() {
  if (frameTimer) return;
  // 画面在音频时钟基础上的滞后帧数(缓冲垫,吸供帧抖动)。
  // 口型偏移感知阈值 ~0.1-0.15s,实测 vlag=0 口型最准且供帧跟得上;可用 ?vlag=N 覆盖
  const VIDEO_LAG_FRAMES = parseInt(new URLSearchParams(location.search).get("vlag") || "0", 10);
  // rAF 驱动(60Hz,跟屏幕刷新):setInterval(40ms) 在 iOS 上漂移严重,
  // 放帧跟不上音频时钟,越落越多再跳帧,表现为一卡一卡
  const tick = () => {
    frameTimer = requestAnimationFrame(tick);
    // 音频播放排空：「说话」结束回待机（角标隐藏，画面由 idle 微动接管）
    if (avatarState === "speaking" && player && player.nextStartTime <= player.ctx.currentTime) {
      setAvatarState("idle");
    }
    if (!player || frameQueue.length === 0) return;
    const pos = player.ctx.currentTime - responseAudioBase;  // 本 response 已播音频秒数
    if (pos < 0) return;
    const target = Math.floor(pos * 25) - VIDEO_LAG_FRAMES;
    // 落后 >1s：跳到最新，保同步优先于完整
    while (frameQueue.length > 0 && videoFrameIdx < target - 25) {
      frameQueue.shift();
      videoFrameIdx++;
    }
    if (frameQueue.length > 0 && videoFrameIdx <= target) {
      drawFrame(frameQueue.shift());
      videoFrameIdx++;
    }
  };
  frameTimer = requestAnimationFrame(tick);
}

// 调试角标（URL 加 ?debug=1 开启）：音频缓冲时长 / 帧队列深度 / 实际到帧率
let frameRecvCount = 0;
let frameRecvWindowStart = performance.now();
if (new URLSearchParams(location.search).has("debug")) {
  const dbg = document.createElement("div");
  dbg.style.cssText =
    "position:fixed;right:8px;bottom:8px;background:#000c;color:#0f0;" +
    "font:12px monospace;padding:6px 10px;border-radius:6px;z-index:99;white-space:pre";
  document.body.appendChild(dbg);
  setInterval(() => {
    const now = performance.now();
    const fps = frameRecvCount / ((now - frameRecvWindowStart) / 1000);
    frameRecvCount = 0;
    frameRecvWindowStart = now;
    const audioBuf = player ? Math.max(0, player.nextStartTime - player.ctx.currentTime) : 0;
    const pos = player ? Math.max(0, player.ctx.currentTime - responseAudioBase) : 0;
    const decodeAvg = frameDecodeCount ? (frameDecodeMs / frameDecodeCount) : 0;
    frameDecodeMs = 0;
    frameDecodeCount = 0;
    dbg.textContent =
      `audio缓冲: ${audioBuf.toFixed(2)}s\n帧队列: ${frameQueue.length}\n到帧率: ${fps.toFixed(1)}fps\n` +
      `解码: ${decodeAvg.toFixed(0)}ms/帧\n音频位置: ${pos.toFixed(2)}s\n帧序号: ${videoFrameIdx} (目标 ${Math.floor(pos * 25)})`;
  }, 1000);
}

function showStill(personaId) {
  els.avatarWrap.classList.remove("streaming");
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
// speaking / idle 隐藏。画面动感由 avatar 服务驱动：倾听/思考时循环 persona
// 嘟囔音频（TTS 预合成）产生真实沉吟/附和微动，待机时纯静音基线微动
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
    // 用户开口（打断）：本地播放队列清空，助手文本行封口
    flushPlayback();
    frameQueue.length = 0;  // 视频帧队列一并清空，嘴型跟着归位
    needVideoBase = true;
    assistantLine = null;
    responseText = "";
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
      appendAssistantDelta(event.delta);
      responseText += event.delta;
    }
  },
  "response.output_text.delta"(event) {
    if (event.delta) {
      appendAssistantDelta(event.delta);
      responseText += event.delta;
    }
  },
  "response.output_audio_transcript.done"(event) {
    // 音频模式下上游不发 delta,只在 done 里带整段文本——必须在这里显示
    if (event.transcript) {
      appendAssistantDelta(event.transcript);
      responseText += event.transcript;
    }
    assistantLine = null;
    maybeSnapshot();
  },
  "response.output_text.done"() {
    assistantLine = null;
    maybeSnapshot();
  },
  "response.output_audio.delta"(event) {
    if (event.delta) {
      if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
      playPCM(int16FromBase64(event.delta));
    }
  },
  "response.audio.delta"(event) {
    if (event.delta) {
      if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
      playPCM(int16FromBase64(event.delta));
    }
  },
  "response.done"() {
    assistantLine = null;
    needVideoBase = true;  // 下一个音频 delta 开启新 response,重设视频对齐基准
    if (avatarState === "thinking") setAvatarState("idle");  // 无音频回复的兜底
    maybeSnapshot();
    responseText = "";
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
    if (ADELAY_OVERRIDE == null && event.avatar_backend) {
      avatarAudioDelay = BACKEND_ADELAY[event.avatar_backend] ?? 0.8;
    }
    els.fallback.classList.toggle("hidden", avatarOn);
    updatePersonaBar();
    showStill(currentPersona);
    return;
  }
  if (event.type === "vox.vision") {
    if (event.state === "scored" || event.state === "error" || event.state === "off") {
      visionBusy = false;
    }
    const msg = {
      scoring: "📷 峰哥打分中…（麦克风暂闭，插话不会打断他）",
      scored: "📷 打分完毕，麦克风恢复",
      busy: "📷 上一张还没看完，别急",
      error: "📷 没看清，换个光线再让他评",
      off: "📷 打分功能未启用（服务器缺 KIMI_API_KEY）",
    }[event.state];
    if (msg) addLine("sys", "", msg);
    return;
  }
  const handler = realtimeHandlers[event.type];
  if (handler) handler(event);
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
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    setStatus("已连接", "live");
    els.micBtn.disabled = false;
    startFramePlayback();
  };
  ws.onclose = () => {
    setStatus("已断开", "warn");
    els.micBtn.disabled = true;
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
    } else {
      const bytes = new Uint8Array(msg.data);
      if (bytes[0] === FRAME_TYPE_JPEG) {
        // tag 0x00=idle：待机微动帧，不进唇同步队列直接画（首个 TTS delta 前
        // 在飞的静音帧若进队列会错位唇同步）；tag 0x01=speech：照常排队
        if (bytes[1] === FRAME_TAG_IDLE) {
          // 尾部音频还在播时丢 idle 帧：response.done 后 avatar 已回 idle 生成，
          // 但浏览器还在播缓冲的说话帧，直画 idle 会盖住唇同步画面
          if (!player || player.nextStartTime <= player.ctx.currentTime + 0.05) {
            drawFrame(bytes.subarray(2));
          }
        } else {
          enqueueFrame(bytes.subarray(2));
        }
      }
    }
  };
}

async function init() {
  const res = await fetch("/api/personas");
  const data = await res.json();
  personas = data.list;
  currentPersona = data.default;
  avatarOn = data.avatar === "on";
  visionOn = data.vision === "on";
  if (data.trigger) visionTrigger = data.trigger;
  els.fallback.classList.toggle("hidden", avatarOn);
  updatePersonaBar();
  showStill(currentPersona);
  setAvatarState("idle");
  connect();
}

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
  } catch (e) {
    addLine("sys", "", `⚠ 麦克风不可用: ${e.message}`);
    return;
  }
  if (SOLO_MODE) return;  // solo 模式不开摄像头（画面隐藏，截帧打分也不可用）
  try {
    await startCamera();
  } catch (e) {
    // 摄像头被拒只降级：左侧显示占位，语音对话不受影响
    addLine("sys", "", `⚠ 摄像头不可用（纯语音继续）: ${e.message}`);
  }
};

init();
