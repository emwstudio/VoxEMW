/* VoxEMW 数字人语音助手前端。
 *
 * 一路 ws（/ws，orchestrator）承载三种流量：
 *   上行 JSON：OpenAI Realtime 事件（input_audio_buffer.append / response.cancel）
 *              + 自定义 {"type":"vox.persona","id":...} 人设切换
 *   下行 JSON：Realtime 事件透传（转写/音频 delta/打断）+ {"type":"vox.status",...}
 *   下行二进制：0x01 + JPEG 数字人视频帧
 *
 * 音频：麦克风 AudioWorklet 16kHz int16 上行；TTS PCM16 delta 在 AudioContext
 * 时间轴上无缝拼接播放；speech_started（打断）时清空播放队列。
 * 视频：JPEG 帧到就画；数字人缺席时显示 persona 静态肖像（纯语音模式）。
 */

"use strict";

const SAMPLE_RATE = 16000;
const FRAME_TYPE_JPEG = 0x01;

const els = {
  status: document.getElementById("status"),
  canvas: document.getElementById("avatar-canvas"),
  still: document.getElementById("avatar-still"),
  avatarWrap: document.querySelector(".avatar-wrap"),
  fallback: document.getElementById("avatar-fallback"),
  personaBar: document.getElementById("persona-bar"),
  transcript: document.getElementById("transcript"),
  micBtn: document.getElementById("mic-btn"),
};

let ws = null;
let mic = null;
let player = null;
let personas = [];
let currentPersona = null;
let avatarOn = false;
let assistantLine = null; // 正在流式累积的助手文本行

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
// 视频始终有帧可放(avatar 攒满 0.96s chunk + ~0.45s 生成 ≈ 0.9s 固有延迟)。
// 实测 1.0s 效果最佳(0.8s 偶发供帧不足)
const AVATAR_AUDIO_DELAY = 1.0;

function playPCM(int16) {
  const p = ensurePlayer();
  const buf = p.ctx.createBuffer(1, int16.length, SAMPLE_RATE);
  const data = buf.getChannelData(0);
  for (let i = 0; i < int16.length; i++) data[i] = int16[i] / 0x8000;
  const src = p.ctx.createBufferSource();
  src.buffer = buf;
  src.connect(p.ctx.destination);
  const start = Math.max(p.ctx.currentTime + (avatarOn ? AVATAR_AUDIO_DELAY : 0.02), p.nextStartTime);
  src.start(start);
  p.nextStartTime = start + buf.duration;
  if (needVideoBase) {
    // 本 response 首个音频 delta:记录它在 ctx 时间轴上的起点作为视频对齐基准。
    // 同时清空队列:里面滞留的是上一回复的"闭嘴尾帧"(句尾零填充生成),
    // 不清掉会被当作本回复的开头播出,嘴型整体慢 ~1s
    responseAudioBase = start;
    videoFrameIdx = 0;
    frameQueue.length = 0;
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
// Realtime 事件处理
// ---------------------------------------------------------------------------

const realtimeHandlers = {
  "input_audio_buffer.speech_started"() {
    // 用户开口（打断）：本地播放队列清空，助手文本行封口
    flushPlayback();
    frameQueue.length = 0;  // 视频帧队列一并清空，嘴型跟着归位
    needVideoBase = true;
    assistantLine = null;
  },
  "conversation.item.input_audio_transcription.completed"(event) {
    const text = (event.transcript || "").trim();
    if (text) addLine("user", "你:", text);
  },
  "response.output_audio_transcript.delta"(event) {
    if (event.delta) appendAssistantDelta(event.delta);
  },
  "response.output_text.delta"(event) {
    if (event.delta) appendAssistantDelta(event.delta);
  },
  "response.output_audio_transcript.done"(event) {
    // 音频模式下上游不发 delta,只在 done 里带整段文本——必须在这里显示
    if (event.transcript) appendAssistantDelta(event.transcript);
    assistantLine = null;
  },
  "response.output_text.done"() {
    assistantLine = null;
  },
  "response.output_audio.delta"(event) {
    if (event.delta) playPCM(int16FromBase64(event.delta));
  },
  "response.audio.delta"(event) {
    if (event.delta) playPCM(int16FromBase64(event.delta));
  },
  "response.done"() {
    assistantLine = null;
    needVideoBase = true;  // 下一个音频 delta 开启新 response,重设视频对齐基准
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
      if (bytes[0] === FRAME_TYPE_JPEG) enqueueFrame(bytes.subarray(1));
    }
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
  connect();
}

els.micBtn.onclick = async () => {
  if (mic) {
    stopMic();
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
  }
};

init();
