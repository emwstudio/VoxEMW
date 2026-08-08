/* VoxEMW 数字人语音助手前端。
 *
 * 下行音画两种模式（vox.status 的 rtc.enabled 决定）：
 *   WebRTC（默认）：POST /rtc/offer 建连，音频（Opus）+ 视频（VP8）走 RTP 轨，
 *                   浏览器按时间戳原生音画同步，<video> 直挂远程流，零补偿参数
 *   WS 兜底：一路 /ws 承载——上行 JSON 麦克风/控制事件；下行 JSON 转写/音频 delta
 *            （AudioContext 拼接播放）；下行二进制 JPEG 帧（沿音频时钟调度上屏）
 *
 * 数字人缺席时显示 persona 静态肖像（纯语音模式）。
 */

"use strict";

const VOX_JS_VERSION = "20260808e";  // 排障用：Console 里 VOX_JS_VERSION 可验版本
console.log("VOXEMW JS", VOX_JS_VERSION);

const SAMPLE_RATE = 16000;
const FRAME_TYPE_JPEG = 0x01;       // 下行：数字人视频帧
const FRAME_TAG_IDLE = 0x00;        // 视频帧 tag：静音驱动的待机微动
const FRAME_TAG_SPEECH = 0x01;      // 视频帧 tag：真实音频驱动

const els = {
  status: document.getElementById("status"),
  canvas: document.getElementById("avatar-canvas"),
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
};

let ws = null;
let mic = null;
let camStream = null;
let player = null;
let personas = [];
let currentPersona = null;
let avatarOn = false;
let assistantLine = null; // 正在流式累积的助手文本行
let rtcEnabled = false;   // vox.status 下发：下行音画走 WebRTC
let pc = null;            // RTCPeerConnection（RTC 模式）
// solo 模式（?solo=1）：demo 录制用，隐藏用户画面、数字人单栏居中、不开摄像头
const SOLO_MODE = new URLSearchParams(location.search).has("solo");
if (SOLO_MODE) document.body.classList.add("solo");


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
// 视频始终有帧可放。AVTR-1 0.2s chunk + 0.2s 前瞻 + 生成 ≈ 实测 0.35s 最佳。
// 可用 ?adelay=N 覆盖(?debug=1 看帧队列深度,经常见底就调回去)
const ADELAY_OVERRIDE = (() => {
  const v = parseFloat(new URLSearchParams(location.search).get("adelay"));
  return Number.isFinite(v) && v >= 0 ? v : null;
})();
const BACKEND_ADELAY = { avtr1: 0.35 };  // 数字人 = AVTR-1（flashhead 已下线）
let avatarAudioDelay = ADELAY_OVERRIDE ?? 0.35;

// 口型-语音时间偏移补偿（帧数）：AVTR-1 模型固有口型滞后（音频包络 vs 唇部开合
// 互相关实测：爆破音段 +3 帧/120ms、自然语音段 +2 帧/80ms，官方 generate_offline
// 同幅——模型属性非链路错位），视频提前 3 帧（120ms）补偿。
// 可用 ?vlag=N 覆盖（正负号：target = pos*25 - videoLagFrames）
const VLAG_OVERRIDE = (() => {
  const v = parseInt(new URLSearchParams(location.search).get("vlag") || "", 10);
  return Number.isFinite(v) ? v : null;
})();
const BACKEND_VLAG = { avtr1: -3 };
let videoLagFrames = VLAG_OVERRIDE ?? 0;

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
// WebRTC 音画（rtc.enabled 时）：RTP 轨原生音画同步，<video> 直挂远程流
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
  const offer = await conn.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: true });
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
    addLine("sys", "", `⚠ WebRTC 建连失败（HTTP ${res.status}），回退 WS 模式看静态图`);
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
const idleQueue = [];        // idle 帧缓冲（无音频时钟，按 ~25fps 均匀释放）
let lastIdleDraw = 0;

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
  // rAF 驱动(60Hz,跟屏幕刷新):setInterval(40ms) 在 iOS 上漂移严重,
  // 放帧跟不上音频时钟,越落越多再跳帧,表现为一卡一卡
  const tick = () => {
    frameTimer = requestAnimationFrame(tick);
    // 音频播放排空：「说话」结束回待机（角标隐藏；句尾后的 idle 帧已排在
    // 说话帧队列尾沿时钟连播，见下行二进制 handler 的注释，无需在此切换通道）
    if (avatarState === "speaking" && player && player.nextStartTime <= player.ctx.currentTime) {
      setAvatarState("idle");
    }
    // idle 帧均匀释放：服务端 0.2s 一簇 5 帧推流，到就画会成簇卡顿
    //（说话结束从时钟播放切换到直画的那一刻尤其明显）
    const now = performance.now();
    if (idleQueue.length > 0 && now - lastIdleDraw >= 38) {
      lastIdleDraw = now;
      drawFrame(idleQueue.shift());
    }
    if (!player || frameQueue.length === 0) return;
    const pos = player.ctx.currentTime - responseAudioBase;  // 本 response 已播音频秒数
    if (pos < 0) return;
    const target = Math.floor(pos * 25) - videoLagFrames;
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
  setInterval(async () => {
    if (rtcEnabled) {
      // RTC 模式：WebRTC 原生统计（到帧率/抖动/码率）
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
      return;
    }
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
    // 用户开口（打断）。RTC 模式：服务端 flush 音画队列，本地无需动作；
    // WS 模式：本地播放队列+帧队列清空，嘴型跟着归位
    if (!rtcEnabled) {
      flushPlayback();
      frameQueue.length = 0;
      idleQueue.length = 0;
      needVideoBase = true;
    }
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
      appendAssistantDelta(event.delta);
    }
  },
  "response.output_text.delta"(event) {
    if (event.delta) {
      appendAssistantDelta(event.delta);
    }
  },
  "response.output_audio_transcript.done"(event) {
    // 音频模式下上游不发 delta,只在 done 里带整段文本——必须在这里显示
    if (event.transcript) {
      appendAssistantDelta(event.transcript);
    }
    assistantLine = null;
  },
  "response.output_text.done"() {
    assistantLine = null;
  },
  "response.output_audio.delta"(event) {
    // RTC 模式事件被剥了音频体（音频走音轨），但事件本身仍标志「开口」
    if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
    if (event.delta && !rtcEnabled) playPCM(int16FromBase64(event.delta));
  },
  "response.audio.delta"(event) {
    if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
    if (event.delta && !rtcEnabled) playPCM(int16FromBase64(event.delta));
  },
  "response.done"() {
    assistantLine = null;
    needVideoBase = true;  // 下一个音频 delta 开启新 response,重设视频对齐基准
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
    if (ADELAY_OVERRIDE == null && event.avatar_backend) {
      avatarAudioDelay = BACKEND_ADELAY[event.avatar_backend] ?? 0.8;
    }
    if (VLAG_OVERRIDE == null && event.avatar_backend) {
      videoLagFrames = BACKEND_VLAG[event.avatar_backend] ?? 0;
    }
    els.fallback.classList.toggle("hidden", avatarOn);
    updatePersonaBar();
    showStill(currentPersona);
    // 下行音画模式：RTC（原生同步）或 WS 兜底（手搓帧调度）
    rtcEnabled = !!(event.rtc && event.rtc.enabled);
    if (rtcEnabled) {
      startRTC().catch((e) =>
        addLine("sys", "", `⚠ WebRTC 建连异常: ${e.message}`)
      );
    } else {
      startFramePlayback();
    }
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
  // ?rtc=0：本会话回退纯 WS 模式（弱网 A/B 对比用，丢帧式降级比 RTC 卡死体感好）
  // ?alead=毫秒：音频压后量（音画对齐补偿，服务端调度器执行），默认 250
  const q = new URLSearchParams(location.search);
  const params = new URLSearchParams();
  if (q.get("rtc") === "0") params.set("rtc", "0");
  if (q.get("alead")) params.set("alead", q.get("alead"));
  const qs = params.toString();
  ws = new WebSocket(`${proto}://${location.host}/ws${qs ? "?" + qs : ""}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    setStatus("已连接", "live");
    els.micBtn.disabled = false;
    // 帧播放/RTC 建连等 vox.status 到了按模式启动
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
    } else {
      const bytes = new Uint8Array(msg.data);
      if (bytes[0] === FRAME_TYPE_JPEG) {
        // tag 0x00=idle：无音频时钟（待机微动/倾听反应），进 idleQueue 按
        // ~25fps 均匀释放（成簇直画会卡）；tag 0x01=speech：照常排队
        if (bytes[1] === FRAME_TAG_IDLE) {
          // 句尾平滑（2026-08-04 终版）：说话帧队列未空（或音频仍在播）时，idle 帧
          // 排进同一队列、沿音频时钟 25fps 连播——引擎内容本就连贯（尾帧回落→idle
          // 微动），按到达顺序播即无跳变也无断供。这正是官方 demo 的播放模型
          // （帧按内容顺序持续上屏）。完全空闲（无音频时钟）才走 idleQueue 直画。
          // 旧实现两处败笔：①积压期丢 idle 帧→接管瞬间姿态跳变；②drain 后才
          // flush→队列耗尽到尾帧到达之间定格。
          if (frameQueue.length > 0 || (player && player.nextStartTime > player.ctx.currentTime)) {
            enqueueFrame(bytes.subarray(2));
          } else {
            if (idleQueue.length >= 10) idleQueue.shift();  // 满则丢最旧
            idleQueue.push(bytes.subarray(2));
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
    // 用户手势已发生：给数字人视频开声音（页面加载时只能 muted 自动播）
    els.avatarVideo.muted = false;
    els.avatarVideo.play().catch(() => {});
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
