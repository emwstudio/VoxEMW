/* 河南妮儿语音助手前端（星空 + VRM 数字人）。
 *
 * 下行音频：WebRTC——POST /rtc/offer 建连，音频（Opus）走 RTP 轨，
 *           挂隐藏 <audio> 播放（Chrome 对 RTC 音轨的解码只在媒体元素上才启动）。
 * WS /ws：上行麦克风/控制事件；下行转写/状态事件（音频体已剥离，走音轨）。
 * 星空：全屏 canvas 跟随对话状态——idle 无序漂移 / listening 向中心收拢的
 *       专注波动（随麦克风能量）/ speaking 随 RTC 音频能量的径向声波。
 */

"use strict";

const VOX_JS_VERSION = "20260829j";  // 排障用：Console 里 VOX_JS_VERSION 可验版本
console.log("VOXEMW JS", VOX_JS_VERSION);

const SAMPLE_RATE = 16000;

const els = {
  status: document.getElementById("status"),
  personaBar: document.getElementById("persona-bar"),
  transcript: document.getElementById("transcript"),
  micBtn: document.getElementById("mic-btn"),
  orb: document.getElementById("orb"),
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
    }),
  });
  if (!res.ok) {
    addLine("sys", `⚠ WebRTC 建连失败（HTTP ${res.status}），刷新重试`);
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
  // iOS WKWebView 会在播放间隙把音频会话挂起，麦克风轨道被系统 mute/end，
  // 页面看起来正常但上行再没声音（聊几轮就「自动不聊」的元凶，2026-08-24 日志实锤：
  // 管线 listening re-enabled 后再没收到任何音频）。挂上监听，自动重启麦克风。
  const track = stream.getAudioTracks()[0];
  if (track) {
    track.onended = () => restartMic("ended");
    track.onmute = () => {
      // 临时静音可能只是路由切换，给 3s 自愈窗口，没恢复就重启
      setTimeout(() => {
        if (mic && track.muted) restartMic("muted");
      }, 3000);
    };
  }
}

async function restartMic(reason) {
  if (!mic || micStarting) return;  // 用户主动停的/正在启动，不动
  addLine("sys", `⚠ 麦克风被系统挂起（${reason}），自动重启`);
  stopMic();
  try {
    await startMic();
  } catch (e) {
    addLine("sys", `⚠ 麦克风重启失败: ${e.message}`);
  }
}

function stopMic() {
  if (!mic) return;
  mic.node.disconnect();
  mic.stream.getTracks().forEach((t) => t.stop());
  mic.ctx.close();
  mic = null;
}

// ---------------------------------------------------------------------------
// WS 音频下行（无 RTC 场景：SSH 隧道 / AutoDL 等 TCP-only 链路，WebRTC 的 UDP
// 媒体过不去）。response.output_audio.delta 的 base64 PCM16 16k 直接进
// WebAudio 队列无缝续播；打断（speech_started）本地清队，与服务端 flush 同步。
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 视觉（妮儿的眼睛）：摄像头帧定时上传，服务端在你说「看看」时取最新帧
// 喂给 MiniCPM-V。640px JPEG q0.7 ≈ 40KB / 1.5s，带宽可忽略。
// 权限被拒/无摄像头静默降级（视觉是增强，不能影响对话）。
// ---------------------------------------------------------------------------
const visionCam = { stream: null, video: null, canvas: null, timer: null };

async function startVisionCam() {
  if (visionCam.stream || visionCam.timer) return;
  try {
    visionCam.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 640, facingMode: "user" },
    });
  } catch (e) {
    addLine("sys", `⚠ 摄像头不可用（${e.message}），「看看」功能关闭`);
    return;
  }
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.srcObject = visionCam.stream;
  await video.play().catch(() => {});
  visionCam.video = video;
  visionCam.canvas = document.createElement("canvas");
  visionCam.timer = setInterval(() => {
    if (!visionCam.video || visionCam.video.readyState < 2) return;
    const v = visionCam.video;
    const s = Math.min(v.videoWidth, v.videoHeight);
    visionCam.canvas.width = 640;
    visionCam.canvas.height = 640;
    const c = visionCam.canvas.getContext("2d");
    c.drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, 640, 640);
    const b64 = visionCam.canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    fetch("/vision/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: b64 }),
    }).catch(() => {});
  }, 1500);
  addLine("sys", "📷 摄像头已就绪，说「妮儿看看」她就会看");
}

// 数字人视频帧渲染：4B float32 音频时间戳 + JPEG → canvas。
// 单循环队列放映（不用 setTimeout 定时器风暴）：帧按到达序进队，
// rAF 循环按播放时钟（ctx.currentTime - responseStartCtx）取「到点的
// 最新帧」上屏——落后自动跳帧追齐，不倒序、不扎堆、不受解码快慢影响。
let avatarEnabled = false;  // vox.status 下发：渲染服务在线
const avatarCanvas = { ctx2d: null, lastBitmap: null, queue: [], rafId: 0 };

function drawAvatarFrame(blob) {
  const canvas = document.getElementById("avatar-canvas");
  if (!canvas) return;
  if (!avatarCanvas.ctx2d) avatarCanvas.ctx2d = canvas.getContext("2d");
  blob.arrayBuffer().then((buf) => {
    const aTime = new DataView(buf).getFloat32(0, true);
    const jpeg = buf.slice(4);
    const item = { aTime, bmp: null };
    // 解码在后台进行；放映循环只画解码完成的
    item.bmp = createImageBitmap(new Blob([jpeg], { type: "image/jpeg" }))
      .then((b) => { item.bmp = b; })
      .catch(() => { item.bmp = "bad"; });
    avatarCanvas.queue.push(item);  // 到达序 = 时间戳序
    startAvatarLoop();
  }).catch(() => {});
}

// 开头提前放映曲线：模型的张嘴过渡天生要 ~0.5s（运动连续性，从闭嘴续接）。
// 过渡帧【按原速】铺进 lead 窗口（首帧就绪 ~0.65s → 出声 1.2s，窗口 0.55s），
// 出声瞬间过渡正好播完（嘴是开的）——过渡帧必须按原速播：
// 上一版把它们拉伸到 1s+（等效 ~8fps），用户看到的就是「前两秒卡顿」。
// 帧晚到（RTF 尖峰）时自动塌缩成「最新到点帧」，平滑降级不卡死。
const AVATAR_ONSET_S = 0.5;    // 模型张嘴过渡段长度（秒，montage 实测）
const AVATAR_ONSET_KEY0 = -0.55; // aTime=0 帧的放映点（= 首帧就绪时刻，pos 轴）
function avatarWarp(a) {
  return a < AVATAR_ONSET_S
    ? a * (0.55 / AVATAR_ONSET_S) + AVATAR_ONSET_KEY0  // 原速 23fps 铺满窗口
    : a;                                                // 过渡后帧级精确同步
}

function startAvatarLoop() {
  if (avatarCanvas.rafId) return;
  const canvas = document.getElementById("avatar-canvas");
  const tick = () => {
    avatarCanvas.rafId = 0;
    const q = avatarCanvas.queue;
    // 播放位置（秒）：本回复音频轴上声音走到哪了
    const pos = (wsPlayer.responseStartCtx > 0 && wsPlayer.ctx)
      ? wsPlayer.ctx.currentTime - wsPlayer.responseStartCtx
      : Infinity;
    // 取「到点的最新帧」：语音帧到点才播（pos >= aTime），待机帧随时可播
    // 但让位给语音帧；一路弹到「未到点语音帧」为止
    let show = null;
    while (q.length > 0) {
      const head = q[0];
      if (head.aTime >= 0 && avatarWarp(head.aTime) > pos) break;  // 还没到点的语音帧（开头按 warp 提前）
      q.shift();
      if (head.aTime >= 0) show = head;       // 到点语音帧：继续找更新的
      else if (!show) show = head;            // 待机帧：暂存，被语音帧覆盖
    }
    if (show) {
      if (show.bmp instanceof Promise || show.bmp === null) {
        q.unshift(show);  // 没解码完，放回去等下一帧（保序）
      } else if (show.bmp !== "bad") {
        const prev = avatarCanvas.lastBitmap;
        avatarCanvas.lastBitmap = show.bmp;
        avatarCanvas.ctx2d.drawImage(show.bmp, 0, 0, canvas.width, canvas.height);
        if (prev) prev.close();
      }
    }
    if (avatarCanvas.queue.length > 0) {
      avatarCanvas.rafId = requestAnimationFrame(tick);
    }
  };
  avatarCanvas.rafId = requestAnimationFrame(tick);
}


function clearAvatarPending() {
  // 打断：未上屏的帧全部作废（嘴立刻回待机）
  for (const item of avatarCanvas.queue) {
    if (item.bmp && !(item.bmp instanceof Promise) && item.bmp !== "bad") item.bmp.close();
  }
  avatarCanvas.queue = [];
}

const wsPlayer = {
  ctx: null,
  nextStart: 0,        // AudioContext 时间轴上下一块的起播时刻
  sources: new Set(),  // 已排程未播完的源（打断时批量 stop）
  leadSec: 0,          // 新回复播放压后（数字人渲染滞后对齐，vox.status 下发）
  _needLead: true,     // 每条回复的首块才加 lead（句间卡顿不重加）
  responseStartCtx: 0, // 本回复音频轴原点（ctx 时钟）：首块的排程起播时刻
  ensure() {
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
    return this.ctx;
  },
  feed(base64) {
    const ctx = this.ensure();
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    const int16 = new Int16Array(bytes.buffer);
    const f32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
    const buf = ctx.createBuffer(1, f32.length, 16000);  // source 播放时自动重采样到 ctx 率
    buf.copyToChannel(f32, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    // 新回复首块加 lead（数字人渲染固有滞后 ~1.2s 的对齐量）；
    // 队列续播/句间小卡只加 30ms jitter 余量
    const firstOfResponse = this._needLead;
    const base = firstOfResponse ? ctx.currentTime + 0.03 + this.leadSec
                                 : ctx.currentTime + 0.03;
    this._needLead = false;
    const t = Math.max(base, this.nextStart);
    if (firstOfResponse) this.responseStartCtx = t;  // 只在回复首块记音频轴原点
    src.start(t);
    this.nextStart = t + buf.duration;
    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  },
  flush() {
    for (const s of this.sources) {
      try { s.stop(); } catch (_) { /* 已停的忽略 */ }
    }
    this.sources.clear();
    this.nextStart = 0;
    this._needLead = true;  // 打断后的新回复重新计 lead
  },
  remainingMs() {
    if (!this.ctx || !this.nextStart) return 0;
    return Math.max(0, (this.nextStart - this.ctx.currentTime) * 1000);
  },
};

// ---------------------------------------------------------------------------
// 转写区
// ---------------------------------------------------------------------------

function addLine(cls, text) {
  // 首条消息出现时摘掉空态提示
  document.getElementById("empty-hint")?.remove();
  const div = document.createElement("div");
  div.className = `line ${cls}`;
  if (cls === "sys") {
    div.appendChild(document.createTextNode(text));
  } else {
    // 结构：div.line > img.avatar + div.bubble > 文本（有头像不带昵称）
    const img = document.createElement("img");
    img.className = "avatar";
    img.src = cls === "user" ? "/static/avatars/wo.jpeg" : "/static/avatars/henannier.png";
    img.alt = "";
    img.onerror = () => { img.style.display = "none"; };  // 克隆仓库无个人照片时不破洞
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.appendChild(document.createTextNode(text));
    if (cls === "user") {
      div.appendChild(bubble);
      div.appendChild(img);
    } else {
      div.appendChild(img);
      div.appendChild(bubble);
    }
  }
  els.transcript.appendChild(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  // 返回气泡（sys 行返回行本身），流式 delta 往里追加
  return div.querySelector(".bubble") || div;
}

function appendAssistantDelta(delta) {
  if (!assistantLine) {
    assistantLine = addLine("assistant", "");
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
let pbGuard = null;     // vox.playback_done 的 45s 兜底定时器

function setIndicator() {
  const el = els.status;
  let orbState = "";
  if (!wsConnected) {
    el.textContent = "已断开";
    el.className = "status warn";
  } else if (!mic) {
    el.textContent = "已连接";
    el.className = "status";
  } else if (avatarState === "listening") {
    el.textContent = "👂 倾听中…";
    el.className = "status state-listening";
    orbState = "state-listening";
  } else if (avatarState === "thinking") {
    el.textContent = "🤔 思考中…";
    el.className = "status state-thinking";
    orbState = "state-thinking";
  } else if (avatarState === "speaking") {
    el.textContent = "🔊 说话中";
    el.className = "status state-speaking";
    orbState = "state-speaking";
  } else {
    el.textContent = "🎙 聆听中";
    el.className = "status live";
    orbState = "state-live";
  }
  // 光环按钮同步状态（loading 态由按钮点击流程直接控制，不覆盖）
  if (!els.orb.classList.contains("state-loading")) {
    els.orb.className = `orb ${orbState}`.trim();
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
    // 用户开口（打断）：服务端 flush 音频队列，本地同步清 WS 播放队列+帧排程
    wsPlayer.flush();
    clearAvatarPending();
    assistantLine = null;
    setAvatarState("listening");
  },
  "input_audio_buffer.speech_stopped"() {
    // 用户说完：到助手首个音频 delta 之前是「思考」窗口
    if (avatarState === "listening") setAvatarState("thinking");
  },
  "conversation.item.input_audio_transcription.completed"(event) {
    const text = (event.transcript || "").trim();
    if (text) addLine("user", text);
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
    // RTC 模式：事件被服务端剥了音频体（音频走 RTC 音轨），事件本身标志「开口」；
    // WS 音频模式：delta 带 base64 PCM，直接进 WebAudio 队列。
    // lvl = 服务端算好的响度，驱动星空/光环能量动画
    if (!rtcEnabled && event.delta) wsPlayer.feed(event.delta);
    if (typeof event.lvl === "number") SPACE.ttsLevel = event.lvl;
    if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
  },
  "response.audio.delta"(event) {
    if (!rtcEnabled && event.delta) wsPlayer.feed(event.delta);
    if (typeof event.lvl === "number") SPACE.ttsLevel = event.lvl;
    if (avatarState === "listening" || avatarState === "thinking") setAvatarState("speaking");
  },
  "response.done"() {
    assistantLine = null;
    wsPlayer._needLead = true;  // 下条回复首块重新计 lead
    if (avatarState === "thinking") setAvatarState("idle");  // 无音频回复的兜底
    // 「说话中」的收尾：RTC 模式等服务端 vox.playback_done（生成完 ≠ 播完）；
    // WS 音频模式服务端不知道本地队列，按本地剩余时长自行收尾（45s 兜底防挂死）
    if (avatarState === "speaking") {
      clearTimeout(pbGuard);
      const wait = rtcEnabled ? 45000 : Math.min(wsPlayer.remainingMs() + 200, 45000);
      pbGuard = setTimeout(() => {
        if (avatarState === "speaking") setAvatarState("idle");
      }, wait);
    }
  },
  "vox.playback_done"() {
    // 服务端：回复音频真正播完了（仅 RTC 模式可信；WS 音频模式它在生成完即触发，
    // 本地队列还在播，忽略之，收尾交给 response.done 的本地计时）
    if (!rtcEnabled) return;
    clearTimeout(pbGuard);
    if (avatarState === "speaking") setAvatarState("idle");
  },
  error(event) {
    addLine("sys", `⚠ ${(event.error && event.error.message) || "未知错误"}`);
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
        addLine("sys", `⚠ WebRTC 建连异常: ${e.message}`)
      );
    }
    // 数字人：渲染服务在线时显示视频画布 + 设音频播放 lead（唇形对齐）
    avatarEnabled = !!(event.avatar && event.avatar.enabled);
    if (avatarEnabled) {
      wsPlayer.leadSec = ((event.avatar && event.avatar.audio_lead_ms) || 0) / 1000;
      document.getElementById("avatar-stage").hidden = false;
    }
    // 视觉：服务端开了 VLM 边车就启动摄像头帧上传
    if (event.vision && event.vision.enabled) startVisionCam();
    return;
  }
  const handler = realtimeHandlers[event.type];
  if (handler) handler(event);
}

// ---------------------------------------------------------------------------
// 星空背景：全屏 canvas，跟随对话状态（avatarState）的三种动态
//   idle      无人说话：无序漂移 + 闪烁 + 朝本位弱回弹
//   listening 你在说话：径向声波随你的音量跳动（麦克风 RMS 驱动）+ 有界内流收拢
//   thinking  良子在想：有界内流随呼吸深浅变化（锚定本位，不会无限缩向中心）
//   speaking  良子说话：径向声波随她的音量跳动（响度由服务端随音频事件下发，
//             不再用客户端 WebAudio 分析器——RTC 重连/自动播放挂起/WebKit 都会弄哑它）
// ---------------------------------------------------------------------------

const SPACE = {
  stars: [],
  w: 0,
  h: 0,
  micLevel: 0,      // 麦克风 RMS（0..1，快攻慢放）
  ttsLevel: 0,      // 良子响度（0..1，服务端随音频事件下发，帧间缓慢衰减）
  last: 0,
};

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
    SPACE.stars = Array.from({ length: n }, () => {
      const x = Math.random() * SPACE.w;
      const y = Math.random() * SPACE.h;
      return {
        x, y,
        hx: x,                // 本位：收拢以此为中心锚点，保证有界（防越聊越缩成一团）
        hy: y,
        vx: (Math.random() - 0.5) * 0.18,   // 无序漂移速度
        vy: (Math.random() - 0.5) * 0.18,
        r: 0.5 + Math.random() * 1.3,       // 基础半径
        p: Math.random() * Math.PI * 2,     // 闪烁相位
        s: 0.4 + Math.random() * 1.2,       // 闪烁速率
      };
    });
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

  // 良子响度随音频事件刷新（见 realtimeHandlers），帧间缓慢衰减；
  // 生成完毕后音频还在播（playback_done 未到）：留一个波纹地板，不装死
  if (mode === "speaking") {
    SPACE.ttsLevel = Math.max(SPACE.ttsLevel * 0.995, 0.12);
  } else {
    SPACE.ttsLevel *= 0.985;
  }
  SPACE.micLevel *= mode === "listening" ? 1 : 0.94;  // 非倾听期麦克风能量淡出

  g.clearRect(0, 0, w, h);
  const breath = 0.5 + 0.5 * Math.sin(t * 0.0011);  // thinking 的深呼吸

  for (const st of stars) {
    let boost = 0;   // 额外亮度（0..1）
    let rx = st.x;   // 渲染位置：波动只改渲染，不改基准位置（防斯托克斯漂移外扩）
    let ry = st.y;
    if (mode === "listening" || mode === "speaking") {
      // 径向声波：以中心为源的正弦波，幅度随音量——你说随麦克风、她说随 RTC 音频
      const lvl = mode === "speaking" ? SPACE.ttsLevel : SPACE.micLevel;
      const dx = st.x - cx;
      const dy = st.y - cy;
      const dist = Math.hypot(dx, dy) || 1;
      const wave = Math.sin(dist * 0.014 - t * 0.007);
      const off = wave * lvl * 14;
      rx = st.x + (dx / dist) * off;
      ry = st.y + (dy / dist) * off;
      boost = lvl * (0.35 + 0.4 * wave);
      if (mode === "listening") {
        // 倾听保留内流收拢：有界地朝「本位→中心」35% 处的锚点收，松手后 idle 弹回
        const tx = st.hx + (cx - st.hx) * 0.35;
        const ty = st.hy + (cy - st.hy) * 0.35;
        st.x += (tx - st.x) * 0.012 * dt;
        st.y += (ty - st.y) * 0.012 * dt;
      }
    } else if (mode === "thinking") {
      st.vx *= 0.99; st.vy *= 0.99;
      // 内流同样有界：收拢深度随呼吸在 18%..33% 之间
      const k = 0.18 + breath * 0.15;
      const tx = st.hx + (cx - st.hx) * k;
      const ty = st.hy + (cy - st.hy) * k;
      st.x += (tx - st.x) * 0.01 * dt;
      st.y += (ty - st.y) * 0.01 * dt;
      boost = breath * 0.2;
    } else {
      // idle：无序漂移 + 朝本位的弱回弹（收拢期攒下的位移慢慢归位）
      if (Math.random() < 0.002) {
        st.vx += (Math.random() - 0.5) * 0.06;
        st.vy += (Math.random() - 0.5) * 0.06;
      }
      st.vx = Math.max(-0.35, Math.min(0.35, st.vx));
      st.vy = Math.max(-0.35, Math.min(0.35, st.vy));
      st.x += st.vx * dt + (st.hx - st.x) * 0.002 * dt;
      st.y += st.vy * dt + (st.hy - st.y) * 0.002 * dt;
    }
    // 出界回卷（基准位置）
    if (st.x < -4) st.x = w + 4; else if (st.x > w + 4) st.x = -4;
    if (st.y < -4) st.y = h + 4; else if (st.y > h + 4) st.y = -4;

    const tw = 0.55 + 0.45 * Math.sin(t * 0.001 * st.s + st.p);
    const alpha = Math.min(1, tw * (0.45 + boost) + boost * 0.3);
    const rad = st.r * (1 + boost * 0.9);
    g.beginPath();
    g.arc(rx, ry, rad, 0, Math.PI * 2);
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
    let text = "";
    if (!pc) { dbg.textContent = "RTC 未建连"; return; }
    const stats = await pc.getStats();
    stats.forEach((r) => {
      if (r.type === "inbound-rtp" && r.kind === "audio") {
        text +=
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

// ---------------------------------------------------------------------------
// 前台自愈：iOS WKWebView 切后台/锁屏会被挂起，回前台时 ws/RTC/麦克风可能都死了。
// 回前台那一刻统一巡检，哪个死了重启哪个（麦克风走 restartMic 的 mute/end 监听，
// 这里补 ws + RTC 两条路）
// ---------------------------------------------------------------------------

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") return;
  if (!wsConnected || !ws || ws.readyState !== WebSocket.OPEN) {
    connect();  // ws 死了：重连（vox.status 到了会触发 RTC 重建）
    return;
  }
  if (mic && rtcEnabled && !pc) {
    startRTC().catch(() => {});  // ws 活着但 RTC 死了（consent expired 之类）：重建
  }
});

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
    if (wantMic) {
      // 断线时点按钮发起的重连：连上后接着自动开麦
      wantMic = false;
      micStarting = false;
      els.micBtn.classList.remove("loading");
      els.orb.classList.remove("state-loading");
      els.micBtn.onclick();
    }
  };
  ws.onclose = () => {
    wsConnected = false;
    setIndicator();
    // 按钮保持可点：断线状态下它就是重连入口（见 micBtn.onclick）
    els.micBtn.disabled = false;
    stopRTC();
    // 对话中断后 3s 自动重连；没开麦的页面则等用户点按钮（防两台设备互踢死循环）
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
    } else if (msg.data instanceof Blob) {
      drawAvatarFrame(msg.data);  // 数字人 JPEG 帧（SoulX 渲染服务）
    }
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

let micStarting = false;  // 启动流程进行中（防首次慢启动时误触连点）
let wantMic = false;      // 断线时点按钮 = 重连并自动开麦（onopen 里兑现）

els.micBtn.onclick = async () => {
  if (micStarting) return;
  if (mic) {
    stopMic();
    setAvatarState("idle");
    els.micBtn.textContent = "🎙";
    els.micBtn.classList.remove("live");
    return;
  }
  if (!wsConnected || !ws || ws.readyState !== WebSocket.OPEN) {
    // 断线状态：按钮就是重连入口（比如被另一台设备顶掉会话后）
    wantMic = true;
    micStarting = true;
    els.micBtn.disabled = true;
    els.micBtn.classList.add("loading");
    els.micBtn.textContent = "⏳";
    els.orb.className = "orb state-loading";
    connect();
    return;
  }
  // 首次启动要拉起麦克风权限/AudioContext/建连，有数百毫秒空窗——进 loading 态防误触
  micStarting = true;
  els.micBtn.disabled = true;
  els.micBtn.classList.add("loading");
  els.micBtn.textContent = "⏳";
  els.orb.className = "orb state-loading";
  try {
    await startMic();
    els.micBtn.textContent = "■";
    els.micBtn.classList.add("live");
    els.orb.classList.remove("state-loading");
    setIndicator();  // 开麦：🎙 聆听中 + 光环同步
    // 用户手势已发生：解锁/补播 RTC 音频元素（autoplay 策略要手势）；
    // WS 音频模式的播放 ctx 也趁手势预热，免得首个 delta 落在 suspended 态
    ensureRtcAudio();
    if (!rtcEnabled) wsPlayer.ensure();
  } catch (e) {
    addLine("sys", `⚠ 麦克风不可用: ${e.message}`);
    els.micBtn.textContent = "🎙";
    els.orb.classList.remove("state-loading");
  } finally {
    micStarting = false;
    els.micBtn.disabled = false;
    els.micBtn.classList.remove("loading");
  }
};

init();
