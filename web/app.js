/* VoxEMW realtime 前端
 *
 * 协议（以 vendor/speech-to-speech 的 api/openai_realtime 实现为准）：
 *   连接  ws://<host>:8765/v1/realtime
 *   上行  session.update { session: { type:"realtime", instructions,
 *                          audio.input.turn_detection, audio.output.voice } }
 *         input_audio_buffer.append { audio: base64(int16 PCM 16kHz 单声道) }
 *   下行  session.created
 *         input_audio_buffer.speech_started / speech_stopped
 *         conversation.item.input_audio_transcription.delta / .completed { transcript }
 *         response.output_audio.delta { delta: base64(int16 PCM 16kHz) } / .done
 *         response.output_audio_transcript.done { transcript }
 *         response.done / error
 *
 * WS 地址：默认 ws://<页面 host>:8765，可用 URL 参数覆盖：
 *   ?ws=192.168.1.10:8765  或  ?ws=ws://host:8765/v1/realtime
 */

"use strict";

const SAMPLE_RATE = 16000;

const els = {
  personaSelect: document.getElementById("persona-select"),
  connectBtn: document.getElementById("connect-btn"),
  status: document.getElementById("status"),
  log: document.getElementById("log"),
};

let personas = [];
let ws = null;
let mic = null; // { ctx, stream, node }
let player = null; // { ctx, nextStartTime }

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

function wsURL() {
  const override = new URLSearchParams(location.search).get("ws");
  if (override) {
    if (override.startsWith("ws://") || override.startsWith("wss://")) return override;
    return `ws://${override}/v1/realtime`;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.hostname}:8765/v1/realtime`;
}

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = `status ${cls}`;
}

function addLog(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const who = role === "user" ? "你" : role === "assistant" ? currentPersonaName() : "系统";
  div.innerHTML = `<span class="who"></span><span class="what"></span>`;
  div.querySelector(".who").textContent = who;
  div.querySelector(".what").textContent = text;
  els.log.appendChild(div);
  els.log.scrollTop = els.log.scrollHeight;
  return div;
}

function currentPersona() {
  return personas.find((p) => p.id === els.personaSelect.value) || personas[0];
}

function currentPersonaName() {
  const p = currentPersona();
  return p ? p.name : "助手";
}

function floatTo16BitPCM(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

function base64FromInt16(int16) {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function int16FromBase64(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

// ---------------------------------------------------------------------------
// 麦克风采集（AudioWorklet，16kHz PCM16）
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
      ws.send(
        JSON.stringify({ type: "input_audio_buffer.append", audio: base64FromInt16(int16) })
      );
    }
  };
  source.connect(node);
  // worklet 需要有输出链路才会跑；接 0 增益避免把麦克风回放出来
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
// 音频播放（PCM16 16kHz 排队播放，可整体清空应对打断）
// ---------------------------------------------------------------------------

function ensurePlayer() {
  if (!player) {
    player = { ctx: new AudioContext({ sampleRate: SAMPLE_RATE }), nextStartTime: 0 };
  }
  if (player.ctx.state === "suspended") player.ctx.resume();
  return player;
}

function playPCM(int16) {
  const p = ensurePlayer();
  const buf = p.ctx.createBuffer(1, int16.length, SAMPLE_RATE);
  const data = buf.getChannelData(0);
  for (let i = 0; i < int16.length; i++) data[i] = int16[i] / 0x8000;
  const src = p.ctx.createBufferSource();
  src.buffer = buf;
  src.connect(p.ctx.destination);
  const now = p.ctx.currentTime;
  const start = Math.max(now + 0.02, p.nextStartTime);
  src.start(start);
  p.nextStartTime = start + buf.duration;
}

function flushPlayback() {
  // 打断（barge-in）：关掉播放 ctx 重建，队列里的音频全部作废
  if (player) {
    player.ctx.close();
    player = null;
  }
}

// ---------------------------------------------------------------------------
// Realtime 会话
// ---------------------------------------------------------------------------

function sendSessionUpdate() {
  const p = currentPersona();
  if (!p || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(
    JSON.stringify({
      type: "session.update",
      session: {
        type: "realtime",
        instructions: p.instructions,
        audio: {
          input: {
            turn_detection: { type: "server_vad", interrupt_response: true },
          },
          output: {
            voice: p.id, // 音色名 = 人设 id，对应配置 tts.voices 的 key
          },
        },
      },
    })
  );
}

const SERVER_EVENTS = {
  "session.created"(msg) {
    setStatus("已连接，说话吧", "connected");
    sendSessionUpdate();
  },
  "input_audio_buffer.speech_started"() {
    flushPlayback(); // barge-in：用户开口即停播放
  },
  "conversation.item.input_audio_transcription.completed"(msg) {
    if (msg.transcript && msg.transcript.trim()) addLog("user", msg.transcript);
  },
  "response.output_audio.delta"(msg) {
    playPCM(int16FromBase64(msg.delta));
  },
  "response.output_audio_transcript.done"(msg) {
    if (msg.transcript && msg.transcript.trim()) addLog("assistant", msg.transcript);
  },
  error(msg) {
    addLog("system", `错误：${(msg.error && msg.error.message) || JSON.stringify(msg)}`);
  },
};

async function connect() {
  els.connectBtn.disabled = true;
  setStatus("连接中…", "connecting");
  try {
    await startMic(); // 先拿麦克风权限，失败则不开 ws
  } catch (e) {
    setStatus("麦克风不可用", "disconnected");
    addLog("system", `麦克风申请失败：${e.message}`);
    els.connectBtn.disabled = false;
    return;
  }

  ws = new WebSocket(wsURL());
  ws.onopen = () => setStatus("握手成功，等待会话…", "connecting");
  ws.onmessage = (e) => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch {
      return;
    }
    const handler = SERVER_EVENTS[msg.type];
    if (handler) handler(msg);
  };
  ws.onerror = () => setStatus("连接出错", "disconnected");
  ws.onclose = () => {
    setStatus("已断开", "disconnected");
    disconnect(true);
  };
  els.connectBtn.textContent = "断开";
  els.connectBtn.disabled = false;
}

function disconnect(silent) {
  if (ws) {
    ws.onclose = null;
    ws.close();
    ws = null;
  }
  stopMic();
  flushPlayback();
  if (!silent) setStatus("未连接", "disconnected");
  els.connectBtn.textContent = "连接";
  els.connectBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------

async function loadPersonas() {
  try {
    const resp = await fetch("personas.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    personas = await resp.json();
  } catch (e) {
    addLog("system", `personas.json 加载失败：${e.message}（先运行 python scripts/build_personas.py）`);
    return;
  }
  els.personaSelect.innerHTML = "";
  for (const p of personas) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.name;
    els.personaSelect.appendChild(opt);
  }
  els.personaSelect.disabled = false;
}

els.connectBtn.addEventListener("click", () => {
  if (ws) disconnect(false);
  else connect();
});

els.personaSelect.addEventListener("change", () => {
  // 人设热切换：重发 session.update（instructions/说话方式 + 音色一起换）
  sendSessionUpdate();
  addLog("system", `已切换到「${currentPersonaName()}」（音色已同步切换）`);
});

loadPersonas();
