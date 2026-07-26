/* VoxEMW 斗地主语音局前端
 *
 * 协议（doudizhu/server.py）：
 *   连接  ws://<host>:8766（?ws=host:8766 可覆盖）
 *   上行  {"type":"audio","pcm":b64} / {"type":"play","cards":[...]} /
 *         {"type":"pass"} / {"type":"bid","call":bool} / {"type":"new_game"}
 *   下行  hello / state{state,events} / stt / subtitle / tts_start/tts/tts_end / error
 *
 * 语音只负责聊天和口令（不要/叫地主）；出牌用鼠标点牌。
 */

"use strict";

const SAMPLE_RATE = 16000;
// 关闭麦克风：对话完全由出牌轮替驱动，不走 VAD/STT。
// 想恢复语音插话，改回 true 即可（服务端 STT 链路一直保留着）。
const MIC_ENABLED = false;

const els = {
  status: document.getElementById("status"),
  connectBtn: document.getElementById("connect-btn"),
  newGameBtn: document.getElementById("new-game-btn"),
  playBtn: document.getElementById("play-btn"),
  passBtn: document.getElementById("pass-btn"),
  bidYesBtn: document.getElementById("bid-yes-btn"),
  bidNoBtn: document.getElementById("bid-no-btn"),
  myHand: document.getElementById("my-hand"),
  myArea: document.getElementById("my-area"),
  turnBanner: document.getElementById("turn-banner"),
  finishBanner: document.getElementById("finish-banner"),
  meta: document.getElementById("meta"),
  logPanel: document.getElementById("log-panel"),
};

let ws = null;
let mic = null;
let player = null;
let names = { you: "你", liangzi: "良子", fengge: "峰哥" };
let selected = new Set();
let lastState = null;
let lastTurnKey = "";  // 轮到你的提示音只在状态切换时响一次

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

function wsURL() {
  const override = new URLSearchParams(location.search).get("ws");
  if (override) {
    if (override.startsWith("ws://") || override.startsWith("wss://")) return override;
    return `ws://${override}`;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.hostname}:8766`;
}

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = cls;
}

function showFinishBanner(text) {
  if (!els.finishBanner) return;
  els.finishBanner.textContent = text || "";
  els.finishBanner.classList.toggle("show", !!text);
}

function addLog(role, text) {
  if (!els.logPanel) return;  // 底部字幕栏已移除（排版清爽），日志调用保留为空操作
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "system" ? "系统" : names[role] || role;
  const what = document.createElement("span");
  what.className = "what";
  what.textContent = text;
  div.appendChild(who);
  div.appendChild(what);
  els.logPanel.appendChild(div);
  els.logPanel.scrollTop = els.logPanel.scrollHeight;
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
// 麦克风 / 播放（与 web/app.js 同款模式）
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
      // bot 语音播放期间不上行麦克风：不然喇叭里 bot 的声音被收回来，
      // STT 当成用户说的话 → bot 自己跟自己搭话（乱搭话的最大乱源）。
      // 播放结束后照常上行，不影响你随时插话
      if (player && player.nextStartTime > player.ctx.currentTime + 0.05) return;
      const int16 = floatTo16BitPCM(e.data);
      ws.send(JSON.stringify({ type: "audio", pcm: base64FromInt16(int16) }));
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
  if (player) {
    player.ctx.close();
    player = null;
  }
}

// 轮到你的提示音：两声短促 beep（复用播放器 AudioContext）
function beep(freq = 880, repeats = 2) {
  try {
    const p = ensurePlayer();
    const t = p.ctx.currentTime;
    for (let i = 0; i < repeats; i++) {
      const d = i * 0.2;
      const osc = p.ctx.createOscillator();
      const g = p.ctx.createGain();
      osc.frequency.value = freq;
      g.gain.setValueAtTime(0.0001, t + d);
      g.gain.exponentialRampToValueAtTime(0.12, t + d + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + d + 0.15);
      osc.connect(g);
      g.connect(p.ctx.destination);
      osc.start(t + d);
      osc.stop(t + d + 0.16);
    }
  } catch {
    /* 提示音失败不影响游戏 */
  }
}

// 出牌音效：短促噪声「啪」，像把牌甩上桌（和 beep 的电子音区分开）
function cardThwack() {
  try {
    const p = ensurePlayer();
    const t = p.ctx.currentTime;
    const dur = 0.06;
    const buf = p.ctx.createBuffer(1, Math.ceil(p.ctx.sampleRate * dur), p.ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      // 白噪声 × 快衰减包络 = 脆响
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / data.length, 3.5);
    }
    const src = p.ctx.createBufferSource();
    src.buffer = buf;
    const lp = p.ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 4500;  // 留点高频，轻脆不闷
    const g = p.ctx.createGain();
    g.gain.value = 0.2;
    src.connect(lp);
    lp.connect(g);
    g.connect(p.ctx.destination);
    src.start(t);
  } catch {
    /* 音效失败不影响游戏 */
  }
}

// ---------------------------------------------------------------------------
// 牌面渲染
// ---------------------------------------------------------------------------

const SUIT_SYMBOL = { S: "♠", H: "♥", D: "♦", C: "♣" };

function cardEl(card, small) {
  const el = document.createElement("div");
  el.className = "card" + (small ? " small" : "");
  if (card === "BJ" || card === "RJ") {
    el.classList.add("joker");
    if (card === "RJ") el.classList.add("red");
    el.textContent = card === "BJ" ? "小王" : "大王";
  } else {
    const suit = card[0];
    const rank = card.slice(1);
    if (suit === "H" || suit === "D") el.classList.add("red");
    el.innerHTML = `<span>${SUIT_SYMBOL[suit]}</span><span>${rank}</span>`;
  }
  el.dataset.card = card;
  return el;
}

function renderCards(container, cards, small) {
  container.innerHTML = "";
  for (const c of cards) container.appendChild(cardEl(c, small));
}

// ---------------------------------------------------------------------------
// 状态渲染
// ---------------------------------------------------------------------------

function roleBadge(seat, st) {
  const el = document.getElementById(`role-${seat}`);
  if (!st.landlord) {
    el.textContent = "";
    el.className = "badge";
  } else if (st.landlord === seat) {
    el.textContent = "地主";
    el.className = "badge landlord";
  } else {
    el.textContent = "农民";
    el.className = "badge farmer";
  }
}

function renderState(st, events) {
  lastState = st;
  // 手牌与计数
  renderCards(els.myHand, st.hand, false);
  for (const c of els.myHand.children) {
    if (selected.has(c.dataset.card)) c.classList.add("selected");
    c.addEventListener("click", () => {
      const card = c.dataset.card;
      if (selected.has(card)) {
        selected.delete(card);
        c.classList.remove("selected");
      } else {
        selected.add(card);
        c.classList.add("selected");
      }
    });
  }
  for (const seat of ["you", "liangzi", "fengge"]) {
    document.getElementById(`count-${seat}`).textContent = `${st.counts[seat]} 张`;
    roleBadge(seat, st);
    // bot 手牌：只展示背面（一扇小蓝牌，张数实时递减）
    const backs = document.getElementById(`backs-${seat}`);
    if (backs) {
      backs.innerHTML = "";
      for (let i = 0; i < st.counts[seat]; i++) {
        const b = document.createElement("div");
        b.className = "card-back";
        backs.appendChild(b);
      }
    }
    const area = document.getElementById(seat === "you" ? "my-area" : `area-${seat}`);
    const active =
      (st.phase === "bidding" && st.bid_turn === seat) ||
      (st.phase === "playing" && st.turn === seat);
    area.classList.toggle("active", !!active);
  }

  // 底牌：斗地主确定后摆到地主旁边（badge 右侧）
  for (const seat of ["you", "liangzi", "fengge"]) {
    const div = document.getElementById(`bottom-${seat}`);
    if (!div) continue;
    div.innerHTML = "";
    if (st.landlord === seat && st.bottom && st.bottom.length) {
      for (const c of st.bottom) div.appendChild(cardEl(c, true));
    }
  }

  // 桌面中央：单牌堆——不管谁出的牌都放正中间，名字+牌型标注是谁的；
  // 「不要」以小 chip 形式堆在牌下面，出新牌时一并清掉
  const COMBO_LABELS = {
    single: "单张", pair: "对子", triple: "三张", triple_one: "三带一",
    triple_pair: "三带二", straight: "顺子", pairs_seq: "连对",
    plane: "飞机", plane_single: "飞机带单", plane_pair: "飞机带对",
    four_two_single: "四带二", four_two_pair: "四带两对",
    bomb: "炸弹", rocket: "王炸",
  };
  const elTrickName = document.getElementById("trick-name");
  const elTrickCards = document.getElementById("trick-cards");
  const setTrickCards = (seat, cards, comboType) => {
    if (!elTrickName || !elTrickCards) return;
    elTrickName.textContent =
      names[seat] + (comboType ? ` · ${COMBO_LABELS[comboType] || comboType}` : "");
    elTrickCards.innerHTML = "";
    for (const c of cards) elTrickCards.appendChild(cardEl(c, false));
    // 飞入动画：算出「出牌者区域中心 -> 桌面中央」的位移，牌从他手边飞上桌
    const src = document.getElementById(seat === "you" ? "my-area" : `area-${seat}`);
    if (src) {
      const s = src.getBoundingClientRect();
      const t = elTrickCards.getBoundingClientRect();
      elTrickCards.style.setProperty("--fx", `${s.left + s.width / 2 - (t.left + t.width / 2)}px`);
      elTrickCards.style.setProperty("--fy", `${s.top + s.height / 2 - (t.top + t.height / 2)}px`);
      elTrickCards.classList.remove("fly");
      void elTrickCards.offsetWidth;  // 强制 reflow，让连续两次出牌都能重新触发
      elTrickCards.classList.add("fly");
    }
  };
  const setTrickText = (seat, text) => {  // 叫地主等纯文本
    if (!elTrickName || !elTrickCards) return;
    elTrickName.textContent = names[seat];
    elTrickCards.innerHTML = `<span class="ttext">${text}</span>`;
  };
  const clearTrick = () => {
    for (const el of [elTrickName, elTrickCards]) {
      if (el) el.innerHTML = "";
    }
  };
  if (st.phase === "bidding" && !st.landlord) clearTrick();  // 新一局发牌

  for (const e of events || []) {
    if (e.type === "play") {
      clearTrick();  // 桌面只留最新一手，清爽
      setTrickCards(e.seat, e.cards, e.combo_type);
      cardThwack();  // 甩牌音效，谁的牌都响
    } else if (e.type === "pass") {
      // 不清桌：上家的牌留着，直到有人出新牌；「不要」由 bot 台词说出口，低音提示辅一下
      beep(440, 1);
    } else if (e.type === "deal") {
      showFinishBanner(null);  // 新一局发牌，收掉完局大标题
    } else if (e.type === "free_turn") {
      // 不清桌：上一手的牌留在桌面给领出的人（和大家）看着，
      // 等他出新牌时 play 分支自然会清掉旧的
    } else if (e.type === "bid") {
      setTrickText(e.seat, e.call ? "叫地主！" : "不叫");
      addLog(e.seat, e.call ? "叫地主！" : "不叫");
    } else if (e.type === "landlord") {
      clearTrick();
      addLog("system", `${names[e.seat]} 当地主，底牌 ${e.bottom.join(" ")}`);
    } else if (e.type === "no_bid_redeal") {
      addLog("system", "没人叫地主，重新发牌");
    } else if (e.type === "bomb") {
      addLog("system", `💣 ${names[e.seat]} 扔了个${e.combo_type === "rocket" ? "王炸" : "炸弹"}！`);
    } else if (e.type === "last_card") {
      addLog("system", `⚠️ ${names[e.seat]} 只剩一张牌！`);
    } else if (e.type === "finish") {
      const text = `🏁 ${e.winner === "landlord" ? "地主" : "农民"}赢！` +
        (e.spring ? "春天！" : "") + `（炸弹 ×${e.bombs}）`;
      addLog("system", text);
      showFinishBanner(`本局结束 · ${e.winner === "landlord" ? "地主赢" : "农民赢"}` +
        (e.spring ? " · 春天！" : ""));
    }
  }

  // 元信息 + 按钮
  if (st.phase === "finished") {
    els.meta.textContent = `本局结束：${st.winner === "landlord" ? "地主赢" : "农民赢"}` +
      (st.spring ? "（春天）" : "");
  } else if (st.phase === "bidding") {
    els.meta.textContent = `叫地主中……轮到 ${names[st.bid_turn]}`;
  } else {
    els.meta.textContent = `第 ${st.round} 局 · 炸弹 ×${st.bombs} · 轮到 ${names[st.turn]}`;
  }

  const myBid = st.phase === "bidding" && st.bid_turn === "you";
  els.bidYesBtn.style.display = myBid ? "" : "none";
  els.bidNoBtn.style.display = myBid ? "" : "none";
  const myTurn = st.phase === "playing" && st.turn === "you";
  els.playBtn.disabled = !myTurn;
  els.passBtn.disabled = !myTurn || !st.last_play;
  els.newGameBtn.disabled = st.phase !== "finished";

  // 轮到你：大字横幅 + 提示音（每次切入你的回合响一次）
  const bannerText = myBid ? "📢 轮到你叫地主！" : myTurn ? "📢 轮到你出牌！" : "";
  els.turnBanner.textContent = bannerText;
  els.turnBanner.classList.toggle("show", !!bannerText);
  const turnKey = myBid ? "bid" : myTurn ? "play" : "";
  if (turnKey && turnKey !== lastTurnKey) beep();
  lastTurnKey = turnKey;

  // 服务端把已出的牌从手牌移除后，清掉失效的选择
  const handSet = new Set(st.hand);
  selected = new Set([...selected].filter((c) => handSet.has(c)));
}

// ---------------------------------------------------------------------------
// 消息分发
// ---------------------------------------------------------------------------

const SERVER_EVENTS = {
  hello(msg) {
    names = { ...names, ...msg.names };
    for (const seat of ["liangzi", "fengge"]) {
      const el = document.querySelector(`#area-${seat} .bot-name`);
      if (el && msg.names[seat]) el.textContent = msg.names[seat];
    }
  },
  state(msg) {
    renderState(msg.state, msg.events || []);
  },
  stt(msg) {
    if (msg.text) addLog("you", msg.text);
  },
  subtitle(msg) {
    addLog(msg.who, msg.text);
    const el = document.getElementById(`speaking-${msg.who}`);
    if (el) el.textContent = `🗣 ${msg.text}`;
  },
  tts(msg) {
    playPCM(int16FromBase64(msg.pcm));
  },
  tts_end(msg) {
    const el = document.getElementById(`speaking-${msg.voice}`);
    if (el) setTimeout(() => (el.textContent = ""), 1500);
  },
  error(msg) {
    addLog("system", `⚠ ${msg.message}`);
  },
};

// ---------------------------------------------------------------------------
// 连接
// ---------------------------------------------------------------------------

async function connect() {
  els.connectBtn.disabled = true;
  setStatus("连接中…", "connecting");
  if (MIC_ENABLED) {
    try {
      await startMic();
    } catch (e) {
      setStatus("麦克风不可用", "disconnected");
      addLog("system", `麦克风申请失败：${e.message}`);
      els.connectBtn.disabled = false;
      return;
    }
  }
  ws = new WebSocket(wsURL());
  ws.onopen = () => setStatus("已上桌，开牌！", "connected");
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
  els.connectBtn.textContent = "下桌";
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
  els.connectBtn.textContent = "上桌";
  els.connectBtn.disabled = false;
  els.newGameBtn.disabled = true;
}

els.connectBtn.addEventListener("click", () => {
  if (ws) disconnect(false);
  else connect();
});

els.newGameBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    selected.clear();
    ws.send(JSON.stringify({ type: "new_game" }));
  }
});

els.playBtn.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN || selected.size === 0) return;
  ws.send(JSON.stringify({ type: "play", cards: [...selected] }));
  selected.clear();
});

els.passBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "pass" }));
});

els.bidYesBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "bid", call: true }));
});

els.bidNoBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "bid", call: false }));
});
