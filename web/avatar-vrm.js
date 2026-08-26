/* 良子数字人：VRM 二次元形象层（three.js + @pixiv/three-vrm，本地 vendor，无构建）。
 *
 * 叠在星空之上、控制 UI 之下（透明 WebGL canvas）。驱动输入来自 assistant.js 的
 * 星空主循环 tickSpace——同一套对话状态，不用自己再起 RAF：
 *   window.VoxAvatar.frame(dt, mode, ttsLevel, micLevel, t)
 *     dt       帧步长（60fps=1）
 *     mode     idle | listening | thinking | speaking
 *     ttsLevel 服务端下发的良子响度 0..1（口型兜底 + 说话小动作的能量）
 *     micLevel 麦克风 RMS 0..1（倾听时的专注反馈）
 *     t        RAF 时间戳（ms）
 *   window.VoxAvatar.attachAudio(el)  挂 RTC 播放 <audio>（口型音素分析的音源，
 *     见下；ensureRtcAudio 每次调用都喂一次，重复喂无副作用）
 *
 * 口型方案（参考 AIRI moeru-ai/airi 的 stage-ui-three/vrm/lip-sync.ts）：
 *   首选 wLipSync（uLipSync 的 WASM 移植）对良子实际发音做音素分析，
 *   输出 A/E/I/O/U 权重 → VRM 预设口型 aa/ee/ih/oh/ou；
 *   音源：影子 <audio> 消费同一路 RTC 流，WebAudio tap 影子元素做分析
 *   （RTC 音轨直接进 WebAudio 在 Chrome 是静音，媒体元素 tap 才可靠）；
 *   wLipSync 接不上/被吞流时退回「服务端响度合成」口型，对话不断流。
 *
 * 模型加载失败只影响本层，星空/对话照常（catch 干净，不抛给主循环）。
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { createWLipSyncNode } from "/static/vendor/wlipsync/wlipsync-single.js";

const MODEL_URL = "/static/avatars/AvatarSample_A.vrm";
const LIP_PROFILE_URL = "/static/vendor/wlipsync/lip-sync-profile.json";

let renderer = null;
let scene = null;
let camera = null;
let vrm = null;
let lookTarget = null;
let modelH = 1.5;   // 模型身高（加载后实测）
let headY = 1.3;    // 头骨世界高度（同上）

// 出厂姿态：每帧从基准四元数重新叠加微动作，增量写法会缓慢漂移
const basePose = new Map();  // boneName -> Quaternion

// 眨眼 / 表情的平滑状态
let blinkPhase = -1;        // <0 = 没在眨眼；0..1 = 一次眨眼的进度
let nextBlinkAt = 0;
const exprCur = { relaxed: 0, happy: 0, angry: 0, sad: 0, surprised: 0 };

// 情绪（服务端按句判定，vox.emotion 下发）：驱动表情 + 头部微姿态，4.5s 自然回落
const EMO_TABLE = {
  neutral:   { expr: {},                          head: {} },
  happy:     { expr: { happy: 0.75 },             head: { z: 0.05 } },
  angry:     { expr: { angry: 0.7 },              head: { x: 0.08 } },
  sad:       { expr: { sad: 0.6 },                head: { x: 0.10, y: 0.05 } },
  surprised: { expr: { surprised: 0.85 },         head: { x: -0.09 } },
};
let emoName = "neutral";
let emoSeq = 0;
let emoUntil = 0;

function setEmotion(name, seq) {
  if (seq && seq < emoSeq) return;      // 并发挥发的分类任务可能乱序到达
  emoSeq = seq || emoSeq;
  emoName = EMO_TABLE[name] ? name : "neutral";
  emoUntil = performance.now() + 4500;
}

// 头部姿态的阻尼平滑（状态切换不能直接跳变，跳一下就很假）
const headCur = { x: 0, y: 0, z: 0 };

// ── wLipSync 口型（AIRI 算法移植：winner+runner 混合、攻击/释放双速平滑、静音检测）──
const LIP_KEYS = ["A", "E", "I", "O", "U"];
const RAW_KEYS = ["A", "E", "I", "O", "U", "S"];
const RAW_TO_LIP = { A: "A", E: "E", I: "I", O: "O", U: "U", S: "I" };
const BLENDSHAPE_MAP = { A: "aa", E: "ee", I: "ih", O: "oh", U: "ou" };
const LIP_ATTACK = 50;    // 闭嘴→张口的追击速度（/s）
const LIP_RELEASE = 30;   // 张口→闭嘴的回落速度（/s）
const LIP_CAP = 0.7;
// 表情权重 0..1 之外再放大 morph：实测本模型 aa=1.0 开口也很小（无 jaw 骨可补），
// three.js 的 morphTargetInfluences 允许 >1 的外推，1.7 倍视觉上「明显在说话」
const MORPH_BOOST = 1.7;
const lipSmooth = { A: 0, E: 0, I: 0, O: 0, U: 0 };
const lipFinal = { A: 0, E: 0, I: 0, O: 0, U: 0 };
const morphBinds = { A: [], E: [], I: [], O: [], U: [] };  // key -> [{mesh, index}]
// 调试：URL ?mouth=0.6 时强制 aa=0.6（绕过分析/合成，量化模型 blendshape 幅度）
const FORCE_MOUTH = (() => {
  const v = parseFloat(new URLSearchParams(location.search).get("mouth"));
  return Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : null;
})();
// 服务端音素口型（主路径）：音素权重在服务端随音频事件下发（Chrome 的 RTC
// 音频进不了 WebAudio，浏览器端 wLipSync 恒哑），feedLip 喂入，按 32ms 一帧消费
const LIP_FRAME_MS = 32;
let lipQueue = [];
let lipNextAt = 0;
let lipLastFeed = 0;
const serverLip = { A: 0, E: 0, I: 0, O: 0, U: 0 };

function feedLip(frames) {
  lipQueue.push(...frames);
  if (lipQueue.length > 60) lipQueue.splice(0, lipQueue.length - 60);  // 积压丢老帧
  lipLastFeed = performance.now();
}

let audioCtx = null;
let tapEl = null;       // 口型分析专用的影子播放元素（不碰主播放元素）
let mediaSrc = null;
let lipNode = null;
let lipAttaching = false;  // attachAudio 异步段防重入
let lipLastActive = 0;    // wLipSync 最近一次检出有效发音（performance.now() ms）
let lipEverActive = false;

const _euler = new THREE.Euler();
const _q = new THREE.Quaternion();

function bone(name) {
  return vrm && vrm.humanoid ? vrm.humanoid.getNormalizedBoneNode(name) : null;
}

// 把「基准姿态 + 微动作」写到骨骼上（euler 为弧度，叠加在出厂姿态之后）
function pose(name, rx, ry, rz) {
  const b = bone(name);
  const base = basePose.get(name);
  if (!b || !base) return;
  _euler.set(rx, ry, rz);
  _q.setFromEuler(_euler);
  b.quaternion.copy(base).multiply(_q);
}

function onResize() {
  if (!renderer || !camera) return;
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function init(canvas) {
  renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(30, 1, 0.05, 20);
  onResize();
  window.addEventListener("resize", onResize);

  // MToon 吃方向光；环境光兜底防死黑
  scene.add(new THREE.AmbientLight(0xffffff, 1.6));
  const dir = new THREE.DirectionalLight(0xffffff, 2.0);
  dir.position.set(1, 1.6, 1.2);
  scene.add(dir);

  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  loader.load(MODEL_URL, onModelLoaded, undefined, (err) => {
    console.warn("VRM 加载失败，本层静默退出（星空不受影响）:", err);
  });
}

function onModelLoaded(gltf) {
  vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  // 蒙皮网格的包围盒不跟随动画，误剔除会整只消失
  gltf.scene.traverse((o) => { o.frustumCulled = false; });
  // 口型 morph 绑定表：frame 里在 vrm.update 之后按 lipFinal × MORPH_BOOST 覆写
  const MORPH_NAMES = {
    A: ["Fcl_MTH_A", "aa", "A"], E: ["Fcl_MTH_E", "ee", "E"],
    I: ["Fcl_MTH_I", "ih", "I"], O: ["Fcl_MTH_O", "oh", "O"], U: ["Fcl_MTH_U", "ou", "U"],
  };
  gltf.scene.traverse((o) => {
    const dict = o.morphTargetDictionary;
    if (!dict) return;
    for (const key of LIP_KEYS) {
      for (const name of MORPH_NAMES[key]) {
        if (name in dict) { morphBinds[key].push({ mesh: o, index: dict[name] }); break; }
      }
    }
  });
  scene.add(gltf.scene);

  vrm.update(0);
  gltf.scene.updateMatrixWorld(true);

  // 实测身高与头高，半身取景：相机怼到胸口以上
  const box = new THREE.Box3().setFromObject(gltf.scene);
  modelH = Math.max(0.8, box.max.y - box.min.y);
  const head = bone("head");
  headY = head ? head.getWorldPosition(new THREE.Vector3()).y : modelH * 0.88;
  camera.position.set(0, headY + 0.02 * modelH, modelH * 0.62);
  camera.lookAt(0, headY - 0.01 * modelH, 0);

  // 注视目标：默认看着观众（相机），叠加缓慢游移显得不死板
  lookTarget = new THREE.Object3D();
  lookTarget.position.copy(camera.position);
  scene.add(lookTarget);
  vrm.lookAt.target = lookTarget;

  // 存出厂姿态
  for (const name of ["hips", "spine", "chest", "upperChest", "neck", "head",
                      "leftShoulder", "rightShoulder", "leftUpperArm", "rightUpperArm"]) {
    const b = bone(name);
    if (b) basePose.set(name, b.quaternion.clone());
  }
  // T-pose 太僵：大臂放下 ~66°、肩略沉，烘进基准姿态（每帧微动作在此之上叠加）
  for (const [name, rz] of [["leftUpperArm", -1.15], ["rightUpperArm", 1.15],
                            ["leftShoulder", -0.12], ["rightShoulder", 0.12]]) {
    const base = basePose.get(name);
    if (!base) continue;
    _euler.set(0, 0, rz);
    _q.setFromEuler(_euler);
    base.multiply(_q);
    const b = bone(name);   // 立即写回骨骼（只改 Map 不生效）
    if (b) b.quaternion.copy(base);
  }

  canvasReady();
}

function canvasReady() {
  const c = document.getElementById("vrm");
  if (c) c.classList.add("ready");
}

function setExpr(name, v) {
  if (vrm && vrm.expressionManager) vrm.expressionManager.setValue(name, v);
}

// 挂口型音源：影子 <audio> 消费同一条 RTC 流，WebAudio 只 tap 影子元素——
// 主播放元素全程不被 createMediaElementSource 接管（接管不可逆，失败会把良子搞哑）。
// ensureRtcAudio 每次调用都喂一次：跟随 RTC 重连换流、手势后补 play/resume。
async function attachAudio(el) {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.resume().catch(() => {});   // autoplay 策略：用户手势处的调用会把它唤醒
    if (!tapEl) {
      tapEl = document.createElement("audio");
      tapEl.style.display = "none";
      document.body.appendChild(tapEl);
    }
    if (el.srcObject && tapEl.srcObject !== el.srcObject) {
      tapEl.srcObject = el.srcObject;    // RTC 重连换了 MediaStream，跟着换
    }
    tapEl.play().catch(() => {});        // 每次调用都补（首次无手势被拒后，按钮点击这里救回来）
    if (mediaSrc || lipAttaching || !tapEl.srcObject) return;
    lipAttaching = true;
    const profile = await (await fetch(LIP_PROFILE_URL)).json();
    lipNode = await createWLipSyncNode(audioCtx, profile);
    mediaSrc = audioCtx.createMediaElementSource(tapEl);
    mediaSrc.connect(lipNode);
    // 零增益落地：不连到 destination 的分支浏览器可以不拉起处理（worklet 不跑、权重恒 0），
    // 实测 Chrome 正是如此——挂上它分析才有数据；增益为 0，影子元素不外放
    const sink = audioCtx.createGain();
    sink.gain.value = 0;
    lipNode.connect(sink);
    sink.connect(audioCtx.destination);
    console.info("wLipSync 口型分析已接入");
  } catch (e) {
    lipNode = null;
    console.warn("wLipSync 接入失败，退回响度合成口型（不影响播放）:", e);
  } finally {
    lipAttaching = false;
  }
}

function frame(dt, mode, tts, micLevel, t) {
  if (!vrm) return;
  dt = Math.min(dt, 3);          // 后台回来大跳帧别炸
  const s = t / 1000;
  const delta = dt * 0.0167;     // 秒（wLipSync 平滑公式按秒算）
  const k = (rate) => 1 - Math.exp(-rate * delta);

  // ── 口型：服务端音素 > 浏览器 wLipSync > 响度合成 ──
  const target = { A: 0, E: 0, I: 0, O: 0, U: 0 };
  if (FORCE_MOUTH !== null) target.A = FORCE_MOUTH;  // 调试覆盖
  // 消费服务端音素帧（每 32ms 一帧，与生成节奏对齐）
  if (!lipNextAt) lipNextAt = t;
  while (lipQueue.length && t >= lipNextAt) {
    Object.assign(serverLip, lipQueue.shift());
    lipNextAt += LIP_FRAME_MS;
  }
  if (lipNextAt < t) lipNextAt = t + LIP_FRAME_MS;  // 追不上就跳到当前
  const serverActive = mode === "speaking" && (performance.now() - lipLastFeed) < 800;
  if (serverActive) {
    for (const key of LIP_KEYS) target[key] = serverLip[key];
  }
  if (!serverActive && lipNode && mode === "speaking") {
    const amp = Math.min((lipNode.volume ?? 0) * 0.9, 1) ** 0.7;
    // S（无声/摩擦）并入 I；同名取最大
    const projected = { A: 0, E: 0, I: 0, O: 0, U: 0 };
    for (const raw of RAW_KEYS) {
      const lip = RAW_TO_LIP[raw];
      const val = (lipNode.weights[raw] ?? 0) * amp;
      if (val > projected[lip]) projected[lip] = val;
    }
    // 只混权重最大的两个口型（五个全混会明显偏向变形最大的 A——AIRI 的修正）
    let winner = "I", runner = "E", winnerVal = -Infinity, runnerVal = -Infinity;
    for (const key of LIP_KEYS) {
      const val = projected[key];
      if (val > winnerVal) {
        runnerVal = winnerVal; runner = winner; winnerVal = val; winner = key;
      } else if (val > runnerVal) {
        runnerVal = val; runner = key;
      }
    }
    const now = performance.now();
    let silent = amp < 0.04 || winnerVal < 0.05;
    if (!silent) { lipLastActive = now; lipEverActive = true; }
    if (now - lipLastActive > 160) silent = true;   // 停顿/句间：闭上
    if (!silent) {
      target[winner] = Math.min(LIP_CAP, winnerVal);
      target[runner] = Math.min(LIP_CAP * 0.5, runnerVal * 0.6);
    }
  }
  // 响度合成兜底：服务端音素和 wLipSync 都没数据时用服务端响度凑
  if (!serverActive && mode === "speaking" && (!lipEverActive || performance.now() - lipLastActive > 800)) {
    const m = Math.min(1, tts * 1.7);
    target.A = Math.max(target.A, m * (0.65 + 0.25 * Math.sin(s * 9.1)));
    target.I = Math.max(target.I, m * 0.4 * (0.5 + 0.5 * Math.sin(s * 7.3 + 1.7)));
    target.U = Math.max(target.U, m * 0.3 * (0.5 + 0.5 * Math.sin(s * 5.9 + 3.1)));
  }
  for (const key of LIP_KEYS) {
    const from = lipSmooth[key];
    const to = target[key];
    lipSmooth[key] = from + (to - from) * (1 - Math.exp(-(to > from ? LIP_ATTACK : LIP_RELEASE) * delta));
    const w = lipSmooth[key] <= 0.01 ? 0 : Math.min(1, lipSmooth[key] * 0.85);
    lipFinal[key] = w;
    setExpr(BLENDSHAPE_MAP[key], w);
  }

  // ── 眨眼：随机间隔，一次 240ms ──
  if (blinkPhase < 0 && t >= nextBlinkAt) blinkPhase = 0;
  if (blinkPhase >= 0) {
    blinkPhase += delta / 0.24;
    if (blinkPhase >= 1) {
      blinkPhase = -1;
      nextBlinkAt = t + 1800 + Math.random() * 4200;
      setExpr("blink", 0);
    } else {
      setExpr("blink", Math.sin(Math.min(1, blinkPhase) * Math.PI));
    }
  }

  // ── 表情：情绪判定驱动；无判定时的老兜底（放松打底，开口带一点开心）──
  const emoActive = performance.now() < emoUntil ? emoName : "neutral";
  const emoExpr = EMO_TABLE[emoActive].expr;
  const exprTarget = {
    relaxed: emoActive === "neutral" ? (mode === "thinking" ? 0.1 : 0.25) : 0.1,
    happy: emoActive === "neutral" && mode === "speaking" ? 0.18 : (emoExpr.happy || 0),
    angry: emoExpr.angry || 0,
    sad: emoExpr.sad || 0,
    surprised: emoExpr.surprised || 0,
  };
  for (const name of ["relaxed", "happy", "angry", "sad", "surprised"]) {
    exprCur[name] += (exprTarget[name] - exprCur[name]) * k(4);
    setExpr(name, exprCur[name]);
  }

  // ── 身体微动作：呼吸常驻 + 状态姿态 ──
  const breath = Math.sin(s * 1.5);
  pose("chest", breath * 0.02, 0, 0);
  if (vrm.scene) vrm.scene.position.y = breath * 0.004;

  // 头部目标姿态（模式切换经 headCur 阻尼过渡，不直接复位）
  const open = lipSmooth.A + lipSmooth.I * 0.5 + lipSmooth.O * 0.5;  // 说话能量
  let hx = 0, hy = Math.sin(s * 0.4) * 0.05, hz = 0;  // idle：缓慢小幅度转头
  if (mode === "listening") {        // 倾听：微微前倾侧头，随你音量点头
    hx = 0.06 + micLevel * 0.06;
    hy = 0;
    hz = 0.08;
  } else if (mode === "thinking") {  // 思考：微微低头侧首（大转头太出戏，2026-08-25 用户反馈）
    hx = 0.05;
    hy = Math.sin(s * 0.5) * 0.04;
    hz = 0.06;
  } else if (mode === "speaking") {  // 说话：随口型能量点头晃脑
    hx = Math.sin(s * 6) * 0.05 * open;
    hy = Math.sin(s * 2.2) * 0.04;
  }
  const ehead = EMO_TABLE[emoActive].head;   // 情绪微姿态（怒前倾/哀低头/惊后仰）
  hx += ehead.x || 0;
  hy += ehead.y || 0;
  hz += ehead.z || 0;
  const hr = k(4.5);
  headCur.x += (hx - headCur.x) * hr;
  headCur.y += (hy - headCur.y) * hr;
  headCur.z += (hz - headCur.z) * hr;
  pose("head", headCur.x, headCur.y, headCur.z);
  pose("neck", headCur.x * 0.4, headCur.y * 0.4, headCur.z * 0.4);

  // ── 注视：看着观众 + 慢游移；思考时眼神飘开 ──
  if (lookTarget && camera) {
    const wx = Math.sin(s * 0.31) * 0.05 + (mode === "thinking" ? 0.05 : 0);
    const wy = Math.sin(s * 0.23) * 0.04 + (mode === "thinking" ? -0.07 : 0);  // 思考时眼神略垂
    lookTarget.position.set(
      camera.position.x + wx,
      headY + wy,
      camera.position.z
    );
  }

  vrm.update(delta);
  // morph 放大：覆写 expressionManager 刚算好的口型影响值（>1 外推，幅度才看得见）
  for (const key of LIP_KEYS) {
    const v = Math.min(lipFinal[key] * MORPH_BOOST, 1.8);
    for (const b of morphBinds[key]) b.mesh.morphTargetInfluences[b.index] = v;
  }
  renderer.render(scene, camera);
}

// 自举：canvas 不在或 WebGL 不可用时静默退出，星空照常
try {
  const canvas = document.getElementById("vrm");
  if (canvas) init(canvas);
} catch (e) {
  console.warn("VRM 初始化失败，本层静默退出（星空不受影响）:", e);
}

function lipInfo() {
  const now = performance.now();
  const src = now - lipLastFeed <= 800 ? "server"
    : lipNode && lipEverActive && now - lipLastActive <= 800 ? "wlipsync" : "synth";
  return {
    src,
    nodeReady: !!lipNode,
    vol: src === "server"
      ? +Math.max(...Object.values(serverLip)).toFixed(2)
      : lipNode ? +(lipNode.volume ?? 0).toFixed(2) : 0,
    w: { ...lipSmooth },
  };
}

window.VoxAvatar = { frame, attachAudio, lipInfo, setEmotion, feedLip };
