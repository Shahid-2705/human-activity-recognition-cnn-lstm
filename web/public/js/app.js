import { FilesetResolver, PoseLandmarker, FaceLandmarker, GestureRecognizer } from '/vendor/tasks-vision/vision_bundle.mjs';
import { ActivityClassifier, Voter, handGesture, finalGesture, HandActions, postureVector } from './pose-classifier.js';
import { KnnModel } from './knn.js';

const $ = (id) => document.getElementById(id);
const el = {
  status: $('status'), view: $('view'), cam: $('cam'), overlay: $('overlay'),
  activity: $('activity'), hint: $('hint'), gesture: $('gesture'), expression: $('expression'),
  source: $('source'), start: $('startBtn'), stop: $('stopBtn'), segLen: $('segLen'), timer: $('timer'),
  file: $('file'), bars: $('bars'), last: $('last'),
  trainLabel: $('trainLabel'), rec: $('recBtn'), counts: $('trainCounts'), useKnn: $('useKnn'),
  clear: $('clearBtn'), trainMsg: $('trainMsg'), presets: $('presets'), secs: $('trainSecs'),
  wizardBtn: $('wizardBtn'), wizardMsg: $('wizardMsg'), report: $('report'), cvBtn: $('cvBtn'),
  sample: $('sample'), predict: $('predictBtn'), acc: $('acc'), sensorOut: $('sensorOut'),
  srcWebcam: $('srcWebcam'), srcPhone: $('srcPhone'), phoneUrl: $('phoneUrl'), phoneHelp: $('phoneHelp'),
  phoneImg: $('phoneImg'), frameCanvas: $('frameCanvas'),
  calibBtn: $('calibBtn'), calibClear: $('calibClear'), calibState: $('calibState'), calibMsg: $('calibMsg'),
  stRun: $('stRun'), stFps: $('stFps'), stSrc: $('stSrc'), stModels: $('stModels'), stStreak: $('stStreak'),
  stChanges: $('stChanges'), history: $('history'), timeline: $('timeline'), legend: $('legend'),
};
const ctx = el.view.getContext('2d');

/* ---------- tabs ---------- */
document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tabpane').forEach((p) => { p.hidden = p.id !== `tab-${b.dataset.tab}`; });
  if (b.dataset.tab === 'sensor') initSensor();
}));

/* ---------- models ---------- */
let pose = null, face = null, hands = null;
const handVoter = new Voter(500);
let handLm = [];
let handNow = 'None';
const handActions = new HandActions();
const actVoter = new Voter(600);
let actNow = 'None';
const posture = new KnnModel();          // sit / stand calibration (body-position features only)
const postureVoter = new Voter(1000);
const classifier = new ActivityClassifier();
const knn = new KnnModel();
const knnVoter = new Voter(900);
const UNSURE_BELOW = 0.5;
const fps = { frames: 0 };
window.har = { classifier, knn, fps };  // handy for debugging in the console

function setStatus(text, cls = '') { el.status.textContent = text; el.status.className = `pill ${cls}`; }

async function loadModels() {
  try {
    const fileset = await FilesetResolver.forVisionTasks('/vendor/tasks-vision/wasm');
    const make = async (Cls, path, extra) => {
      for (const delegate of ['GPU', 'CPU']) {
        try {
          return await Cls.createFromOptions(fileset, { baseOptions: { modelAssetPath: path, delegate }, runningMode: 'VIDEO', ...extra });
        } catch (e) { if (delegate === 'CPU') throw e; }
      }
    };
    pose = await make(PoseLandmarker, '/models/pose_landmarker_full.task', { numPoses: 1 });
    face = await make(FaceLandmarker, '/models/face_landmarker.task', { numFaces: 1, outputFaceBlendshapes: true });
    try {
      hands = await make(GestureRecognizer, '/models/gesture_recognizer.task', { numHands: 2 });
    } catch (e) {
      console.warn('Hand-gesture model unavailable; using body-pose gestures only.', e);
    }
    classifier.wristFaceGestures = !hands;
    setStatus('Models ready', 'ok');
    el.start.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus('Model load failed', 'err');
    el.overlay.innerHTML = '<p>Could not load the pose/face models.<br>Run <code>npm install</code> in <code>web/</code> (it downloads them) and reload.</p>';
  }
}
el.start.disabled = true;

/* ---------- camera ---------- */
let stream = null, fileURL = null, running = false, rafId = 0, mirror = true;
let usingPhone = false, srcType = 'webcam';
const fc = el.frameCanvas, fctx = fc.getContext('2d', { willReadFrequently: true });
const probe = document.createElement('canvas'); probe.width = 16; probe.height = 9;
const pctx = probe.getContext('2d', { willReadFrequently: true });
let lastProbe = -1;

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

function setSrcType(type) {
  srcType = type === 'phone' ? 'phone' : 'webcam';
  el.srcWebcam.classList.toggle('on', srcType === 'webcam');
  el.srcPhone.classList.toggle('on', srcType === 'phone');
  el.source.hidden = srcType === 'phone';
  el.phoneUrl.hidden = srcType !== 'phone';
  el.phoneHelp.hidden = srcType !== 'phone';
  el.file.closest('label').hidden = srcType === 'phone';
  store.set('har_src', srcType);
}
el.srcWebcam.addEventListener('click', () => { if (!running) setSrcType('webcam'); });
el.srcPhone.addEventListener('click', () => { if (!running) setSrcType('phone'); });
el.phoneUrl.value = store.get('har_phone_url') || '';
el.phoneUrl.addEventListener('change', () => store.set('har_phone_url', el.phoneUrl.value.trim()));
setSrcType(store.get('har_src'));

/** Wait for the first frame of the relayed phone stream; turn failures into a readable message. */
async function openPhoneStream(raw) {
  const src = '/phone-stream?url=' + encodeURIComponent(raw);
  const im = el.phoneImg;
  try {
    await new Promise((resolve, reject) => {
      const to = setTimeout(() => reject(new Error('timeout')), 10000);
      im.onload = () => { clearTimeout(to); resolve(); };
      im.onerror = () => { clearTimeout(to); reject(new Error('error')); };
      im.src = src;
    });
  } catch {
    im.removeAttribute('src');
    let why = 'No video arrived from the phone. Check the address and that its camera app is running.';
    try { // the relay explains what went wrong in JSON
      const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), 9000);
      const r = await fetch(src, { signal: ctl.signal }); clearTimeout(to);
      if (!r.ok) why = (await r.json()).error || why; else ctl.abort();
    } catch { /* keep the generic message */ }
    const err = new Error(why); err.userMessage = why; throw err;
  }
}

async function listCameras() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    const prev = el.source.value;
    el.source.innerHTML = '<option value="">Default camera</option>' +
      devs.map((d, i) => `<option value="${d.deviceId}">${d.label || `Camera ${i + 1}`}</option>`).join('') +
      (fileURL ? '<option value="__file">Video file</option>' : '');
    if ([...el.source.options].some((o) => o.value === prev)) el.source.value = prev;
  } catch { /* no camera API */ }
}

async function startCamera() {
  stopCamera();
  el.overlay.classList.remove('hide');
  el.overlay.innerHTML = '<p>Opening camera…</p>';
  usingPhone = false;
  try {
    if (srcType === 'phone') {
      const raw = el.phoneUrl.value.trim();
      if (!raw) throw Object.assign(new Error('x'), { userMessage: 'Enter the phone camera address first, e.g. http://192.168.1.100:4747/video' });
      store.set('har_phone_url', raw);
      await openPhoneStream(raw);
      usingPhone = true; mirror = false; lastProbe = -1;
    } else if (el.source.value === '__file' && fileURL) {
      el.cam.srcObject = null; el.cam.src = fileURL; el.cam.loop = true; mirror = false;
    } else {
      const id = el.source.value;
      stream = await navigator.mediaDevices.getUserMedia({
        video: { ...(id ? { deviceId: { exact: id } } : {}), width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      });
      el.cam.removeAttribute('src'); el.cam.srcObject = stream; mirror = true;
    }
    if (!usingPhone) await el.cam.play();
  } catch (e) {
    const msg = e.userMessage ? e.userMessage : e.name === 'NotAllowedError' ? 'Camera permission was denied. Allow it in the address bar and press Start again.'
      : e.name === 'NotFoundError' ? 'No camera found.' : `Could not open the camera: ${e.message}`;
    el.overlay.innerHTML = '<p>' + esc(msg) + '</p>'; setStatus('Camera error', 'err');
    return;
  }
  await listCameras();
  el.overlay.classList.add('hide');
  if (usingPhone) { el.view.width = el.phoneImg.naturalWidth || 960; el.view.height = el.phoneImg.naturalHeight || 540; }
  else { el.view.width = el.cam.videoWidth || 960; el.view.height = el.cam.videoHeight || 540; }
  classifier.reset(); knnVoter.reset(); handActions.reset(); actVoter.reset(); handVoter.reset(); postureVoter.reset(); resetSegment(true);
  running = true; lastT = 0; frameN = 0; startDash();
  el.start.disabled = true; el.stop.disabled = false;
  setStatus('Live', 'ok');
  rafId = requestAnimationFrame(tick);
}

function stopCamera() {
  running = false; cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((t) => t.stop()); stream = null;
  el.cam.pause();
  el.phoneImg.removeAttribute('src'); usingPhone = false; // closes the relayed stream
  el.stop.disabled = true; el.start.disabled = !pose;
  if (pose) setStatus('Models ready', 'ok');
  el.overlay.classList.remove('hide');
  el.overlay.innerHTML = '<p>Press <b>Start</b> to open the camera</p>';
  cancelTraining = true;
}

el.start.addEventListener('click', startCamera);
el.stop.addEventListener('click', stopCamera);
el.file.addEventListener('change', () => {
  const f = el.file.files[0]; if (!f) return;
  if (fileURL) URL.revokeObjectURL(fileURL);
  fileURL = URL.createObjectURL(f);
  listCameras().then(() => { el.source.value = '__file'; startCamera(); });
});

/* ---------- main loop ---------- */
let lastT = 0, frameN = 0, lastUi = 0, lastVideoTime = -1;
const seg = { stats: {}, elapsed: 0, index: 1, prevT: 0 };

function resetSegment(full) {
  seg.stats = {}; seg.elapsed = 0; seg.prevT = 0;
  if (full) { seg.index = 1; el.last.textContent = ''; }
  renderBars();
}
const segMs = () => Number(el.segLen.value) * 1000;
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

function tick() {
  if (!running) return;
  rafId = requestAnimationFrame(tick);
  let v = el.cam, aw, ah;
  if (usingPhone) {
    const im = el.phoneImg;
    if (!im.naturalWidth) return;
    if (fc.width !== im.naturalWidth || fc.height !== im.naturalHeight) {
      fc.width = im.naturalWidth; fc.height = im.naturalHeight;
      el.view.width = fc.width; el.view.height = fc.height;
    }
    // A relayed MJPEG image has no frame clock: only process frames whose pixels changed,
    // otherwise repeated frames would read as "no motion" and distort walking detection.
    pctx.drawImage(im, 0, 0, 16, 9);
    const px = pctx.getImageData(0, 0, 16, 9).data; let sum = 0;
    for (let i = 0; i < px.length; i++) sum = (sum * 31 + px[i]) % 1000000007;
    if (sum === lastProbe) return;
    lastProbe = sum;
    fctx.drawImage(im, 0, 0);
    v = fc; aw = fc.width; ah = fc.height;
  } else {
    if (v.readyState < 2 || v.currentTime === lastVideoTime) return;
    lastVideoTime = v.currentTime; aw = v.videoWidth; ah = v.videoHeight;
  }

  const t = Math.max(performance.now(), lastT + 1); lastT = t;
  let lm = null, blend = null;
  try {
    const pr = pose.detectForVideo(v, t);
    lm = pr.landmarks?.[0] || null;
    window.har.world = pr.worldLandmarks?.[0] || null;   // 3D body estimate (debug / analysis)
    if (frameN++ % 3 === 0) {
      const fr = face.detectForVideo(v, t);
      blend = fr.faceBlendshapes?.[0]?.categories || null;
    }
    if (hands && frameN % 2 === 0) {
      const hr = hands.recognizeForVideo(v, t);
      handLm = hr.landmarks || [];
      window.har.hands = { lm: hr.landmarks || [], g: (hr.gestures || []).map((c) => c?.[0] ? [c[0].categoryName, +c[0].score.toFixed(2)] : null) };
      handNow = handVoter.push(t, handGesture(hr)).label;
      actNow = actVoter.push(t, handActions.update({ hands: handLm.map((h) => h.map((p) => ({ x: p.x, y: p.y }))), landmarks: lm, aspect: aw / ah, t })).label;
    }
  } catch (e) { console.warn(e); }

  const r = classifier.update({ landmarks: lm, blendshapes: blend, t, aspect: aw / ah });
  fps.frames++;
  const pv = lm ? postureVector(lm, aw / ah) : null;
  if (recorder && t - recorder.last >= 100) {
    if (recorder.kind === 'posture') {
      if (pv) { posture.add(recorder.label, pv, recorder.rec); recorder.n++; recorder.last = t; }
    } else if (r.featureVector && classifier.hasBody) {
      knn.add(recorder.label, r.featureVector.slice(), recorder.rec); recorder.n++; recorder.last = t;
    }
  }

  let label = r.activity, tag = '', maybe = '';
  if (el.useKnn.checked && knn.ready() && r.featureVector) {
    const p = knn.predict(r.featureVector);
    // Not confident -> say so instead of guessing
    const unsure = p.confidence < UNSURE_BELOW;
    label = knnVoter.push(t, unsure ? 'Unsure' : p.label).label; tag = ' · trained';
    maybe = unsure ? `Maybe ${p.label}${p.runnerUp ? ' or ' + p.runnerUp : ''} (${Math.round(p.confidence * 100)}%)` : '';
  }
  // Sitting vs standing from the user's own calibration (leg geometry is unreliable on a webcam)
  let calibrated = false;
  if (!el.useKnn.checked && posture.ready() && pv && ['Standing', 'Sitting', 'Upper body only'].includes(label)) {
    const pp = posture.predict(pv);
    const pl = postureVoter.push(t, pp.confidence >= 0.6 ? pp.label : 'Unsure').label;
    if (pl === 'Sitting' || pl === 'Standing') { label = pl; calibrated = true; }
  }
  if (!r.bodyVisible) label = 'No person';

  draw(v, lm);
  accumulate(label, t);

  if (t - lastUi > 120) {
    lastUi = t;
    el.activity.textContent = label;
    el.hint.textContent = el.useKnn.checked && tag ? (maybe || 'Using your trained model') : (calibrated ? 'Sitting / standing from your calibration' : (posture.ready() ? '' : r.hint));
    const handSeen = handLm.length > 0;
    const g = finalGesture(r.bodyVisible ? r.gesture : 'None', handNow, hands ? actNow : 'None');
    el.gesture.textContent = g === 'None' && !r.bodyVisible && !handSeen ? '—' : g;
    el.expression.textContent = r.expression;
    el.timer.textContent = `${fmt(seg.elapsed)} / ${fmt(segMs() / 1000)} · #${seg.index}`;
    renderBars();
    updateDash(t, label);
  }
}

function draw(v, lm) {
  const w = el.view.width, h = el.view.height;
  ctx.save();
  if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
  ctx.drawImage(v, 0, 0, w, h);
  ctx.restore();
  const X = (x) => (mirror ? 1 - x : x) * w, Y = (y) => y * h;
  ctx.fillStyle = '#facc15';
  for (const hand of handLm) for (const p of hand) { ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), Math.max(2, w / 450), 0, 7); ctx.fill(); }
  if (!lm) return;
  ctx.lineWidth = Math.max(2, w / 400);
  ctx.strokeStyle = 'rgba(56,189,248,.9)';
  for (const c of PoseLandmarker.POSE_CONNECTIONS) {
    const a = lm[c.start], b = lm[c.end];
    if ((a.visibility ?? 1) < 0.5 || (b.visibility ?? 1) < 0.5) continue;
    ctx.beginPath(); ctx.moveTo(X(a.x), Y(a.y)); ctx.lineTo(X(b.x), Y(b.y)); ctx.stroke();
  }
  ctx.fillStyle = '#fff';
  for (const p of lm) {
    if ((p.visibility ?? 1) < 0.5) continue;
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), Math.max(2.5, w / 330), 0, 7); ctx.fill();
  }
}

/* ---------- continuous 1-minute segments ---------- */
function accumulate(label, t) {
  if (seg.prevT) {
    const dt = Math.min((t - seg.prevT) / 1000, 0.25);
    seg.elapsed += dt;
    if (label !== 'No person') seg.stats[label] = (seg.stats[label] || 0) + dt;
  }
  seg.prevT = t;
  if (seg.elapsed >= segMs() / 1000) {
    const top = Object.entries(seg.stats).sort((a, b) => b[1] - a[1]).slice(0, 3)
      .map(([k, s]) => `${k} ${Math.round(s)}s`).join(', ');
    el.last.textContent = `Segment #${seg.index}: ${top || 'no person detected'}`;
    addHistory(`#${seg.index} · ${top || 'no person detected'}`);
    seg.index++; seg.stats = {}; seg.elapsed = 0;
  }
}

function renderBars() {
  const entries = Object.entries(seg.stats).sort((a, b) => b[1] - a[1]).slice(0, 6);
  if (!entries.length) { el.bars.innerHTML = '<div class="empty">Waiting for data</div>'; return; }
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
  el.bars.innerHTML = entries.map(([k, s]) =>
    `<div class="bar"><span>${esc(k)}</span><div class="track"><div class="fill" style="width:${(s / total) * 100}%"></div></div><span class="val">${Math.round(s)}s</span></div>`).join('');
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* ---------- train on my own activities ---------- */
const KEY = 'har_knn_v2';
function loadKnn() { try { const s = localStorage.getItem(KEY); if (s) knn.fromJSON(s); } catch { /* storage unavailable */ } renderCounts(); }
function saveKnn() { try { localStorage.setItem(KEY, knn.toJSON()); } catch { /* quota / private mode */ } }

function renderCounts() {
  const c = knn.counts();
  el.counts.innerHTML = Object.entries(c).map(([k, n]) =>
    `<span>${esc(k)} · ${n}<button data-l="${esc(k)}" title="Delete">×</button></span>`).join('') || '<span>No samples yet</span>';
  el.counts.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { knn.clearLabel(b.dataset.l); saveKnn(); renderCounts(); }));
  el.useKnn.disabled = !knn.ready();
  if (!knn.ready()) el.useKnn.checked = false;
}

const PRESETS = {
  Standing: 'Stand facing the camera with your hips and knees in view.',
  Sitting: 'Sit the way you normally do. Step back so your hips and knees show.',
  Laying: 'Lie down so the camera sees your whole body.',
  Walking: 'Walk (or march in place) so your legs move.',
  Running: 'Jog in place with your arms moving.',
  Jumping: 'Jump up and down.',
  Waving: 'Wave one hand above your shoulder.',
  Clapping: 'Clap your hands steadily.',
  'Phone call': 'Hold a phone (or your hand) to your ear.',
  Eating: 'Bring your hand to your mouth as if eating or drinking.',
  Crying: 'Act it out: frown, brows up, cover your face if you like.',
  Yawning: 'Yawn with your hand over your mouth.',
};
const DEFAULT_ON = ['Standing', 'Sitting', 'Laying', 'Walking', 'Waving', 'Crying'];
el.presets.innerHTML = Object.keys(PRESETS).map((k) =>
  `<label><input type="checkbox" value="${esc(k)}" ${DEFAULT_ON.includes(k) ? 'checked' : ''}> ${esc(k)}</label>`).join('');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let cancelTraining = false, training = false;

/**
 * Record `secs` seconds of feature vectors for one label. Samples are taken from the
 * video loop (about every 100 ms of video time), so a throttled timer cannot starve it.
 */
let recorder = null;
function recordClip(label, secs, onTick, kind = 'activity') {
  return new Promise((resolve) => {
    const model = kind === 'posture' ? posture : knn;
    recorder = { label, secs, onTick, resolve, kind, rec: model.newRecording(), n: 0, last: 0, start: performance.now() };
  });
}
function pollRecorder() {
  if (!recorder) return;
  const elapsed = (performance.now() - recorder.start) / 1000;
  recorder.onTick?.(Math.min(elapsed / recorder.secs, 1), Math.max(Math.ceil(recorder.secs - elapsed), 0));
  if (elapsed >= recorder.secs || cancelTraining || !running) {
    const done = recorder; recorder = null; done.resolve(done.n);
  }
}
setInterval(pollRecorder, 100);

function setWizard(html, pct = 0) {
  el.wizardMsg.hidden = false;
  el.wizardMsg.innerHTML = `${html}<div class="prog"><i style="width:${pct * 100}%"></i></div>`;
}

async function runGuided() {
  if (training) { cancelTraining = true; return; }
  const list = [...el.presets.querySelectorAll('input:checked')].map((i) => i.value);
  if (!running) { el.trainMsg.textContent = 'Start the camera first.'; return; }
  if (list.length < 2) { el.trainMsg.textContent = 'Tick at least two activities.'; return; }
  const secs = Number(el.secs.value);
  training = true; cancelTraining = false; el.wizardBtn.textContent = 'Cancel'; el.trainMsg.textContent = '';
  const useBefore = el.useKnn.checked; el.useKnn.checked = false;
  try {
    for (let i = 0; i < list.length && !cancelTraining; i++) {
      const label = list[i];
      const head = `Step ${i + 1}/${list.length} · <b>${esc(label)}</b>${esc(PRESETS[label] || '')}<br>`;
      knn.clearLabel(label); // re-training an activity replaces its old samples
      for (let c = 4; c > 0 && !cancelTraining; c--) { setWizard(`${head}Get ready… ${c}`, 0); await sleep(1000); }
      if (cancelTraining) break;
      const n = await recordClip(label, secs, (p, left) => setWizard(`${head}Recording… ${left}s`, p));
      if (n < 10 && !cancelTraining) { el.trainMsg.textContent = `"${label}": too few frames - make sure your body is visible.`; knn.clearLabel(label); }
    }
  } finally {
    training = false; el.wizardBtn.textContent = 'Start guided training';
    saveKnn(); renderCounts();
  }
  el.wizardMsg.hidden = true;
  if (cancelTraining) { el.trainMsg.textContent = 'Training cancelled (samples recorded so far were kept).'; el.useKnn.checked = useBefore && knn.ready(); return; }
  if (knn.ready()) { el.useKnn.checked = true; showReport(); }
}

function showReport() {
  const cv = knn.crossValidate();
  if (!cv) { el.report.textContent = 'Record at least two activities to measure accuracy.'; return; }
  const pct = Math.round(cv.accuracy * 100), cls = pct >= 85 ? 'good' : pct >= 65 ? 'mid' : 'bad';
  const rows = Object.entries(cv.perLabel).sort((a, b) => a[1].recall - b[1].recall);
  const weak = rows.filter(([, v]) => v.recall < 0.75);
  el.report.innerHTML = `<div class="score ${cls}">${pct}% <small>estimated accuracy</small></div>
    <div class="note">Tested on parts of your recordings the model did not learn from.</div>
    <div class="rows">${rows.map(([k, v]) => `<span class="${v.recall < 0.75 ? 'weak' : ''}">${esc(k)} ${Math.round(v.recall * 100)}%</span>`).join('')}</div>
    ${weak.length ? `<div class="tip">Weak: ${weak.map(([k, v]) => esc(k) + (v.confusedWith ? ` (mixed up with ${esc(v.confusedWith)})` : '')).join('; ')}. Re-record those, making them look more different.</div>` : ''}`;
}

el.wizardBtn.addEventListener('click', runGuided);
el.cvBtn.addEventListener('click', showReport);

el.rec.addEventListener('click', async () => {
  const label = el.trainLabel.value.trim();
  if (!running) { el.trainMsg.textContent = 'Start the camera first.'; return; }
  if (!label) { el.trainMsg.textContent = 'Type an activity name first.'; return; }
  if (training) return;
  training = true; cancelTraining = false; el.rec.disabled = true;
  const n = await recordClip(label, 10, (p, left) => { el.rec.textContent = `Recording ${left}s`; });
  training = false; el.rec.disabled = false; el.rec.textContent = 'Record 10 s';
  saveKnn(); renderCounts();
  el.trainMsg.textContent = n < 10 ? 'Too few frames captured - make sure your body is visible.'
    : knn.ready() ? `Saved ${n} samples for "${label}".` : `Saved ${n} samples. Add at least one more activity to enable the model.`;
});
el.clear.addEventListener('click', () => { knn.clear(); saveKnn(); renderCounts(); el.report.innerHTML = ''; el.trainMsg.textContent = 'Cleared.'; });

/* ---------- dashboard: session stats, timeline, history ---------- */
const COLORS = {
  Standing: '#a78bfa', Sitting: '#f472b6', Laying: '#34d399', Walking: '#38bdf8', Running: '#fb923c',
  Jumping: '#facc15', Bending: '#2dd4bf', 'Upper body only': '#64748b', 'No person': '#334155', Unsure: '#94a3b8',
};
function colorFor(label) {
  if (COLORS[label]) return COLORS[label];
  let h = 0; for (const c of label) h = (h * 31 + c.charCodeAt(0)) % 360;
  return `hsl(${h} 65% 60%)`;
}
const dash = { start: 0, samples: [], lastSample: 0, lastFrames: 0, lastFpsT: 0, fps: 0, changes: 0, prevLabel: null, streakStart: 0, longest: { label: '', s: 0 }, history: [] };
const TIMELINE_S = 300;

function startDash() {
  Object.assign(dash, { start: performance.now(), samples: [], lastSample: 0, lastFrames: fps.frames, lastFpsT: performance.now(), fps: 0, changes: 0, prevLabel: null, streakStart: performance.now(), longest: { label: '', s: 0 } });
  el.stSrc.textContent = usingPhone ? 'Phone (Wi-Fi)' : el.source.value === '__file' ? 'Video file' : (el.source.selectedOptions[0]?.textContent || 'Webcam').replace(/\s*\(.*\)\s*$/, '');
  el.stModels.textContent = `Pose · Face${hands ? ' · Hands' : ''}`;
  el.stChanges.textContent = '0'; el.stStreak.textContent = '—';
}

function updateDash(t, label) {
  const now = performance.now();
  el.stRun.textContent = fmt((now - dash.start) / 1000);
  if (now - dash.lastFpsT >= 1000) {
    dash.fps = ((fps.frames - dash.lastFrames) * 1000) / (now - dash.lastFpsT);
    dash.lastFrames = fps.frames; dash.lastFpsT = now;
    el.stFps.textContent = `${dash.fps.toFixed(0)} fps`;
  }
  // streaks and changes ignore the "no one in view" state
  if (label !== 'No person') {
    if (label !== dash.prevLabel) {
      if (dash.prevLabel) dash.changes++;
      dash.prevLabel = label; dash.streakStart = now;
    }
    const s = (now - dash.streakStart) / 1000;
    if (s > dash.longest.s) dash.longest = { label, s };
    el.stChanges.textContent = String(dash.changes);
    el.stStreak.textContent = dash.longest.s >= 1 ? `${dash.longest.label} ${fmt(dash.longest.s)}` : '—';
  }
  if (now - dash.lastSample >= 1000) {
    dash.lastSample = now;
    dash.samples.push(label);
    if (dash.samples.length > TIMELINE_S) dash.samples.shift();
    drawTimeline();
  }
}

function drawTimeline() {
  const c = el.timeline, g = c.getContext('2d'), n = dash.samples.length;
  g.clearRect(0, 0, c.width, c.height);
  const w = c.width / TIMELINE_S;
  dash.samples.forEach((lab, i) => { g.fillStyle = colorFor(lab); g.fillRect((TIMELINE_S - n + i) * w, 6, Math.ceil(w), c.height - 12); });
  const seen = [...new Set(dash.samples)];
  el.legend.innerHTML = seen.length
    ? seen.map((l) => `<span><i style="background:${colorFor(l)}"></i>${esc(l)}</span>`).join('')
    : '<span class="empty">Starts when the camera runs</span>';
}

function addHistory(text) {
  dash.history.unshift(text);
  dash.history = dash.history.slice(0, 5);
  el.history.innerHTML = dash.history.map((h) => `<li>${esc(h)}</li>`).join('');
}

/* ---------- sit / stand calibration ---------- */
const POSTURE_KEY = 'har_posture_v1';
function loadPosture() { const t = store.get(POSTURE_KEY); if (t) posture.fromJSON(t); renderCalib(); }
function renderCalib() {
  const c = posture.counts();
  const ok = posture.ready();
  el.calibState.textContent = ok ? `Calibrated · Sitting ${c.Sitting || 0} · Standing ${c.Standing || 0} samples` : 'Not calibrated';
  el.calibState.className = ok ? 'ok' : '';
  el.calibClear.hidden = !ok;
  el.calibBtn.textContent = ok ? 'Re-calibrate (25 s)' : 'Calibrate (25 s)';
}
function setCalibMsg(html, pct = 0) {
  el.calibMsg.hidden = false;
  el.calibMsg.innerHTML = `${html}<div class="prog"><i style="width:${pct * 100}%"></i></div>`;
}
async function calibratePosture() {
  if (training) { cancelTraining = true; return; }
  if (!running) { setCalibMsg('Start the camera first, then press Calibrate.'); return; }
  training = true; cancelTraining = false; el.calibBtn.textContent = 'Cancel';
  const steps = [['Sitting', 'Sit the way you normally do, then stay still.'], ['Standing', 'Stand where you normally stand, then stay still.']];
  const keep = { Sitting: 0, Standing: 0 };
  posture.clear(); let failed = '';
  try {
    for (let i = 0; i < steps.length && !cancelTraining; i++) {
      const [label, text] = steps[i];
      const head = `Step ${i + 1}/2 · <b>${esc(label)}</b>${esc(text)}<br>`;
      for (let c = 5; c > 0 && !cancelTraining; c--) { setCalibMsg(`${head}Get ready… ${c}`, 0); await sleep(1000); }
      if (cancelTraining) break;
      const n = await recordClip(label, 10, (p, left) => setCalibMsg(`${head}Recording… ${left}s`, p), 'posture');
      keep[label] = n;
      if (n < 20 && !cancelTraining) { failed = `"${label}": too few frames - keep your shoulders in view.`; break; }
    }
  } finally {
    training = false;
  }
  if (cancelTraining || failed) { posture.clear(); setCalibMsg(failed || 'Calibration cancelled.'); }
  else {
    store.set(POSTURE_KEY, posture.toJSON());
    const cv = posture.crossValidate();
    el.calibMsg.hidden = true;
    if (cv) setCalibMsg(`Done. Estimated accuracy ${Math.round(cv.accuracy * 100)}% (tested on parts of your recordings; real use can be lower).`);
  }
  postureVoter.reset(); renderCalib();
}
el.calibBtn.addEventListener('click', calibratePosture);
el.calibClear.addEventListener('click', () => { posture.clear(); store.set(POSTURE_KEY, ''); el.calibMsg.hidden = true; postureVoter.reset(); renderCalib(); });
loadPosture();

/* ---------- sensor model tab ---------- */
let sensorInit = false;
async function initSensor() {
  if (sensorInit) return; sensorInit = true;
  try {
    const [s, st] = await Promise.all([fetch('/api/samples').then(chk), fetch('/api/status').then(chk)]);
    el.sample.innerHTML = s.samples.map((n) => `<option value="${esc(n)}">${esc(n.replace('sample_', '').replace(/_/g, ' '))}</option>`).join('');
    el.acc.textContent = st.test_accuracy != null ? `Test accuracy ${(st.test_accuracy * 100).toFixed(1)}% · ${st.device}` : st.device;
  } catch (e) { sensorInit = false; el.sensorOut.textContent = e.message; }
}
const chk = async (r) => { const j = await r.json(); if (!r.ok) throw new Error(j.error || j.detail || `HTTP ${r.status}`); return j; };

el.predict.addEventListener('click', async () => {
  try {
    const r = await fetch('/api/predict', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ sample: el.sample.value }) }).then(chk);
    const rows = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]);
    el.sensorOut.innerHTML = `<div class="big">${esc(r.prediction)}</div>
      <div class="meta">${(r.confidence * 100).toFixed(1)}% confidence · ${r.latency_ms} ms</div>
      <div class="bars">${rows.map(([k, p]) => `<div class="bar"><span>${esc(k)}</span><div class="track"><div class="fill" style="width:${p * 100}%"></div></div><span class="val">${(p * 100).toFixed(1)}%</span></div>`).join('')}</div>`;
  } catch (e) { el.sensorOut.textContent = e.message; }
});

/* ---------- init ---------- */
loadKnn();
listCameras();
loadModels();
