/**
 * Activity classifier on top of MediaPipe pose landmarks + face blendshapes.
 * Pure logic (no DOM / no MediaPipe import) so it runs in the browser and in Node tests.
 *
 * Output per update():
 *   activity   - Laying | Jumping | Running | Walking | Sitting | Standing | Bending | Upper body only | No person
 *   gesture    - Waving | Hands Up | Hand Raised | Phone Call | Eating / Drinking | Clapping | None
 *   expression - Crying-like | Smiling | Neutral | No face   (an estimate, not a measurement)
 */

export const IDX = {
  NOSE: 0, L_EAR: 7, R_EAR: 8, MOUTH_L: 9, MOUTH_R: 10,
  L_SH: 11, R_SH: 12, L_EL: 13, R_EL: 14, L_WR: 15, R_WR: 16,
  L_HIP: 23, R_HIP: 24, L_KN: 25, R_KN: 26, L_AN: 27, R_AN: 28,
};

const VIS = 0.5;
const WINDOW_MS = 2000;
const CLAP_HOLD_MS = 1500;
const WALK_SH = 0.10;         // shoulder sideways speed, torso-lengths per second
const WALK_HALF = 0.06;       // both halves of the 2 s window must be moving
const WALK_RATIO_LO = 0.6;    // head speed / shoulder speed must be near 1 (they move together)
const WALK_RATIO_HI = 1.8;

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const deg = (r) => (r * 180) / Math.PI;
const avg = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;
const p2p = (xs) => (xs.length ? Math.max(...xs) - Math.min(...xs) : 0);

/** Number of direction reversals whose swing is at least minAmp. */
export function oscillations(series, minAmp) {
  if (series.length < 3) return 0;
  let lastExt = series[0], dir = 0, count = 0;
  for (let i = 1; i < series.length; i++) {
    const v = series[i];
    if (dir === 0) {
      if (Math.abs(v - lastExt) >= minAmp) { dir = v > lastExt ? 1 : -1; lastExt = v; }
    } else if (dir === 1) {
      if (v > lastExt) lastExt = v;
      else if (lastExt - v >= minAmp) { dir = -1; count++; lastExt = v; }
    } else {
      if (v < lastExt) lastExt = v;
      else if (v - lastExt >= minAmp) { dir = 1; count++; lastExt = v; }
    }
  }
  return count;
}

/**
 * Body points used to tell walking from arm/face movements. A lower visibility floor than the
 * rest of the classifier: MediaPipe still estimates hips when they are partly out of frame.
 */
const BODY_VIS = 0.25;
export function bodyPoints(lm, aspect, t) {
  const P = (i) => ({ x: lm[i].x * aspect, y: lm[i].y, v: lm[i].visibility ?? 1 });
  const [ls, rs, lh, rh, nose] = [P(IDX.L_SH), P(IDX.R_SH), P(IDX.L_HIP), P(IDX.R_HIP), P(IDX.NOSE)];
  if (ls.v < BODY_VIS || rs.v < BODY_VIS) return null;
  const sh = mid(ls, rs);
  const hip = lh.v >= BODY_VIS && rh.v >= BODY_VIS ? mid(lh, rh) : null;
  const scale = Math.max(hip ? dist(sh, hip) : dist(ls, rs) * 1.5, 0.05);
  return { t, sh, hip, head: nose.v >= BODY_VIS ? { x: nose.x, y: nose.y } : null, scale };
}

function sidewaysSpeed(w, key) {
  let sum = 0, n = 0;
  for (let i = 1; i < w.length; i++) {
    const a = w[i - 1], b = w[i], dt = (b.t - a.t) / 1000;
    if (dt <= 0 || dt > 0.3 || !a[key] || !b[key]) continue;
    sum += Math.abs(b[key].x - a[key].x) / dt / b.scale; n++;
  }
  return n >= 4 ? sum / n : null;
}

/**
 * Whole-body travel: shoulders and head must move TOGETHER (ratio near 1), and both halves of
 * the window must be moving. Waving/arm raises move the shoulders without the head, turning and
 * bending move the head more than the shoulders, so neither looks like walking.
 * Calibrated on real recordings (stand/sit/wave ~0%, walking ~85% of windows).
 */
export function bodyTravel(bp) {
  const none = { sh: 0, ratio: 0, halfMin: 0, walking: false };
  if (bp.length < 8) return none;
  const sh = sidewaysSpeed(bp, 'sh'), head = sidewaysSpeed(bp, 'head');
  if (sh == null || head == null) return none;
  const h = bp.length >> 1;
  const a = sidewaysSpeed(bp.slice(0, h), 'sh'), b = sidewaysSpeed(bp.slice(h), 'sh');
  const halfMin = a != null && b != null ? Math.min(a, b) : 0;
  const ratio = head / Math.max(sh, 1e-6);
  return {
    sh, ratio, halfMin,
    walking: sh >= WALK_SH && halfMin >= WALK_HALF && ratio >= WALK_RATIO_LO && ratio <= WALK_RATIO_HI,
  };
}

/** Mean speed (scale-units per second) of each tracked joint over the window. */
export function jointSpeeds(h) {
  const out = [];
  for (let k = 0; k < 6; k++) {
    let sum = 0, n = 0;
    for (let i = 1; i < h.length; i++) {
      const a = h[i - 1].j?.[k], b = h[i].j?.[k];
      const dt = (h[i].t - h[i - 1].t) / 1000;
      if (a && b && dt > 0 && dt < 0.3) { sum += Math.hypot(b.x - a.x, b.y - a.y) / dt; n++; }
    }
    out.push(n ? Math.min(sum / n, 5) : 0);
  }
  return out;
}

const HAND_NAMES = {
  Closed_Fist: 'Fist', Open_Palm: 'Open palm', Pointing_Up: 'Pointing up',
  Thumb_Down: 'Thumbs down', Thumb_Up: 'Thumbs up', Victory: 'Peace sign', ILoveYou: 'I love you',
};
const SPECIFIC_SIGNS = new Set(['Pointing up', 'Thumbs down', 'Thumbs up', 'Peace sign', 'I love you']);

/** Best confident finger gesture from a MediaPipe GestureRecognizer result, or 'None'. */
export function handGesture(result, minScore = 0.6) {
  let best = null;
  for (const cats of result?.gestures ?? []) {
    const c = cats?.[0];
    if (c && c.categoryName !== 'None' && c.score >= minScore && HAND_NAMES[c.categoryName] && (!best || c.score > best.score)) best = c;
  }
  return best ? HAND_NAMES[best.categoryName] : 'None';
}

/**
 * Combine the arm-level gesture (from body pose) with the finger-level gesture (from the hand model).
 * Specific finger signs win; a plain fist/open palm defers to what the arm is doing (waving, phone call...).
 */
export function mergeGesture(pose, hand) {
  if (!hand || hand === 'None') return pose || 'None';
  if (SPECIFIC_SIGNS.has(hand)) return hand;
  return pose && pose !== 'None' && pose !== 'Hand Raised' ? pose : hand;
}


/**
 * Body-position features for the sit/stand calibration: size and height in the frame, and how much
 * of the lower body is visible. Deliberately ignores arms and motion, so what you do with your hands
 * cannot change the answer.
 */
export function postureVector(lm, aspect = 1) {
  if (!lm || (lm[IDX.L_SH].visibility ?? 1) < 0.3 || (lm[IDX.R_SH].visibility ?? 1) < 0.3) return null;
  const m = (i, j) => [((lm[i].x + lm[j].x) / 2) * aspect, (lm[i].y + lm[j].y) / 2];
  const vis = (i, j) => ((lm[i].visibility ?? 1) + (lm[j].visibility ?? 1)) / 2;
  const sh = m(IDX.L_SH, IDX.R_SH), hip = m(IDX.L_HIP, IDX.R_HIP), knee = m(IDX.L_KN, IDX.R_KN), ank = m(IDX.L_AN, IDX.R_AN);
  const shW = Math.hypot((lm[IDX.L_SH].x - lm[IDX.R_SH].x) * aspect, lm[IDX.L_SH].y - lm[IDX.R_SH].y);
  const torso = Math.hypot(sh[0] - hip[0], sh[1] - hip[1]);
  const thigh = Math.max(Math.min((knee[1] - hip[1]) / Math.max(torso, 0.05), 1.5), -0.5);
  return [
    shW, sh[1], lm[IDX.NOSE].y,
    vis(IDX.L_HIP, IDX.R_HIP), vis(IDX.L_KN, IDX.R_KN), vis(IDX.L_AN, IDX.R_AN),
    hip[1], knee[1], ank[1],
    torso / Math.max(shW, 1e-3), thigh,
  ];
}

/**
 * Clapping, eating/drinking and phone-call from the HAND model (palms and fingertips), measured against
 * the face. Distances are in shoulder-widths. Thresholds come from real recordings: fingertips sit about
 * 0.06 from the mouth when eating, ~0.2-0.4 when touching the face, ~0.5 with palms together at the chest;
 * clapping shows a small repeated change in palm distance.
 */
export class HandActions {
  constructor() { this.palm = []; this.clapUntil = 0; this.twoAt = -1e9; }
  reset() { this.palm = []; this.clapUntil = 0; this.twoAt = -1e9; }

  /** @param {{hands:{x:number,y:number}[][], landmarks:any[]|null, aspect:number, t:number}} input */
  update({ hands, landmarks, aspect = 1, t }) {
    this.palm = this.palm.filter((e) => t - e.t <= 2000);
    // Claps come in bursts, and the hand model drops a hand when the palms meet (only 1 hand visible in
    // ~2/3 of clap frames), so a detected clap is held briefly instead of flickering.
    if (!landmarks || !hands || !hands.length) return t < this.clapUntil ? 'Clapping' : 'None';
    const P = (i) => ({ x: landmarks[i].x * aspect, y: landmarks[i].y, v: landmarks[i].visibility ?? 1 });
    if (P(IDX.L_SH).v < 0.3 || P(IDX.R_SH).v < 0.3) return 'None';
    const shW = dist(P(IDX.L_SH), P(IDX.R_SH));
    if (shW < 0.03) return 'None';
    const mouth = mid(P(IDX.MOUTH_L), P(IDX.MOUTH_R));
    const ears = [P(IDX.L_EAR), P(IDX.R_EAR)].filter((e) => e.v >= 0.2);
    const H = hands.map((h) => {
      const pts = h.map((p) => ({ x: p.x * aspect, y: p.y }));
      const cen = { x: avg(pts.map((p) => p.x)), y: avg(pts.map((p) => p.y)) };
      return { cen, tips: [4, 8, 12].map((i) => pts[i]), palm: pts[9] };
    });

    if (H.length >= 2) { this.palm.push({ t, d: dist(H[0].palm, H[1].palm) / shW }); this.twoAt = t; }
    if (this.palm.length >= 8 && this.palm[this.palm.length - 1].t - this.palm[0].t >= 1000) {
      const ds = this.palm.map((e) => e.d);
      if (oscillations(ds, 0.08) >= 2 && p2p(ds) >= 0.15 && Math.min(...ds) < 0.6) { this.clapUntil = t + CLAP_HOLD_MS; }
    }
    if (t < this.clapUntil) return 'Clapping';

    const info = H.map((h) => ({
      dTip: Math.min(...h.tips.map((p) => dist(p, mouth))) / shW,
      dMouth: dist(h.cen, mouth) / shW,
      dEar: ears.length ? Math.min(...ears.map((e) => dist(h.cen, e))) / shW : Infinity,
    }));
    // one hand right at the mouth (two hands close together near the face is more likely clapping)
    for (let i = 0; i < info.length; i++) {
      if (info[i].dTip < 0.15 && info[i].dMouth < 0.22) {
        const others = info.filter((_, j) => j !== i);
        if (!others.length || others.every((o) => o.dMouth > 0.5)) return 'Eating / Drinking';
      }
    }
    const clapLike = t - this.twoAt < 1500;   // hands were together a moment ago
    if (clapLike && H.length < 2) return 'None';
    for (let i = 0; i < info.length; i++) {
      const o = info[i], others = info.filter((_, j) => j !== i);
      if (o.dEar < 0.32 && o.dMouth >= 0.22 && others.every((x) => x.dMouth > 0.5 && x.dEar > 0.5)) return 'Phone Call';
    }
    if (H.length >= 2 && dist(H[0].palm, H[1].palm) / shW < 0.7) return 'Hands together';
    for (const o of info) if (Math.min(o.dMouth, o.dEar) < 0.55) return 'Hand at face';
    return 'None';
  }
}

/** Final gesture from arm pose, finger sign and hand-near-face action. */
export function finalGesture(pose, finger, action) {
  if (action === 'Clapping' || action === 'Eating / Drinking' || action === 'Phone Call') return action;
  if (SPECIFIC_SIGNS.has(finger)) return finger;
  if (action === 'Hands together') return action;
  if (action === 'Hand at face') return pose === 'Waving' ? pose : 'Hand at face';
  return mergeGesture(pose, finger);
}

/** Majority vote over a sliding time window. */
export class Voter {
  constructor(windowMs) { this.windowMs = windowMs; this.items = []; }
  push(t, label) {
    this.items.push({ t, label });
    while (this.items.length && t - this.items[0].t > this.windowMs) this.items.shift();
    const counts = new Map();
    for (const it of this.items) counts.set(it.label, (counts.get(it.label) || 0) + 1);
    let best = label, n = 0;
    for (const [k, v] of counts) if (v > n) { best = k; n = v; }
    return { label: best, share: n / this.items.length };
  }
  reset() { this.items = []; }
}

/** Geometry of one frame (aspect-corrected). Returns null when no body is visible. */
export function frameFeatures(lm, aspect = 1) {
  const P = (i) => ({ x: lm[i].x * aspect, y: lm[i].y, v: lm[i].visibility ?? 1 });
  const pts = {};
  for (let i = 0; i < lm.length; i++) pts[i] = P(i);
  const seen = (...ids) => ids.every((i) => pts[i] && pts[i].v >= VIS);

  if (!seen(IDX.L_SH, IDX.R_SH)) return null;
  const shMid = mid(pts[IDX.L_SH], pts[IDX.R_SH]);
  const shW = dist(pts[IDX.L_SH], pts[IDX.R_SH]);
  const hips = seen(IDX.L_HIP, IDX.R_HIP);
  const hipMid = hips ? mid(pts[IDX.L_HIP], pts[IDX.R_HIP]) : null;
  const torsoLen = hips ? dist(shMid, hipMid) : shW * 1.5;
  const scale = Math.max(torsoLen, 0.05);

  // 0 = upright, 90 = horizontal, >90 = upside down
  const torsoAngle = hips ? deg(Math.atan2(Math.abs(hipMid.x - shMid.x), hipMid.y - shMid.y)) : null;

  const thighs = [];
  if (seen(IDX.L_HIP, IDX.L_KN)) thighs.push((pts[IDX.L_KN].y - pts[IDX.L_HIP].y) / scale);
  if (seen(IDX.R_HIP, IDX.R_KN)) thighs.push((pts[IDX.R_KN].y - pts[IDX.R_HIP].y) / scale);
  const thighV = thighs.length ? avg(thighs) : null;

  const ankles = seen(IDX.L_AN, IDX.R_AN);
  return {
    pts, seen, shMid, shW, hipMid, hips, scale, torsoAngle, thighV,
    ankleL: ankles ? pts[IDX.L_AN] : null,
    ankleR: ankles ? pts[IDX.R_AN] : null,
  };
}

/** Face blendshape categories -> compact scores in 0..1. */
export function faceScores(categories) {
  if (!categories || !categories.length) return null;
  const m = {};
  for (const c of categories) m[c.categoryName] = c.score;
  const g = (k) => m[k] ?? 0;
  return {
    frown: (g('mouthFrownLeft') + g('mouthFrownRight')) / 2,
    brow: g('browInnerUp'),
    squint: (g('eyeSquintLeft') + g('eyeSquintRight')) / 2,
    smile: (g('mouthSmileLeft') + g('mouthSmileRight')) / 2,
    jaw: g('jawOpen'),
    blink: (g('eyeBlinkLeft') + g('eyeBlinkRight')) / 2,
    press: (g('mouthPressLeft') + g('mouthPressRight')) / 2,
  };
}

export function rawExpression(f) {
  if (!f) return 'No face';
  if (f.smile > 0.4) return 'Smiling';
  if ((f.frown > 0.3 && f.brow > 0.3) || f.frown * 0.5 + f.brow * 0.35 + f.squint * 0.15 > 0.4) return 'Crying-like';
  return 'Neutral';
}

/** Numeric feature vector for the trainable kNN (pose + motion + face). */
export function buildFeatureVector(frame, win, face) {
  if (!frame) return null;
  const origin = frame.shMid;
  const v = [];
  for (const i of [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    const p = frame.pts[i];
    if (p && p.v >= 0.3) v.push((p.x - origin.x) / frame.scale, (p.y - origin.y) / frame.scale);
    else v.push(0, 0);
  }
  v.push(
    frame.torsoAngle == null ? 0 : frame.torsoAngle / 90,
    frame.thighV ?? 0,
    win.ankleP2P, win.ankleRev / 6, win.hipP2P,
    win.wristLRev / 6, win.wristRRev / 6, win.wristDistP2P,
    ...win.speeds, win.body.sh, win.body.halfMin, Math.min(win.body.ratio, 3),
  );
  const f = face || { frown: 0, brow: 0, squint: 0, smile: 0, jaw: 0, blink: 0 };
  v.push(f.frown, f.brow, f.squint, f.smile, f.jaw, f.blink);
  return v;
}

export class ActivityClassifier {
  constructor() {
    this.history = [];
    this.face = null;
    this.faceT = -1e9;
    this.prevPosture = 'Standing';
    // Wrist-based clap / eat / phone guesses are poor (the wrist is as near the ear as the mouth);
    // the app turns them off when the hand model is available.
    this.wristFaceGestures = true;
    this.activityVoter = new Voter(900);
    this.gestureVoter = new Voter(600);
    this.expressionVoter = new Voter(800);
    this.lastFeatureVector = null;
    this.hasBody = false;
  }

  reset() {
    this.history = [];
    this.face = null;
    this.activityVoter.reset(); this.gestureVoter.reset(); this.expressionVoter.reset();
    this.lastFeatureVector = null;
  }

  windowStats(t) {
    const h = this.history.filter((e) => t - e.t <= WINDOW_MS);
    const col = (k) => h.map((e) => e[k]).filter((v) => v != null);
    const span = h.length ? h[h.length - 1].t - h[0].t : 0;
    const ankleDiff = col('ankleDiff'), hipY = col('hipY');
    return {
      span,
      ankleP2P: p2p(ankleDiff), ankleRev: oscillations(ankleDiff, 0.12),
      hipP2P: p2p(hipY), hipRev: oscillations(hipY, 0.15),
      wristLRev: oscillations(col('wlx'), 0.15), wristRRev: oscillations(col('wrx'), 0.15),
      wristDistP2P: p2p(col('wDist')), wristDistRev: oscillations(col('wDist'), 0.2),
      wristDistMin: col('wDist').length ? Math.min(...col('wDist')) : 1,
      speeds: jointSpeeds(h),
      body: bodyTravel(h.map((e) => e.bp).filter(Boolean)),
    };
  }

  /** @param {{landmarks:any[]|null, blendshapes:any[]|null, t:number, aspect:number}} input */
  update({ landmarks, blendshapes, t, aspect = 1 }) {
    if (blendshapes) { this.face = faceScores(blendshapes); this.faceT = t; }
    if (t - this.faceT > 1500) this.face = null;

    const frame = landmarks ? frameFeatures(landmarks, aspect) : null;
    if (frame) {
      const { pts, scale } = frame;
      const wl = frame.seen(IDX.L_WR) ? pts[IDX.L_WR] : null;
      const wr = frame.seen(IDX.R_WR) ? pts[IDX.R_WR] : null;
      this.history.push({
        t,
        hipY: frame.hips ? frame.hipMid.y / scale : null,
        bp: bodyPoints(landmarks, aspect, t),
        ankleDiff: frame.ankleL ? (frame.ankleL.y - frame.ankleR.y) / scale : null,
        wlx: wl ? wl.x / scale : null,
        wrx: wr ? wr.x / scale : null,
        wDist: wl && wr ? dist(wl, wr) / scale : null,
        // joints relative to the shoulders (limb motion) + shoulder centre (body translation)
        j: [IDX.NOSE, IDX.L_WR, IDX.R_WR, IDX.L_AN, IDX.R_AN].map((i) => (
          frame.seen(i) ? { x: (pts[i].x - frame.shMid.x) / scale, y: (pts[i].y - frame.shMid.y) / scale } : null
        )).concat([{ x: frame.shMid.x / scale, y: frame.shMid.y / scale }]),
      });
    }
    while (this.history.length && t - this.history[0].t > WINDOW_MS) this.history.shift();
    const win = this.windowStats(t);
    const warm = win.span >= 1200;

    let raw = 'No person', posture = null, hint = '';
    if (frame) {
      ({ posture, hint } = this.posture(frame));
      raw = posture;
      if (frame.torsoAngle != null && frame.torsoAngle > 60) raw = 'Laying';
      else if (warm) {
        const jumping = frame.hips && win.hipP2P >= 0.35 && win.hipRev >= 2 && win.ankleRev < 3;
        const legsRun = frame.ankleL && win.ankleRev >= 5 && win.ankleP2P >= 0.5;
        const legsWalk = frame.ankleL && win.ankleRev >= 3 && win.ankleP2P >= 0.12;
        // Whole-body travel (head + shoulders moving together); works with only the upper body in view.
        const travelWalk = win.body.walking;
        if (jumping) raw = 'Jumping';
        else if (legsRun) raw = 'Running';
        else if (legsWalk || travelWalk) raw = 'Walking';
      }
    }

    const act = this.activityVoter.push(t, raw);
    const gest = this.gestureVoter.push(t, frame ? this.gesture(frame, win, warm) : 'None');
    const expr = this.expressionVoter.push(t, rawExpression(this.face));
    this.lastFeatureVector = buildFeatureVector(frame, win, this.face);
    this.hasBody = !!frame && warm;

    return {
      activity: act.label, share: act.share, rawActivity: raw, posture, hint,
      gesture: gest.label, expression: expr.label, face: this.face,
      featureVector: this.lastFeatureVector, bodyVisible: !!frame, body: win.body,
    };
  }

  posture(f) {
    const hint = "Can't tell sitting from standing from this view. Use 'Calibrate sitting & standing' below so it learns your camera.";
    // The knee/hip angle is only trustworthy when the whole leg is in view; with legs cropped at the
    // frame edge it reads standing as sitting (measured on real recordings).
    if (!f.hips || f.thighV == null || !f.ankleL) return { posture: 'Upper body only', hint };
    let p = this.prevPosture;
    if (f.thighV < 0.55) p = 'Sitting';
    else if (f.thighV > 0.75) p = 'Standing';
    if (p === 'Standing' && f.torsoAngle != null && f.torsoAngle > 35) p = 'Bending';
    if (p !== 'Bending') this.prevPosture = p;
    return { posture: p, hint: '' };
  }

  gesture(f, win, warm) {
    const { pts, scale, shMid } = f;
    const wl = f.seen(IDX.L_WR) ? pts[IDX.L_WR] : null;
    const wr = f.seen(IDX.R_WR) ? pts[IDX.R_WR] : null;
    if (!wl && !wr) return 'None';
    const nose = pts[IDX.NOSE];
    const mouth = mid(pts[IDX.MOUTH_L], pts[IDX.MOUTH_R]);
    const ears = [pts[IDX.L_EAR], pts[IDX.R_EAR]].filter((e) => e && e.v >= 0.3);
    const hands = [wl, wr].filter(Boolean);

    if (this.wristFaceGestures && warm && wl && wr && win.wristDistRev >= 3 && win.wristDistP2P >= 0.4 && win.wristDistMin < 0.3) return 'Clapping';

    const raised = hands.filter((w) => w.y < shMid.y - 0.4 * scale);
    if (wl && wr && wl.y < nose.y && wr.y < nose.y) return 'Hands Up';
    if (raised.length) {
      const rev = Math.max(win.wristLRev, win.wristRRev);
      return warm && rev >= 3 ? 'Waving' : 'Hand Raised';
    }
    if (this.wristFaceGestures) for (const w of hands) {
      const dMouth = dist(w, mouth);
      const dEar = ears.length ? Math.min(...ears.map((e) => dist(w, e))) : Infinity;
      if (dMouth < 0.3 * scale && dMouth < dEar) return 'Eating / Drinking';
      if (dEar < 0.5 * scale) return 'Phone Call';
    }
    return 'None';
  }
}
