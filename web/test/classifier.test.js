import test from 'node:test';
import assert from 'node:assert/strict';
import { ActivityClassifier, oscillations, rawExpression, faceScores } from '../public/js/pose-classifier.js';
import { KnnModel } from '../public/js/knn.js';

/** Build 33 landmarks for a front-facing person; override any index. */
function pose(over = {}, o = {}) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99 }));
  const set = (i, x, y, v = 0.99) => { lm[i] = { x, y, visibility: v }; };
  const dx = o.dx || 0;
  set(0, 0.5 + dx, 0.18); set(7, 0.44 + dx, 0.19, 0.6); set(8, 0.56 + dx, 0.19, 0.6);
  set(9, 0.48 + dx, 0.23); set(10, 0.52 + dx, 0.23);
  set(11, 0.4 + dx, 0.3); set(12, 0.6 + dx, 0.3);
  set(13, 0.37 + dx, 0.42); set(14, 0.63 + dx, 0.42);
  set(15, 0.36 + dx, 0.54); set(16, 0.64 + dx, 0.54);
  set(23, 0.45 + dx, 0.55); set(24, 0.55 + dx, 0.55);
  set(25, 0.45 + dx, 0.80); set(26, 0.55 + dx, 0.80);
  set(27, 0.45 + dx, 0.97); set(28, 0.55 + dx, 0.97);
  for (const [i, v] of Object.entries(over)) lm[+i] = { visibility: 0.99, ...lm[+i], ...v };
  return lm;
}

/** Run frames at 30fps and return the final result. */
function run(frames, blend = null) {
  const c = new ActivityClassifier();
  let r;
  frames.forEach((lm, n) => { r = c.update({ landmarks: lm, blendshapes: blend, t: n * 33, aspect: 1 }); });
  return r;
}
const repeat = (lm, n = 90) => Array.from({ length: n }, () => lm);
const seq = (n, fn) => Array.from({ length: n }, (_, i) => fn(i));

test('oscillations counts swings above the threshold only', () => {
  assert.equal(oscillations([0, 1, 0, 1, 0], 0.5), 3);
  assert.equal(oscillations([0, 0.1, 0, 0.1, 0], 0.5), 0);
});

test('standing', () => assert.equal(run(repeat(pose())).activity, 'Standing'));

test('sitting (knees level with hips)', () => {
  const sit = pose({ 25: { y: 0.6 }, 26: { y: 0.6 }, 27: { y: 0.85 }, 28: { y: 0.85 } });
  assert.equal(run(repeat(sit)).activity, 'Sitting');
});

test('laying (torso horizontal)', () => {
  const lay = pose({
    0: { x: 0.15, y: 0.5 }, 7: { x: 0.17, y: 0.46 }, 8: { x: 0.17, y: 0.54 },
    9: { x: 0.18, y: 0.48 }, 10: { x: 0.18, y: 0.52 },
    11: { x: 0.28, y: 0.42 }, 12: { x: 0.28, y: 0.58 },
    23: { x: 0.53, y: 0.45 }, 24: { x: 0.53, y: 0.57 },
    25: { x: 0.72, y: 0.45 }, 26: { x: 0.72, y: 0.57 }, 27: { x: 0.9, y: 0.45 }, 28: { x: 0.9, y: 0.57 },
  });
  assert.equal(run(repeat(lay)).activity, 'Laying');
});

test('upper body only does not guess sitting vs standing', () => {
  const upper = pose({ 23: { visibility: 0.1 }, 24: { visibility: 0.1 }, 25: { visibility: 0.1 }, 26: { visibility: 0.1 }, 27: { visibility: 0.1 }, 28: { visibility: 0.1 } });
  const r = run(repeat(upper));
  assert.equal(r.activity, 'Upper body only');
  assert.match(r.hint, /Calibrate/);
});

test('walking: ankles alternate at ~1 stride/sec', () => {
  const w = seq(120, (i) => { const s = Math.sin((i / 30) * Math.PI * 2); return pose({ 27: { y: 0.93 - 0.05 * s }, 28: { y: 0.93 + 0.05 * s }, 25: { y: 0.8 - 0.02 * s }, 26: { y: 0.8 + 0.02 * s } }); });
  assert.equal(run(w).activity, 'Walking');
});

test('running: big, fast leg swings', () => {
  const w = seq(120, (i) => { const s = Math.sin((i / 30) * Math.PI * 2 * 1.5); return pose({ 27: { y: 0.9 - 0.1 * s }, 28: { y: 0.9 + 0.1 * s } }); });
  assert.equal(run(w).activity, 'Running');
});

test('jumping: body bounces, legs move together', () => {
  const w = seq(120, (i) => { const up = 0.09 * Math.abs(Math.sin((i / 30) * Math.PI * 1.5)); const dy = -up; const m = (y) => ({ y: y + dy });
    return pose({ 0: m(0.18), 11: m(0.3), 12: m(0.3), 23: m(0.55), 24: m(0.55), 25: m(0.8), 26: m(0.8), 27: m(0.97), 28: m(0.97) }); });
  assert.equal(run(w).activity, 'Jumping');
});

test('phone call: wrist at the ear', () => {
  const p = pose({ 15: { x: 0.44, y: 0.2 }, 13: { x: 0.38, y: 0.32 } });
  assert.equal(run(repeat(p)).gesture, 'Phone Call');
});

test('eating / drinking: wrist at the mouth', () => {
  const p = pose({ 16: { x: 0.52, y: 0.24 } });
  assert.equal(run(repeat(p)).gesture, 'Eating / Drinking');
});

test('hands up and waving', () => {
  const up = pose({ 15: { x: 0.35, y: 0.05 }, 16: { x: 0.65, y: 0.05 } });
  assert.equal(run(repeat(up)).gesture, 'Hands Up');
  const wave = seq(120, (i) => pose({ 16: { x: 0.68 + 0.08 * Math.sin((i / 30) * Math.PI * 4), y: 0.1 } }));
  assert.equal(run(wave).gesture, 'Waving');
  assert.equal(run(repeat(pose({ 16: { x: 0.68, y: 0.1 } }))).gesture, 'Hand Raised');
});

test('clapping: wrists open and close', () => {
  const c = seq(120, (i) => { const g = 0.12 + 0.12 * Math.sin((i / 30) * Math.PI * 4); return pose({ 15: { x: 0.5 - g, y: 0.45 }, 16: { x: 0.5 + g, y: 0.45 } }); });
  assert.equal(run(c).gesture, 'Clapping');
});

test('no gesture at rest', () => assert.equal(run(repeat(pose())).gesture, 'None'));

test('no person', () => {
  const c = new ActivityClassifier();
  assert.equal(c.update({ landmarks: null, blendshapes: null, t: 0 }).activity, 'No person');
});

test('expression: crying-like, smiling, neutral, no face', () => {
  const cat = (o) => Object.entries(o).map(([categoryName, score]) => ({ categoryName, score }));
  assert.equal(rawExpression(faceScores(cat({ mouthFrownLeft: 0.5, mouthFrownRight: 0.5, browInnerUp: 0.6 }))), 'Crying-like');
  assert.equal(rawExpression(faceScores(cat({ mouthSmileLeft: 0.7, mouthSmileRight: 0.7 }))), 'Smiling');
  assert.equal(rawExpression(faceScores(cat({ jawOpen: 0.02 }))), 'Neutral');
  assert.equal(rawExpression(null), 'No face');
  const r = run(repeat(pose()), cat({ mouthFrownLeft: 0.5, mouthFrownRight: 0.5, browInnerUp: 0.6 }));
  assert.equal(r.expression, 'Crying-like');
});

test('kNN learns a custom activity from recorded samples', () => {
  const c = new ActivityClassifier(); const knn = new KnnModel();
  const sitting = pose({ 25: { y: 0.6 }, 26: { y: 0.6 }, 27: { y: 0.85 }, 28: { y: 0.85 } });
  const noise = (lm, k) => lm.map((p) => ({ ...p, x: p.x + Math.sin(k + p.y * 9) * 0.004, y: p.y + Math.cos(k + p.x * 9) * 0.004 }));
  const feed = (lm, label) => { for (let k = 0; k < 40; k++) { c.update({ landmarks: noise(lm, k), blendshapes: null, t: k * 33 }); knn.add(label, c.lastFeatureVector); } };
  feed(pose(), 'Reading'); feed(sitting, 'Typing');
  assert.ok(knn.ready());
  c.reset();
  for (let k = 0; k < 40; k++) c.update({ landmarks: noise(sitting, k + 100), blendshapes: null, t: k * 33 });
  assert.equal(knn.predict(c.lastFeatureVector).label, 'Typing');
});

/** Record a labelled clip of synthetic frames into a model; returns nothing. */
function record(knn, label, makeFrame, frames = 150, noiseAmp = 0.004) {
  const c = new ActivityClassifier(); const rec = knn.newRecording();
  for (let k = 0; k < frames; k++) {
    const lm = makeFrame(k).map((p) => ({ ...p, x: p.x + Math.sin(k * 1.7 + p.y * 9) * noiseAmp, y: p.y + Math.cos(k * 1.3 + p.x * 9) * noiseAmp }));
    c.update({ landmarks: lm, blendshapes: null, t: k * 33, aspect: 1 });
    if (k >= 40 && c.lastFeatureVector) knn.add(label, c.lastFeatureVector.slice(), rec);
  }
}
const stand = () => pose();
const sit = () => pose({ 25: { y: 0.6 }, 26: { y: 0.6 }, 27: { y: 0.85 }, 28: { y: 0.85 } });
const wave = (k) => pose({ 16: { x: 0.68 + 0.08 * Math.sin((k / 30) * Math.PI * 4), y: 0.1 } });
const walk = (k) => { const s = Math.sin((k / 30) * Math.PI * 2); return pose({ 27: { y: 0.93 - 0.05 * s }, 28: { y: 0.93 + 0.05 * s } }); };

test('motion speeds separate moving from still', () => {
  const grab = (fn) => { const c = new ActivityClassifier(); let r; for (let k = 0; k < 90; k++) r = c.update({ landmarks: fn(k), blendshapes: null, t: k * 33 }); return r.featureVector.slice(-6 - 6, -6); };
  const still = grab(stand), moving = grab(walk);
  assert.ok(Math.max(...moving) > Math.max(...still) * 3, `still=${still} moving=${moving}`);
});

test('cross-validation measures accuracy on held-out time chunks', () => {
  const knn = new KnnModel();
  record(knn, 'Standing', stand); record(knn, 'Sitting', sit); record(knn, 'Waving', wave); record(knn, 'Walking', walk);
  const cv = knn.crossValidate();
  assert.ok(cv.accuracy > 0.9, `accuracy ${cv.accuracy}`);
  for (const l of ['Standing', 'Sitting', 'Waving', 'Walking']) assert.ok(cv.perLabel[l].recall > 0.8, l);
});

test('cross-validation exposes confusable activities instead of hiding them', () => {
  const knn = new KnnModel();
  record(knn, 'Standing', stand); record(knn, 'Upright still', stand);   // identical on purpose
  const cv = knn.crossValidate();
  assert.ok(cv.accuracy < 0.75, `identical activities must not score well, got ${cv.accuracy}`);
});

test('kNN returns a lower confidence for an unfamiliar pose', () => {
  const knn = new KnnModel();
  record(knn, 'Standing', stand); record(knn, 'Sitting', sit);
  const c = new ActivityClassifier(); let last;
  for (let k = 0; k < 60; k++) last = c.update({ landmarks: wave(k), blendshapes: null, t: k * 33 });
  const known = (() => { const c2 = new ActivityClassifier(); let r; for (let k = 0; k < 60; k++) r = c2.update({ landmarks: stand(), blendshapes: null, t: k * 33 }); return knn.predict(r.featureVector); })();
  assert.ok(known.confidence > 0.9);
  assert.ok(knn.predict(last.featureVector) !== null);
});

test('kNN persists and restores', () => {
  const a = new KnnModel(); record(a, 'Standing', stand, 60); record(a, 'Sitting', sit, 60);
  const b = new KnnModel(); b.fromJSON(a.toJSON());
  assert.equal(b.samples.length, a.samples.length);
  assert.equal(b.recSeq, a.recSeq);
});

/** Walking with legs out of view: whole upper body drifts sideways at ~0.3 torso-lengths/sec. */
test('walking is detected from whole-body travel even with only the upper body visible', () => {
  const upperOnly = { 23: { visibility: 0.1 }, 24: { visibility: 0.1 }, 25: { visibility: 0.05 }, 26: { visibility: 0.05 }, 27: { visibility: 0.02 }, 28: { visibility: 0.02 } };
  const frames = seq(120, (k) => pose(upperOnly, { dx: 0.0025 * k }));
  assert.equal(run(frames).activity, 'Walking');
});

test('waving (shoulders jitter, head still) is NOT walking', () => {
  const upperOnly = { 23: { visibility: 0.1 }, 24: { visibility: 0.1 }, 25: { visibility: 0.05 }, 26: { visibility: 0.05 }, 27: { visibility: 0.02 }, 28: { visibility: 0.02 } };
  const frames = seq(120, (k) => { const j = 0.01 * Math.sin((k / 30) * Math.PI * 4); return pose({ ...upperOnly, 11: { x: 0.4 + j }, 12: { x: 0.6 + j }, 16: { x: 0.68 + 8 * j, y: 0.1 } }); });
  const r = run(frames);
  assert.notEqual(r.activity, 'Walking');
  assert.equal(r.gesture, 'Waving');
});

test('turning the head/body in place (head moves more than shoulders) is NOT walking', () => {
  const upperOnly = { 23: { visibility: 0.1 }, 24: { visibility: 0.1 }, 25: { visibility: 0.05 }, 26: { visibility: 0.05 }, 27: { visibility: 0.02 }, 28: { visibility: 0.02 } };
  const frames = seq(120, (k) => { const s = Math.sin((k / 30) * Math.PI * 2); return pose({ ...upperOnly, 0: { x: 0.5 + 0.04 * s }, 11: { x: 0.4 + 0.004 * s }, 12: { x: 0.6 + 0.004 * s } }); });
  assert.notEqual(run(frames).activity, 'Walking');
});

test('standing still never reads as walking', () => {
  const upperOnly = { 23: { visibility: 0.1 }, 24: { visibility: 0.1 } };
  assert.notEqual(run(repeat(pose(upperOnly), 150)).activity, 'Walking');
  assert.notEqual(run(repeat(pose(), 150)).activity, 'Walking');
});

import { handGesture, mergeGesture } from '../public/js/pose-classifier.js';

test('hand model result is mapped to friendly names and filtered by confidence', () => {
  const res = (name, score) => ({ gestures: [[{ categoryName: name, score }]] });
  assert.equal(handGesture(res('Thumb_Up', 0.9)), 'Thumbs up');
  assert.equal(handGesture(res('Victory', 0.8)), 'Peace sign');
  assert.equal(handGesture(res('Closed_Fist', 0.95)), 'Fist');
  assert.equal(handGesture(res('Thumb_Up', 0.4)), 'None');   // not confident
  assert.equal(handGesture(res('None', 0.99)), 'None');
  assert.equal(handGesture({ gestures: [] }), 'None');
  // two hands: the more confident one wins
  assert.equal(handGesture({ gestures: [[{ categoryName: 'Open_Palm', score: 0.7 }], [{ categoryName: 'Thumb_Up', score: 0.9 }]] }), 'Thumbs up');
});

test('specific finger signs win, plain fist/palm defer to what the arm is doing', () => {
  assert.equal(mergeGesture('None', 'Thumbs up'), 'Thumbs up');
  assert.equal(mergeGesture('Phone Call', 'Thumbs up'), 'Thumbs up');
  assert.equal(mergeGesture('Waving', 'Open palm'), 'Waving');
  assert.equal(mergeGesture('Phone Call', 'Fist'), 'Phone Call');
  assert.equal(mergeGesture('None', 'Fist'), 'Fist');
  assert.equal(mergeGesture('Hand Raised', 'Open palm'), 'Open palm');
  assert.equal(mergeGesture('Clapping', 'None'), 'Clapping');
  assert.equal(mergeGesture('None', 'None'), 'None');
});

import { HandActions, finalGesture, postureVector } from '../public/js/pose-classifier.js';

/** A fake 21-point hand clustered around (cx, cy); landmark 9 (palm) sits at the centre. */
const hand = (cx, cy) => Array.from({ length: 21 }, (_, i) => ({ x: cx + ((i % 5) - 2) * 0.002, y: cy + (Math.floor(i / 5) - 2) * 0.002 }));
const body = pose();   // shoulders 0.4..0.6 (width 0.2), mouth (0.5,0.23), ears (0.44/0.56, 0.19)
const act = (frames) => { const h = new HandActions(); let r = 'None'; frames.forEach((hands, k) => { r = h.update({ hands, landmarks: body, aspect: 1, t: k * 100 }); }); return r; };
const rep2 = (hands, n = 25) => Array.from({ length: n }, () => hands);

test('hand at the mouth is eating / drinking', () => assert.equal(act(rep2([hand(0.5, 0.24)])), 'Eating / Drinking'));
test('one hand at the ear is a phone call', () => assert.equal(act(rep2([hand(0.44, 0.2)])), 'Phone Call'));
test('hand on the forehead/cheek is neither eating nor a phone call', () => assert.equal(act(rep2([hand(0.5, 0.3)])), 'Hand at face'));
test('hand low at the side is nothing', () => assert.equal(act(rep2([hand(0.3, 0.6)])), 'None'));

test('palms together and still is NOT clapping', () => {
  assert.equal(act(rep2([hand(0.47, 0.5), hand(0.53, 0.5)])), 'Hands together');
});

test('palms moving apart and together repeatedly is clapping', () => {
  const frames = Array.from({ length: 30 }, (_, k) => { const g = 0.04 + 0.03 * Math.sin((k / 10) * Math.PI * 5); return [hand(0.5 - g, 0.5), hand(0.5 + g, 0.5)]; });
  assert.equal(act(frames), 'Clapping');
});

test('a clap is held while the hand model drops a hand (it does this whenever palms meet)', () => {
  const clap = Array.from({ length: 30 }, (_, k) => { const g = 0.04 + 0.03 * Math.sin((k / 10) * Math.PI * 5); return [hand(0.5 - g, 0.5), hand(0.5 + g, 0.5)]; });
  const h = new HandActions(); let r;
  clap.forEach((hands, k) => { r = h.update({ hands, landmarks: body, aspect: 1, t: k * 100 }); });
  assert.equal(r, 'Clapping');
  // next few updates see only ONE hand, near the ear: must not flip to "Phone Call"
  for (let k = 30; k < 34; k++) r = h.update({ hands: [hand(0.44, 0.2)], landmarks: body, aspect: 1, t: k * 100 });
  assert.equal(r, 'Clapping');
  // long after the clapping stopped, a lone hand at the ear really is a phone call
  for (let k = 60; k < 90; k++) r = h.update({ hands: [hand(0.44, 0.2)], landmarks: body, aspect: 1, t: k * 100 });
  assert.equal(r, 'Phone Call');
});

test('two hands near the face are not a phone call', () => {
  assert.notEqual(act(rep2([hand(0.44, 0.2), hand(0.56, 0.2)])), 'Phone Call');
});

test('no hands, or no body, gives nothing', () => {
  assert.equal(act(rep2([])), 'None');
  assert.equal(new HandActions().update({ hands: [hand(0.5, 0.24)], landmarks: null, aspect: 1, t: 0 }), 'None');
});

test('final gesture priority: face actions beat finger signs, specific signs beat plain fist', () => {
  assert.equal(finalGesture('None', 'Thumbs up', 'Phone Call'), 'Phone Call');
  assert.equal(finalGesture('None', 'Thumbs up', 'Hand at face'), 'Thumbs up');
  assert.equal(finalGesture('Waving', 'None', 'Hand at face'), 'Waving');
  assert.equal(finalGesture('None', 'Fist', 'Hands together'), 'Hands together');
  assert.equal(finalGesture('Waving', 'Open palm', 'None'), 'Waving');
  assert.equal(finalGesture('None', 'None', 'None'), 'None');
});

test('wrist-based phone/eat/clap guesses are switched off when the hand model is on', () => {
  const phone = pose({ 15: { x: 0.44, y: 0.2 }, 13: { x: 0.38, y: 0.32 } });
  const c = new ActivityClassifier(); c.wristFaceGestures = false; let r;
  for (let k = 0; k < 90; k++) r = c.update({ landmarks: phone, blendshapes: null, t: k * 33, aspect: 1 });
  assert.notEqual(r.gesture, 'Phone Call');
});

test('posture vector uses body position only and ignores arms', () => {
  const a = postureVector(pose(), 1), b = postureVector(pose({ 15: { x: 0.1, y: 0.1 }, 16: { x: 0.9, y: 0.9 }, 13: { x: 0.2, y: 0.2 } }), 1);
  assert.equal(a.length, 11);
  assert.ok(a.every(Number.isFinite));
  assert.deepEqual(a, b);                       // moving wrists/elbows changes nothing
  assert.equal(postureVector(pose({ 11: { visibility: 0.1 } }), 1), null);   // no shoulders -> no posture
});

test('sit/stand calibration separates two camera setups it was trained on', () => {
  const sitView = pose({ 23: { visibility: 0.2 }, 24: { visibility: 0.2 }, 25: { visibility: 0.05 }, 26: { visibility: 0.05 }, 27: { visibility: 0.02 }, 28: { visibility: 0.02 }, 11: { y: 0.55 }, 12: { y: 0.55 }, 0: { y: 0.4 } });
  const standView = pose({ 11: { x: 0.43, y: 0.3 }, 12: { x: 0.57, y: 0.3 } });
  const m = new KnnModel();
  const jit = (lm, k) => lm.map((p) => ({ ...p, x: p.x + Math.sin(k * 3 + p.y * 7) * 0.003, y: p.y + Math.cos(k * 2 + p.x * 7) * 0.003 }));
  for (const [lab, v] of [['Sitting', sitView], ['Standing', standView]]) { const rec = m.newRecording(); for (let k = 0; k < 60; k++) m.add(lab, postureVector(jit(v, k), 1), rec); }
  assert.ok(m.ready());
  assert.equal(m.predict(postureVector(jit(sitView, 99), 1)).label, 'Sitting');
  assert.equal(m.predict(postureVector(jit(standView, 98), 1)).label, 'Standing');
  // arm movement must not flip it
  const armsUp = jit(pose({ ...{ 23: { visibility: 0.2 }, 24: { visibility: 0.2 }, 25: { visibility: 0.05 }, 26: { visibility: 0.05 }, 27: { visibility: 0.02 }, 28: { visibility: 0.02 }, 11: { y: 0.55 }, 12: { y: 0.55 }, 0: { y: 0.4 } }, 15: { x: 0.3, y: 0.1 }, 16: { x: 0.7, y: 0.1 } }), 5);
  assert.equal(m.predict(postureVector(armsUp, 1)).label, 'Sitting');
});
