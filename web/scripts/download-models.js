// Downloads the MediaPipe model files into public/models (skipped if already present).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models');
const base = 'https://storage.googleapis.com/mediapipe-models';
const files = {
  'pose_landmarker_full.task': `${base}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`,
  'gesture_recognizer.task': `${base}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task`,
  'face_landmarker.task': `${base}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
};

fs.mkdirSync(dir, { recursive: true });
for (const [name, url] of Object.entries(files)) {
  const dest = path.join(dir, name);
  if (fs.existsSync(dest) && fs.statSync(dest).size > 1e6) continue;
  try {
    console.log(`Downloading ${name} ...`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  } catch (err) {
    // Don't fail `npm install`; the UI reports missing models clearly.
    console.warn(`Could not download ${name}: ${err.message}. Run "node scripts/download-models.js" later.`);
  }
}
