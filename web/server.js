// Node server: serves the UI, MediaPipe assets, and proxies /api/* to the Python sensor-model API.
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { Readable } from 'node:stream';
import { fileURLToPath } from 'node:url';
import { parsePhoneUrl } from './lib/phone-url.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const PORT = Number(process.env.PORT || 3000);
// Listen on this computer only by default; set HOST=0.0.0.0 to open the dashboard to your network.
const HOST = process.env.HOST || '127.0.0.1';
const API_PORT = Number(process.env.API_PORT || 8000);
const API_URL = process.env.API_URL || `http://127.0.0.1:${API_PORT}`;

const app = express();
app.use(express.json({ limit: '5mb' }));
app.use('/vendor/tasks-vision', express.static(path.join(here, 'node_modules', '@mediapipe', 'tasks-vision')));
app.use(express.static(path.join(here, 'public')));

// Relay a phone camera's MJPEG stream (DroidCam / IP Webcam) so the page can read it same-origin.
// Only local-network addresses are accepted (see lib/phone-url.js).
app.get('/phone-stream', async (req, res) => {
  const url = parsePhoneUrl(req.query.url);
  if (!url) {
    return res.status(400).json({ error: 'Use an http(s) address on your local network, e.g. http://192.168.1.100:4747/video' });
  }
  const ctl = new AbortController();
  res.on('close', () => ctl.abort());
  const timer = setTimeout(() => ctl.abort(), 8000); // only limits the time to first response
  try {
    const up = await fetch(url, { signal: ctl.signal, redirect: 'manual' });
    clearTimeout(timer);
    if (up.status >= 300 && up.status < 400) return res.status(502).json({ error: 'The phone camera redirected the request. Use its direct video URL.' });
    if (!up.ok || !up.body) return res.status(502).json({ error: `The phone camera answered HTTP ${up.status}.` });
    res.status(200).set({ 'content-type': up.headers.get('content-type') || 'multipart/x-mixed-replace', 'cache-control': 'no-store' });
    Readable.fromWeb(up.body).on('error', () => res.end()).pipe(res);
  } catch {
    clearTimeout(timer);
    if (!res.headersSent) res.status(502).json({ error: 'Could not reach the phone camera. Check the address, that the phone is on the same Wi-Fi, and that its camera app is running.' });
    else res.end();
  }
});

app.use('/api', async (req, res) => {
  try {
    const upstream = await fetch(API_URL + req.originalUrl, {
      method: req.method,
      headers: { 'content-type': 'application/json' },
      body: ['GET', 'HEAD'].includes(req.method) ? undefined : JSON.stringify(req.body ?? {}),
      signal: AbortSignal.timeout(15000),
    });
    res.status(upstream.status).type('json').send(await upstream.text());
  } catch {
    res.status(503).json({ error: 'Sensor model API is offline. Start it with: python -m uvicorn api:app --port ' + API_PORT });
  }
});

// Start the Python API alongside the UI unless START_API=0.
let api = null;
if (process.env.START_API !== '0') {
  const venvPy = process.platform === 'win32'
    ? path.join(root, '.venv', 'Scripts', 'python.exe')
    : path.join(root, '.venv', 'bin', 'python');
  const py = fs.existsSync(venvPy) ? venvPy : 'python';
  api = spawn(py, ['-m', 'uvicorn', 'api:app', '--host', '127.0.0.1', '--port', String(API_PORT)], { cwd: root, stdio: 'inherit' });
  api.on('error', (e) => console.warn('Could not start Python API:', e.message));
  api.on('exit', (code) => console.warn(`Python API exited (${code}). Sensor tab will be unavailable.`));
}
const stop = () => { api?.kill(); process.exit(0); };
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
process.on('exit', () => api?.kill());

app.listen(PORT, HOST, () => console.log(`HAR UI on http://localhost:${PORT}` + (HOST === '127.0.0.1' ? '' : ` (listening on ${HOST})`)));
