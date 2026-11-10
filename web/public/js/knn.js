/**
 * k-nearest-neighbour classifier trained from the user's own recordings.
 * - features are standardised (z-score) so no single feature dominates
 * - votes are class-balanced so a label with more samples does not win by default
 * - crossValidate() reports accuracy on held-out *time chunks* of each recording
 *   (random splits would look great because neighbouring frames are near-duplicates)
 */
export class KnnModel {
  constructor() { this.samples = []; this.recSeq = 0; this._z = null; }

  newRecording() { return ++this.recSeq; }
  add(label, vec, rec = 0) { this.samples.push({ label, vec, rec }); this._z = null; }
  clear() { this.samples = []; this._z = null; }
  clearLabel(label) { this.samples = this.samples.filter((s) => s.label !== label); this._z = null; }

  counts() {
    const c = {};
    for (const s of this.samples) c[s.label] = (c[s.label] || 0) + 1;
    return c;
  }

  /** Needs at least two labels with a handful of samples each. */
  ready() {
    const c = Object.values(this.counts());
    return c.length >= 2 && c.every((n) => n >= 10);
  }

  _fit() {
    if (this._z) return;
    const dim = Math.max(...this.samples.map((s) => s.vec.length));
    const mean = new Array(dim).fill(0), std = new Array(dim).fill(0);
    for (const s of this.samples) for (let i = 0; i < dim; i++) mean[i] += (s.vec[i] ?? 0);
    for (let i = 0; i < dim; i++) mean[i] /= this.samples.length;
    for (const s of this.samples) for (let i = 0; i < dim; i++) std[i] += ((s.vec[i] ?? 0) - mean[i]) ** 2;
    // floor keeps near-constant features from exploding into noise
    for (let i = 0; i < dim; i++) std[i] = Math.max(Math.sqrt(std[i] / this.samples.length), 0.05);
    this._mean = mean; this._std = std;
    this._z = this.samples.map((s) => this._norm(s.vec));
    this._counts = this.counts();
  }

  _norm(vec) { return this._mean.map((m, i) => ((vec[i] ?? 0) - m) / this._std[i]); }

  _vote(zq, ok, k) {
    const near = [];
    for (let i = 0; i < this._z.length; i++) {
      if (!ok(i)) continue;
      let sum = 0; const z = this._z[i];
      for (let d = 0; d < z.length; d++) { const diff = z[d] - zq[d]; sum += diff * diff; }
      near.push({ label: this.samples[i].label, d: Math.sqrt(sum) });
    }
    near.sort((a, b) => a.d - b.d);
    const votes = new Map(); let total = 0;
    for (const nb of near.slice(0, k)) {
      const w = 1 / (nb.d + 0.5) / Math.sqrt(this._counts[nb.label]);
      votes.set(nb.label, (votes.get(nb.label) || 0) + w); total += w;
    }
    const ranked = [...votes.entries()].sort((a, b) => b[1] - a[1]);
    if (!ranked.length) return null;
    return {
      label: ranked[0][0], confidence: ranked[0][1] / total,
      runnerUp: ranked[1]?.[0] ?? null,
    };
  }

  predict(vec, k = 9) {
    if (!vec || !this.samples.length) return null;
    this._fit();
    return this._vote(this._norm(vec), () => true, k);
  }

  /** Leave-one-time-chunk-out accuracy. Returns null if there is too little data. */
  crossValidate(k = 9, chunks = 3, gap = 8) {
    if (!this.ready()) return null;
    this._fit();
    const meta = new Array(this.samples.length);
    const byRec = new Map();
    this.samples.forEach((s, i) => { (byRec.get(s.rec) || byRec.set(s.rec, []).get(s.rec)).push(i); });
    for (const [rec, idxs] of byRec) {
      const n = idxs.length;
      idxs.forEach((si, pos) => {
        const chunk = Math.min(Math.floor((pos * chunks) / n), chunks - 1);
        meta[si] = { rec, pos, chunk, lo: Math.floor((chunk * n) / chunks), hi: Math.floor(((chunk + 1) * n) / chunks) };
      });
    }
    const per = {}; let right = 0, total = 0;
    for (let i = 0; i < this.samples.length; i++) {
      const m = meta[i];
      const ok = (j) => {
        const t = meta[j];
        if (t.rec !== m.rec) return true;
        return t.pos < m.lo - gap || t.pos >= m.hi + gap; // skip the held-out chunk and its borders
      };
      const r = this._vote(this._z[i], ok, k);
      const label = this.samples[i].label;
      const p = per[label] || (per[label] = { right: 0, n: 0, confusedWith: {} });
      p.n++; total++;
      if (r && r.label === label) { p.right++; right++; }
      else if (r) p.confusedWith[r.label] = (p.confusedWith[r.label] || 0) + 1;
    }
    const perLabel = {};
    for (const [label, p] of Object.entries(per)) {
      const worst = Object.entries(p.confusedWith).sort((a, b) => b[1] - a[1])[0];
      perLabel[label] = { recall: p.right / p.n, n: p.n, confusedWith: worst ? worst[0] : null };
    }
    return { accuracy: right / total, perLabel, n: total };
  }

  toJSON() { return JSON.stringify({ seq: this.recSeq, s: this.samples.map((s) => ({ l: s.label, r: s.rec, v: s.vec.map((x) => +x.toFixed(4)) })) }); }
  fromJSON(text) {
    try {
      const o = JSON.parse(text);
      this.recSeq = o.seq || 0;
      this.samples = o.s.map((x) => ({ label: x.l, rec: x.r, vec: x.v }));
    } catch { this.samples = []; }
    this._z = null;
  }
}
