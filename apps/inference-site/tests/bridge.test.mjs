import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../worker/index.js';
class Bucket {
  records = new Map(); sequence = 0;
  async get(key) { const value = this.records.get(key); if (!value) return null; return { ...value, body: new Blob([value.bytes]).stream(), json: async () => JSON.parse(new TextDecoder().decode(value.bytes)) }; }
  async put(key, value, options = {}) {
    const bytes = new Uint8Array(await new Response(value).arrayBuffer());
    const old = this.records.get(key), condition = options.onlyIf;
    if (condition?.etagMatches && old?.etag !== condition.etagMatches) return null;
    if (condition?.etagDoesNotMatch === '*' && old) return null;
    const record = { bytes, size: bytes.length, etag: String(++this.sequence), customMetadata: options.customMetadata || {} }; this.records.set(key, record); return record;
  }
  async delete(key) { for (const item of Array.isArray(key) ? key : [key]) this.records.delete(item); }
  async list({ prefix }) { return { objects: [...this.records].filter(([key]) => key.startsWith(prefix)).map(([key, value]) => ({ key, ...value })), truncated: false }; }
}
const deadline = new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString();
const model = { id: 'mask3d-unsamv2-e600', name: 'Pretrained', available: true, task: 'mask3d' };
function setup() {
  const env = { BUCKET: new Bucket(), DELIMIT3D_BRIDGE_TOKEN: 'test-secret' };
  const req = async (path, body, options = {}) => {
    const headers = { ...(path.startsWith('/bridge/') ? { authorization: 'Bearer test-secret', 'x-worker-id': 'worker-a' } : {}), ...options.headers };
    const value = body == null ? undefined : typeof body === 'object' ? JSON.stringify(body) : body;
    if (value !== undefined) headers['content-length'] ??= String(Buffer.byteLength(value));
    const result = await worker.fetch(new Request(`https://example.test${path}`, { method: value === undefined ? 'GET' : 'POST', body: value, ...options, headers }), env);
    return result;
  };
  const beat = async (extra = {}) => req('/bridge/heartbeat', { worker_id: 'worker-a', models: [model], examples: [{ id: '0', name: 'Scene' }], active_job_id: null, available_until: deadline, ...extra });
  return { env, req, beat };
}
test('private worker routes require a separate bridge secret', async () => {
  const { req } = setup(); assert.equal((await req('/bridge/job?worker_id=worker-a', null, { headers: { authorization: 'Bearer wrong' } })).status, 401);
});
test('offline models are honest and mutation is refused', async () => {
  const { req } = setup(); assert.equal((await (await req('/api/models')).json()).online, false);
  assert.equal((await req('/api/scenes', 'bytes', { headers: { 'x-filename': 'scene.npz' } })).status, 503);
});
test('heartbeat lease prevents two GPU owners and exposes availability', async () => {
  const { req, beat } = setup(); assert.equal((await beat()).status, 200); assert.equal((await beat({ worker_id: 'worker-b' })).status, 409);
  const data = await (await req('/api/status')).json(); assert.equal(data.online, true); assert.equal(data.available_until, deadline);
});
test('example goes through atomic claim, completion and idempotent acknowledgement', async () => {
  const { req, beat } = setup(); await beat(); const queued = await (await req('/api/examples/0')).json();
  assert.equal((await req(queued.status_url)).status, 202);
  const { job } = await (await req('/bridge/job?worker_id=worker-a')).json(); assert.equal(job.kind, 'example'); assert.equal(job.example_id, '0');
  const repeated = await (await req('/bridge/job?worker_id=worker-a')).json(); assert.equal(repeated.job.id, job.id);
  const result = { scene_id: 'scene-a', point_count: 1, points: [[0, 0, 0]], point_ids: [0] };
  assert.equal((await req(`/bridge/complete/${job.id}`, { worker_id: 'worker-a', ok: true, result })).status, 200);
  assert.equal((await req(`/bridge/complete/${job.id}`, { worker_id: 'worker-a', ok: true, result })).status, 200);
  assert.deepEqual(await (await req(queued.status_url)).json(), result);
  assert.equal((await (await req('/bridge/job?worker_id=worker-a')).json()).job, null);
});
test('scene upload is streamed and deleted after completion', async () => {
  const { req, beat } = setup(); await beat(); const queued = await req('/api/scenes', 'scene-bytes', { headers: { 'x-filename': 'my%20scene.npz' } }); assert.equal(queued.status, 202);
  const { job } = await (await req('/bridge/job?worker_id=worker-a')).json(); assert.equal(job.filename, 'my scene.npz');
  assert.equal(await (await req(`/bridge/blob/${job.blob_id}`)).text(), 'scene-bytes');
  await req(`/bridge/complete/${job.id}`, { worker_id: 'worker-a', ok: true, result: { scene_id: 'scene-a' } });
  assert.equal((await req(`/bridge/blob/${job.blob_id}`)).status, 410);
});
test('prediction inputs are validated and failures reach the browser', async () => {
  const { req, beat } = setup(); await beat(); const input = { model_id: model.id, scene_id: 'scene-a', clicks: [{ point_index: 2, object_id: 1 }], score_threshold: 0.5 };
  assert.equal((await req('/api/predict', { ...input, score_threshold: 3 })).status, 400);
  const queued = await (await req('/api/predict', input)).json(); const { job } = await (await req('/bridge/job?worker_id=worker-a')).json(); assert.deepEqual(job.request, input);
  await req(`/bridge/complete/${job.id}`, { worker_id: 'worker-a', ok: false, error: 'Scene expired; load it again.' });
  const result = await req(queued.status_url); assert.equal(result.status, 422); assert.match((await result.json()).error, /Scene expired/);
});
test('results are downloadable and repeat upload keeps the original artifact', async () => {
  const { req, beat } = setup(); await beat(); await req('/bridge/result/result-a.npz', 'prediction'); await req('/bridge/result/result-a.npz', 'different');
  const result = await req('/api/results/result-a.npz'); assert.equal(await result.text(), 'prediction'); assert.match(result.headers.get('content-disposition'), /attachment/);
});
test('cross-origin uploads and unsupported or oversized files are refused', async () => {
  const { req, beat } = setup(); await beat();
  assert.equal((await req('/api/scenes', 'x', { headers: { origin: 'https://other.test', 'x-filename': 'a.npz' } })).status, 403);
  assert.equal((await req('/api/scenes', 'x', { headers: { 'x-filename': '../a.npz' } })).status, 400);
  assert.equal((await req('/api/scenes', 'x', { headers: { 'x-filename': 'a.npz', 'content-length': String(128 * 1024 * 1024 + 1) } })).status, 413);
});
test('concurrent enqueue is bounded and does not overwrite other jobs', async () => {
  const { req, beat } = setup(); await beat(); const results = await Promise.all(Array.from({ length: 12 }, () => req('/api/examples/0')));
  assert.equal(results.filter(result => result.status === 202).length, 8); assert.equal(results.filter(result => result.status === 429).length, 4);
});
test('expired worker and requests cannot appear ready or accept late completion', async () => {
  const { req, beat, env } = setup(); await beat(); const queued = await (await req('/api/examples/0')).json(); const { job } = await (await req('/bridge/job?worker_id=worker-a')).json();
  const originalNow = Date.now; Date.now = () => originalNow() + 31 * 60 * 1000;
  try {
    assert.equal((await (await req('/api/models')).json()).models[0].available, false);
    assert.equal((await req(queued.status_url)).status, 408);
    await beat(); assert.equal((await req(`/bridge/complete/${job.id}`, { worker_id: 'worker-a', ok: true, result: {} })).status, 410);
  } finally { Date.now = originalNow; }
});
test('new worker marks old scene jobs failed instead of silently replaying them', async () => {
  const { req, beat } = setup(); await beat(); const queued = await (await req('/api/examples/0')).json();
  const originalNow = Date.now; Date.now = () => originalNow() + 181000;
  try { await beat({ worker_id: 'worker-b' }); await req('/bridge/job?worker_id=worker-b'); const response = await req(queued.status_url); assert.equal(response.status, 422); assert.match((await response.json()).error, /restarted/); }
  finally { Date.now = originalNow; }
});
test('artifact serves embedded local UI with security headers', async () => {
  const { req } = setup(); const index = await req('/'); assert.equal(index.status, 200); assert.match(await index.text(), /Private GPU session/); assert.match(index.headers.get('content-security-policy'), /default-src 'self'/);
  const app = await req('/static/app.mjs'); assert.match(await app.text(), /status_url/);
});

test('a missing scene upload fails its job and does not trap the GPU poll loop', async () => {
  const { req, beat, env } = setup(); await beat(); const queued = await (await req('/api/scenes', 'x', { headers: { 'x-filename': 'a.npz' } })).json();
  const { job } = await (await req('/bridge/job?worker_id=worker-a')).json(); await env.BUCKET.delete(`delimit3d/v1/blobs/${job.blob_id}`);
  assert.equal((await req(`/bridge/blob/${job.blob_id}`)).status, 410); assert.equal((await req(queued.status_url)).status, 422);
  assert.equal((await (await req('/bridge/job?worker_id=worker-a')).json()).job, null);
});
test('GPU availability ends at the allocation deadline even with a fresh heartbeat', async () => {
  const { req, beat } = setup(); await beat({ available_until: new Date(Date.now() - 1000).toISOString() });
  assert.equal((await (await req('/api/status')).json()).online, false); assert.equal((await (await req('/api/models')).json()).models[0].available, false);
});
test('heartbeat schedules bounded cleanup of expired artifacts', async () => {
  const { env } = setup(); await env.BUCKET.put('delimit3d/v1/results/expired.npz', 'old', { customMetadata: { created_at: String(Date.now() - 25 * 60 * 60 * 1000) } });
  await env.BUCKET.put('delimit3d/v1/results/current.npz', 'new', { customMetadata: { created_at: String(Date.now()) } });
  const pending = []; const result = await worker.fetch(new Request('https://example.test/bridge/heartbeat', { method: 'POST', headers: { authorization: 'Bearer test-secret' }, body: JSON.stringify({ worker_id: 'worker-a', models: [], examples: [] }) }), env, { waitUntil: promise => pending.push(promise) });
  assert.equal(result.status, 200); await Promise.all(pending);
  assert.equal(await env.BUCKET.get('delimit3d/v1/results/expired.npz'), null); assert.ok(await env.BUCKET.get('delimit3d/v1/results/current.npz'));
});
