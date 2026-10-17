import { PointViewer, instanceColor } from './viewer.mjs';

const $ = id => document.getElementById(id);
const state = { models: [], scene: null, clicks: [], result: null, busy: false, foreground: true, objectId: 1, objectCount: 1, stale: false };
const number = value => new Intl.NumberFormat().format(value);
const selectedModel = () => state.models.find(model => model.id === $('model').value);
const isInteractive = model => ['interactive', 'agile3d'].includes(model?.task);
let viewer;

function status(title, detail, kind = '') {
  $('status-title').textContent = title; $('status-detail').textContent = detail; $('status-dot').className = `status-dot ${kind}`;
}
function errorMessage(error) { return error instanceof Error ? error.message : String(error); }
async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, { ...options, headers: { ...options.headers } }); }
  catch { throw new Error('The local server could not be reached. Check that it is running, then try again.'); }
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); }
  catch { throw new Error(response.ok ? 'The server returned an unreadable response.' : `The request failed (${response.status}). Check the server log for details.`); }
  if (!response.ok) throw new Error(typeof data.error === 'string' ? data.error : data.error?.message || data.detail || `The request failed (${response.status}).`);
  return data;
}
function setBusy(value, message = '') {
  state.busy = value; $('busy').hidden = !value; $('busy-text').textContent = message;
  $('drop-zone').classList.toggle('disabled', value);
  updateControls();
}
function updateControls() {
  const model = selectedModel(), interactive = isInteractive(model), available = Boolean(model?.available), hasScene = Boolean(state.scene);
  const foregroundCount = state.clicks.filter(click => click.object_id > 0).length;
  $('model').disabled = state.busy || !state.models.length;
  $('refresh-models').disabled = state.busy;
  $('file').disabled = state.busy;
  $('empty-upload').disabled = state.busy;
  $('load-example').disabled = state.busy;
  $('example').disabled = state.busy;
  $('automatic-controls').hidden = interactive;
  $('interactive-controls').hidden = !interactive;
  $('threshold').disabled = state.busy || !available;
  $('object').disabled = state.busy;
  $('new-object').disabled = state.busy || state.objectCount >= 100;
  $('positive').disabled = state.busy;
  $('negative').disabled = state.busy;
  $('undo').disabled = state.busy || !state.clicks.length;
  $('reset').disabled = state.busy || !state.clicks.length;
  $('run').disabled = state.busy || !available || !hasScene || !viewer || (interactive && !foregroundCount);
  $('run-text').textContent = state.busy ? 'Working…' : interactive ? (state.result ? 'Update segmentation' : 'Segment with prompts') : 'Run segmentation';
  $('run-hint').textContent = !viewer ? 'A working 3D viewer is required.' : !model ? 'Register a trained model to enable inference.' : !available ? 'This model is not available. See the details above.' : !hasScene ? 'Add a scene to get started.' : interactive && !foregroundCount ? 'Add at least one object prompt to run.' : state.stale ? 'Changes are ready. Run to update the masks.' : interactive ? 'Add prompts, then run to see the masks.' : 'Inference runs on the local server.';
  $('click-count').textContent = `${state.clicks.length} prompt${state.clicks.length === 1 ? '' : 's'}`;
  const canPrompt = interactive && hasScene && !state.busy;
  if (viewer) viewer.interactive = canPrompt;
  $('viewport').classList.toggle('prompting', canPrompt);
  $('click-mode-indicator').hidden = !canPrompt;
  $('click-mode-text').textContent = state.foreground ? `Click to prompt Object ${state.objectId}` : 'Click to mark background';
  $('click-mode-dot').style.background = state.foreground ? `rgb(${instanceColor(state.objectId).join(',')})` : '#ea6268';
  $('positive').classList.toggle('active', state.foreground); $('negative').classList.toggle('active', !state.foreground);
  $('positive').setAttribute('aria-pressed', String(state.foreground)); $('negative').setAttribute('aria-pressed', String(!state.foreground));
  $('task-badge').textContent = !model ? 'Point cloud' : interactive ? 'Interactive segmentation' : 'Instance segmentation';
}
function disableDownload() {
  $('download').removeAttribute('href'); $('download').classList.add('disabled'); $('download').setAttribute('aria-disabled', 'true');
}
function clearPrediction() {
  state.result = null; state.stale = false;
  viewer?.setLabels(null); setColorMode('rgb');
  $('instance-view').disabled = true; $('object-count').textContent = '—'; $('object-count').title = '';
  $('provenance').hidden = true; disableDownload();
}
function resetPrompts() {
  state.clicks = []; state.objectId = 1; state.objectCount = 1;
  $('object').replaceChildren(new Option('Object 1', '1')); viewer?.setClicks([]);
}
function showModel(reset = true) {
  const model = selectedModel();
  $('model-kind').textContent = model ? (isInteractive(model) ? 'Click-guided' : 'Automatic') : 'No models';
  $('model-state').textContent = model ? (model.available ? 'Available' : 'Unavailable') : 'Setup needed';
  $('model-state').className = `availability ${model?.available ? 'ready' : 'unavailable'}`;
  $('model-description').textContent = model?.description || (model ? model.name : 'No trained models are registered on this server.');
  $('model-reason').textContent = model?.reason || (model && !model.available ? 'The required checkpoint files are not available on this server.' : '');
  $('model-reason').hidden = !$('model-reason').textContent;
  if (reset) {
    resetPrompts(); clearPrediction();
    if (state.scene) status('Scene ready', model?.available ? 'Explore the scene or run the selected model.' : 'You can explore the scene while a model is unavailable.');
  }
  updateControls();
}
async function loadModels() {
  $('refresh-models').disabled = true;
  const oldId = $('model').value;
  try {
    const data = await api('/api/models');
    state.models = Array.isArray(data.models) ? data.models : [];
    $('model').replaceChildren();
    for (const [task, label] of [['automatic', 'Automatic instance segmentation'], ['interactive', 'Interactive segmentation']]) {
      const models = state.models.filter(model => task === 'interactive' ? isInteractive(model) : !isInteractive(model));
      if (!models.length) continue;
      const group = document.createElement('optgroup'); group.label = label;
      for (const model of models) group.append(new Option(`${model.name}${model.available ? '' : ' · unavailable'}`, model.id));
      $('model').append(group);
    }
    if (!state.models.length) $('model').append(new Option('No trained models registered', ''));
    const oldModel = state.models.find(model => model.id === oldId);
    $('model').value = (oldModel || state.models.find(model => model.available) || state.models[0])?.id || '';
    showModel(oldId !== $('model').value);
  } catch (error) {
    $('model-state').textContent = 'Offline'; $('model-state').className = 'availability unavailable';
    $('model-reason').textContent = errorMessage(error); $('model-reason').hidden = false;
    status('Connection failed', errorMessage(error), 'error');
  } finally { updateControls(); }
}
async function loadExamples() {
  try {
    const data = await api('/api/examples');
    if (!Array.isArray(data.examples) || !data.examples.length) return;
    $('example').replaceChildren(...data.examples.map(item => new Option(`${item.name}${item.point_count ? ` · ${number(item.point_count)} pts` : ''}`, item.id)));
    $('examples-controls').hidden = false;
  } catch { /* Prepared scenes are optional; file upload remains available. */ }
}
function acceptScene(scene) {
  if (!Array.isArray(scene.points) || !scene.points.length || !Array.isArray(scene.point_ids) || scene.point_ids.length !== scene.points.length) throw new Error('The server returned an incomplete scene. Display points and original point indices must match.');
  viewer.setScene(scene);
  state.scene = scene; resetPrompts(); clearPrediction();
  $('empty-state').hidden = true; $('viewer-toolbar').hidden = false;
  $('scene-name').textContent = scene.name; $('upload-label').textContent = scene.name;
  $('upload-hint').textContent = 'Choose another file to replace this scene'; $('drop-zone').classList.add('loaded');
  $('point-count').textContent = number(scene.point_count);
  $('display-count').textContent = `${number(scene.points.length)} shown${scene.point_count > scene.points.length ? ' · sampled' : ''}`;
  status('Scene ready', `${number(scene.point_count)} points loaded. ${isInteractive(selectedModel()) ? 'Select a visible point to add an object prompt.' : 'Run the selected model to predict instance masks.'}`, 'success');
  updateControls();
}
async function loadScene(load, name) {
  if (state.busy || !viewer) return;
  setBusy(true, 'Loading point cloud…'); status('Loading scene', name, 'busy');
  try { acceptScene(await load()); }
  catch (error) { status('Could not load scene', errorMessage(error), 'error'); }
  finally { setBusy(false); $('file').value = ''; }
}
function upload(file) {
  if (!file || state.busy) return;
  if (!/\.(npz|ply)$/i.test(file.name)) { status('Unsupported file', 'Choose an NPZ or PLY point cloud.', 'error'); return; }
  loadScene(() => api('/api/scenes', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name) }, body: file }), file.name);
}
function promptsChanged() {
  viewer?.setClicks(state.clicks); state.stale = Boolean(state.result);
  if (state.result) disableDownload();
  status(state.result ? 'Prompts changed' : 'Prompts ready', `${state.clicks.length} prompt${state.clicks.length === 1 ? '' : 's'}. Run the model to ${state.result ? 'update' : 'predict'} the masks.`, 'pending');
  updateControls();
}
function onPick(index, temporaryBackground) {
  if (state.busy || !state.scene || !isInteractive(selectedModel())) return;
  if (index < 0) { status('No point selected', 'Click closer to a visible point, or zoom in to refine your selection.'); return; }
  if (state.clicks.length >= 200) { status('Prompt limit reached', 'A prediction supports up to 200 prompts. Undo or reset prompts before adding more.', 'error'); return; }
  state.clicks.push({ point_index: state.scene.point_ids[index], object_id: state.foreground && !temporaryBackground ? state.objectId : 0, display_index: index });
  promptsChanged();
}
function setColorMode(mode) {
  viewer?.setMode(mode);
  for (const name of ['rgb', 'instance']) { const active = mode === (name === 'rgb' ? 'rgb' : 'instances'); $(name + '-view').classList.toggle('active', active); $(name + '-view').setAttribute('aria-pressed', String(active)); }
}
async function predict() {
  if ($('run').disabled) return;
  const model = selectedModel();
  setBusy(true, 'Predicting instance masks…'); status('Running inference', 'The first prediction may take longer while the model loads.', 'busy');
  disableDownload();
  const start = performance.now();
  try {
    const result = await api('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: model.id, scene_id: state.scene.scene_id, clicks: state.clicks.map(({ point_index, object_id }) => ({ point_index, object_id })), score_threshold: Number($('threshold').value) }) });
    if (!Array.isArray(result.labels) || result.labels.length !== state.scene.points.length || !result.labels.every(value => Number.isInteger(value) && value >= 0)) throw new Error('The model returned labels that do not match the displayed scene.');
    state.result = result; state.stale = false; viewer.setLabels(result.labels); setColorMode('instances');
    $('instance-view').disabled = false;
    const shownObjects = new Set(result.labels.filter(label => label > 0)).size;
    const count = result.instance_count ?? shownObjects;
    $('object-count').textContent = number(count);
    $('object-count').title = result.instance_count == null ? 'Distinct objects in the displayed points' : 'Predicted objects in the full scene';
    if (result.download_url) {
      const url = new URL(result.download_url, window.location.href);
      if (url.origin !== window.location.origin) throw new Error('The server returned a download outside this local session.');
      $('download').href = url.href; $('download').classList.remove('disabled'); $('download').setAttribute('aria-disabled', 'false');
    }
    $('provenance-text').textContent = JSON.stringify(result.provenance || { model: model.name }, null, 2); $('provenance').hidden = false;
    const seconds = ((performance.now() - start) / 1000).toFixed(1);
    status('Prediction complete', `${count ? `${number(count)} object${count === 1 ? '' : 's'}${result.instance_count == null ? ' in the displayed points' : ''}` : 'No objects found at this score threshold'} · ${seconds} s. ${result.download_url ? 'Full-resolution predictions are ready to download.' : ''}`, 'success');
  } catch (error) {
    state.stale = Boolean(state.result);
    status('Prediction failed', errorMessage(error), 'error');
  } finally { setBusy(false); }
}

$('model').addEventListener('change', () => showModel());
$('refresh-models').addEventListener('click', loadModels);
$('file').addEventListener('change', e => upload(e.target.files[0]));
$('empty-upload').addEventListener('click', () => $('file').click());
$('load-example').addEventListener('click', () => { const id = $('example').value; if (id) loadScene(() => api(`/api/examples/${encodeURIComponent(id)}`), $('example').selectedOptions[0].textContent); });
for (const event of ['dragenter', 'dragover']) $('drop-zone').addEventListener(event, e => { e.preventDefault(); if (!state.busy) $('drop-zone').classList.add('dragover'); });
for (const event of ['dragleave', 'drop']) $('drop-zone').addEventListener(event, e => { e.preventDefault(); $('drop-zone').classList.remove('dragover'); });
$('drop-zone').addEventListener('drop', e => upload(e.dataTransfer.files[0]));
// Dropping outside the upload area must not navigate away from a running session.
window.addEventListener('dragover', e => e.preventDefault()); window.addEventListener('drop', e => e.preventDefault());
$('threshold').addEventListener('input', () => { $('threshold-value').textContent = Number($('threshold').value).toFixed(2); if (state.result) { state.stale = true; disableDownload(); status('Score changed', 'Run segmentation to apply the new minimum mask score.', 'pending'); } updateControls(); });
$('positive').addEventListener('click', () => { state.foreground = true; updateControls(); });
$('negative').addEventListener('click', () => { state.foreground = false; updateControls(); });
$('object').addEventListener('change', () => { state.objectId = Number($('object').value); state.foreground = true; updateControls(); });
$('new-object').addEventListener('click', () => { state.objectId = ++state.objectCount; $('object').append(new Option(`Object ${state.objectId}`, String(state.objectId))); $('object').value = String(state.objectId); state.foreground = true; updateControls(); });
$('undo').addEventListener('click', () => { state.clicks.pop(); promptsChanged(); });
$('reset').addEventListener('click', () => { resetPrompts(); promptsChanged(); });
$('run').addEventListener('click', predict);
$('rgb-view').addEventListener('click', () => setColorMode('rgb')); $('instance-view').addEventListener('click', () => setColorMode('instances'));
$('fit').addEventListener('click', () => viewer?.fit()); $('point-size').addEventListener('input', () => viewer?.setPointSize(Number($('point-size').value)));
$('help-toggle').addEventListener('click', () => { const open = $('help').hidden; $('help').hidden = !open; $('help-toggle').setAttribute('aria-expanded', String(open)); if (open) $('help').scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'nearest' }); });
$('download').addEventListener('click', e => { if ($('download').getAttribute('aria-disabled') === 'true') e.preventDefault(); });

try { viewer = new PointViewer($('canvas'), $('markers'), onPick, message => { viewer = null; updateControls(); status('Viewer unavailable', message, 'error'); }); }
catch (error) { status('Viewer unavailable', errorMessage(error), 'error'); $('empty-state').querySelector('p').textContent = errorMessage(error); }
updateControls();
await Promise.all([loadModels(), loadExamples()]);
