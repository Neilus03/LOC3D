// Small, self-contained WebGL point viewer. No network or rendering dependencies.
export const PALETTE = [
  [237, 145, 102], [104, 185, 174], [151, 155, 230], [225, 190, 104],
  [211, 127, 166], [137, 187, 112], [108, 168, 223], [218, 139, 119],
  [163, 133, 199], [115, 194, 201], [211, 208, 135], [185, 155, 124],
];
export function instanceColor(id) {
  return id > 0 ? PALETTE[(id - 1) % PALETTE.length] : [60, 73, 91];
}
export function normalizePoints(points) {
  if (!Array.isArray(points) || !points.length) throw new Error('The scene contains no display points.');
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    if (!Array.isArray(p) || p.length !== 3 || !p.every(Number.isFinite)) throw new Error('The scene contains invalid coordinates.');
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], p[k]); max[k] = Math.max(max[k], p[k]); }
  }
  const center = min.map((v, k) => (v + max[k]) / 2);
  const scale = Math.max(...max.map((v, k) => v - min[k])) / 2 || 1;
  const positions = new Float32Array(points.length * 3);
  points.forEach((p, i) => p.forEach((v, k) => { positions[i * 3 + k] = (v - center[k]) / scale; }));
  return { positions, center, scale };
}
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function normalized(v) { const n = Math.hypot(...v) || 1; return v.map(x => x / n); }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) {
    for (let k = 0; k < 4; k++) out[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
  }
  return out;
}
export function cameraMatrix(yaw, pitch, distance, target, aspect) {
  const z = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
  const x = normalized(cross([0, 0, 1], z)), y = cross(z, x);
  const eye = z.map((v, k) => target[k] + distance * v);
  const view = [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
  const f = 1 / Math.tan(Math.PI / 8), near = 0.01, far = 100;
  const projection = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0];
  return { matrix: multiply(projection, view), right: x, up: y };
}
export function projectPoint(matrix, x, y, z, width, height) {
  const w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
  if (w <= 0) return null;
  const depth = (matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]) / w;
  if (depth < -1 || depth > 1) return null;
  return {
    x: ((matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) / w + 1) * width / 2,
    y: (1 - (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) / w) * height / 2,
    depth,
  };
}
export function pickPoint(matrix, positions, width, height, x, y, radius = 12, pointSize = 2.5) {
  // Match the circular rendered points. A projected point hidden by a closer
  // point cannot receive a prompt even if its center is closest to the cursor.
  const candidates = [], halfSize = pointSize / 2;
  for (let i = 0; i < positions.length; i += 3) {
    const p = projectPoint(matrix, positions[i], positions[i + 1], positions[i + 2], width, height);
    if (!p || p.x < 0 || p.x > width || p.y < 0 || p.y > height) continue;
    const d2 = (p.x - x) ** 2 + (p.y - y) ** 2;
    if (d2 <= (radius + halfSize) ** 2) candidates.push({ ...p, d2, index: i / 3 });
  }
  candidates.sort((a, b) => a.d2 - b.d2 || a.depth - b.depth);
  for (const p of candidates) {
    if (p.d2 > radius ** 2) continue;
    const hidden = candidates.some(q => q.depth < p.depth - 0.000001 && (q.x - p.x) ** 2 + (q.y - p.y) ** 2 < halfSize ** 2);
    if (!hidden) return p.index;
  }
  return -1;
}

export class PointViewer {
  constructor(canvas, overlay, onPick, onError) {
    this.canvas = canvas; this.overlay = overlay; this.onPick = onPick; this.onError = onError;
    this.positions = new Float32Array(); this.rgb = new Float32Array(); this.labels = null;
    this.clicks = []; this.pointSize = 2.5; this.mode = 'rgb'; this.interactive = false;
    this.target = [0, 0, 0]; this.yaw = -1.1; this.pitch = 0.7; this.distance = 4.1;
    const gl = this.gl = canvas.getContext('webgl', { alpha: true, antialias: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error('This browser could not start WebGL. Enable hardware acceleration or try a browser with WebGL support.');
    const compile = (type, source) => {
      const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error('The 3D renderer could not compile a shader.');
      return shader;
    };
    const program = this.program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, 'attribute vec3 a_position; attribute vec3 a_color; uniform mat4 u_matrix; uniform float u_size; varying vec3 v_color; void main(){ gl_Position = u_matrix * vec4(a_position, 1.0); gl_PointSize = u_size; v_color = a_color; }'));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, 'precision mediump float; varying vec3 v_color; void main(){ vec2 p = gl_PointCoord - vec2(0.5); if (dot(p,p) > 0.25) discard; gl_FragColor = vec4(v_color,1.0); }'));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('The 3D renderer could not initialize.');
    this.positionBuffer = gl.createBuffer(); this.colorBuffer = gl.createBuffer();
    this.positionLocation = gl.getAttribLocation(program, 'a_position'); this.colorLocation = gl.getAttribLocation(program, 'a_color');
    this.matrixLocation = gl.getUniformLocation(program, 'u_matrix'); this.sizeLocation = gl.getUniformLocation(program, 'u_size');
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.clearColor(0, 0, 0, 0);
    canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); this.lost = true; onError('The browser lost its 3D graphics context. Reload the page to restore the viewer.'); });
    this.bindEvents(); this.resizeObserver = new ResizeObserver(() => this.requestDraw()); this.resizeObserver.observe(canvas);
    this.requestDraw();
  }
  setScene(scene) {
    const { positions } = normalizePoints(scene.points);
    this.positions = positions; this.rgb = new Float32Array(positions.length);
    for (let i = 0; i < scene.points.length; i++) {
      const c = scene.colors?.[i] || [166, 180, 193];
      for (let k = 0; k < 3; k++) this.rgb[i * 3 + k] = Math.max(0, Math.min(255, c[k] ?? 166)) / 255;
    }
    this.labels = null; this.clicks = []; this.mode = 'rgb';
    const gl = this.gl; gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer); gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    this.updateColors(); this.fit();
  }
  setLabels(labels) { this.labels = labels; this.updateColors(); }
  setMode(mode) { this.mode = mode; this.updateColors(); }
  setClicks(clicks) { this.clicks = clicks; this.requestDraw(); }
  setPointSize(size) { this.pointSize = size; this.requestDraw(); }
  updateColors() {
    const gl = this.gl;
    let colors = this.rgb;
    if (this.mode === 'instances' && this.labels) {
      colors = new Float32Array(this.positions.length);
      this.labels.forEach((label, i) => { const c = instanceColor(label); for (let k = 0; k < 3; k++) colors[i * 3 + k] = c[k] / 255; });
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer); gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW); this.requestDraw();
  }
  fit() { this.yaw = -1.1; this.pitch = 0.7; this.distance = 4.1; this.target = [0, 0, 0]; this.requestDraw(); }
  camera() { return cameraMatrix(this.yaw, this.pitch, this.distance, this.target, this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight)); }
  pan(dx, dy) {
    const { right, up } = this.camera(), scale = this.distance * 0.83 / Math.max(1, this.canvas.clientHeight);
    this.target = this.target.map((v, k) => v - right[k] * dx * scale + up[k] * dy * scale);
  }
  bindEvents() {
    const canvas = this.canvas, pointers = new Map();
    let drag = null, pinch = null;
    const pinchState = () => { const [a, b] = [...pointers.values()]; return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, distance: Math.hypot(a.x - b.x, a.y - b.y) }; };
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('pointerdown', e => {
      if (e.button !== 0 && e.button !== 2) return;
      canvas.focus({ preventScroll: true }); canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1) drag = { x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY, moved: false, pan: e.button === 2 || e.shiftKey, button: e.button };
      if (pointers.size === 2) { pinch = pinchState(); if (drag) drag.moved = true; }
    });
    canvas.addEventListener('pointermove', e => {
      if (!pointers.has(e.pointerId) || !drag) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size >= 2) {
        const next = pinchState();
        if (pinch) { this.distance = Math.max(0.25, Math.min(30, this.distance * pinch.distance / Math.max(1, next.distance))); this.pan(next.x - pinch.x, next.y - pinch.y); }
        pinch = next; this.requestDraw(); return;
      }
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > 4) drag.moved = true;
      if (drag.moved) {
        if (drag.pan || e.shiftKey) this.pan(dx, dy);
        else { this.yaw -= dx * 0.008; this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.008)); }
        this.requestDraw();
      }
      drag.x = e.clientX; drag.y = e.clientY;
    });
    const finish = e => {
      if (!pointers.has(e.pointerId)) return;
      if (e.type === 'pointerup' && drag && !drag.moved && !drag.pan && drag.button === 0 && this.interactive && pointers.size === 1) {
        const bounds = canvas.getBoundingClientRect();
        const index = pickPoint(this.camera().matrix, this.positions, bounds.width, bounds.height, e.clientX - bounds.left, e.clientY - bounds.top, 12, this.pointSize);
        this.onPick(index, e.altKey);
      }
      pointers.delete(e.pointerId); pinch = null;
      if (pointers.size) { const [p] = pointers.values(); drag = { ...drag, ...p, moved: true }; } else drag = null;
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    };
    canvas.addEventListener('pointerup', finish); canvas.addEventListener('pointercancel', finish);
    canvas.addEventListener('wheel', e => { e.preventDefault(); const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? canvas.clientHeight : 1); this.distance = Math.max(0.25, Math.min(30, this.distance * Math.exp(Math.max(-1, Math.min(1, delta * 0.001))))); this.requestDraw(); }, { passive: false });
    canvas.addEventListener('keydown', e => { if (e.key.toLowerCase() === 'f') { this.fit(); e.preventDefault(); } });
  }
  requestDraw() { if (!this.drawPending) { this.drawPending = true; requestAnimationFrame(() => { this.drawPending = false; this.draw(); }); } }
  draw() {
    if (this.lost) return;
    const { canvas, overlay, gl } = this, ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) { canvas.width = overlay.width = pixelWidth; canvas.height = overlay.height = pixelHeight; }
    const { matrix } = this.camera();
    gl.viewport(0, 0, pixelWidth, pixelHeight); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (this.positions.length) {
      gl.useProgram(this.program);
      for (const [buffer, location] of [[this.positionBuffer, this.positionLocation], [this.colorBuffer, this.colorLocation]]) {
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 0, 0);
      }
      gl.uniformMatrix4fv(this.matrixLocation, false, matrix); gl.uniform1f(this.sizeLocation, this.pointSize * ratio);
      gl.drawArrays(gl.POINTS, 0, this.positions.length / 3);
    }
    const ctx = overlay.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
    this.clicks.forEach((click, ordinal) => {
      const offset = click.display_index * 3;
      const p = projectPoint(matrix, this.positions[offset], this.positions[offset + 1], this.positions[offset + 2], width, height);
      if (!p || p.x < 0 || p.x > width || p.y < 0 || p.y > height) return;
      const color = click.object_id ? instanceColor(click.object_id) : [234, 98, 104];
      ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.fillStyle = '#142033bb'; ctx.fill();
      ctx.strokeStyle = `rgb(${color.join(',')})`; ctx.lineWidth = 2; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p.x - 3, p.y); ctx.lineTo(p.x + 3, p.y);
      if (click.object_id) { ctx.moveTo(p.x, p.y - 3); ctx.lineTo(p.x, p.y + 3); }
      ctx.lineWidth = 1.4; ctx.stroke();
      ctx.font = '10px system-ui'; ctx.fillStyle = '#e2e9f0'; ctx.strokeStyle = '#111923'; ctx.lineWidth = 3;
      const text = `${ordinal + 1}`; ctx.strokeText(text, p.x + 10, p.y - 8); ctx.fillText(text, p.x + 10, p.y - 8);
    });
  }
}
