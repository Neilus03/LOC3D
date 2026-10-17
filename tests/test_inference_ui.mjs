// Run with: node --test tests/test_inference_ui.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizePoints, cameraMatrix, projectPoint, pickPoint, instanceColor } from '../src/delimit3d/inference/static/viewer.mjs';

test('scene normalization preserves input indices and handles a constant axis', () => {
  const input = [[10, 20, 30], [20, 30, 30], [15, 25, 30]];
  const { positions, center, scale } = normalizePoints(input);
  assert.deepEqual(center, [15, 25, 30]); assert.equal(scale, 5);
  assert.deepEqual([...positions], [-1, -1, 0, 1, 1, 0, 0, 0, 0]);
  assert.deepEqual([...normalizePoints([[8, 8, 8]]).positions], [0, 0, 0]);
});
test('invalid coordinates are rejected before they reach the graphics buffer', () => {
  for (const points of [[], [[NaN, 0, 1]], [[0, 0]], [[Infinity, 0, 1]]]) assert.throws(() => normalizePoints(points));
});
test('camera projects its target at the screen center and clips behind-camera points', () => {
  const { matrix } = cameraMatrix(0, 0, 4, [0, 0, 0], 2);
  const center = projectPoint(matrix, 0, 0, 0, 800, 400);
  assert.ok(Math.abs(center.x - 400) < 0.001); assert.ok(Math.abs(center.y - 200) < 0.001);
  assert.equal(projectPoint(matrix, 5, 0, 0, 800, 400), null);
});
test('picking selects visible points and does not prompt an occluded point', () => {
  const { matrix } = cameraMatrix(0, 0, 4, [0, 0, 0], 1);
  const points = new Float32Array([-1, 0, 0, 1, 0, 0, 0, 0.5, 0]);
  assert.equal(pickPoint(matrix, points, 400, 400, 200, 200), 1);
  const side = projectPoint(matrix, 0, 0.5, 0, 400, 400);
  assert.equal(pickPoint(matrix, points, 400, 400, side.x, side.y), 2);
  assert.equal(pickPoint(matrix, points, 400, 400, 10, 10), -1);
});
test('picking accounts for point size rather than selecting a hidden center', () => {
  const { matrix } = cameraMatrix(0, 0, 4, [0, 0, 0], 1);
  const points = new Float32Array([-1, 0, 0, 1, 0.003, 0]);
  assert.equal(pickPoint(matrix, points, 400, 400, 200, 200, 12, 6), 1);
});
test('background is neutral and object palette remains stable across predictions', () => {
  assert.deepEqual(instanceColor(0), [60, 73, 91]);
  assert.notDeepEqual(instanceColor(1), instanceColor(2));
  assert.deepEqual(instanceColor(1), instanceColor(13));
});
