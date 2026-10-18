import assert from 'node:assert/strict';
import test from 'node:test';
import { collectConfig, statusText } from '../src/features/tools/ccnc_debug.js';

test('each browser input edits its own field without touching defaults', () => {
  const defaults = { transmit: false, vehicles: { LF: { enabled: false, distance: 10, lateral: 3, detect: 6 },
    FF: { enabled: false, distance: 10, lateral: 0, detect: 6 } },
    lanes: { left: { enabled: false, display: 2, position: 15 }, right: { enabled: false, display: 2, position: 15 } },
    curvature: 0, direction: 0 };
  const before = structuredClone(defaults);
  const inputs = [
    { dataset: { debugPath: 'vehicles.LF.distance' }, type: 'range', value: '22.2' },
    { dataset: { debugPath: 'vehicles.FF.lateral' }, type: 'range', value: '-6.4' },
    { dataset: { debugPath: 'vehicles.LF.enabled' }, type: 'checkbox', checked: true },
    { dataset: { debugPath: 'lanes.right.display' }, type: 'select-one', value: '10' },
  ];
  const result = collectConfig({ querySelectorAll: () => inputs }, defaults);
  assert.equal(result.vehicles.LF.distance, 22.2);
  assert.equal(result.vehicles.LF.detect, 6);
  assert.equal(result.vehicles.LF.enabled, true);
  assert.equal(result.vehicles.FF.lateral, -6.4);
  assert.equal(result.lanes.right.display, 10);
  assert.deepEqual(result.lanes.left, defaults.lanes.left);
  assert.equal(result.transmit, true);  // Turning debug on transmits; there is no separate TX switch.
  assert.equal(collectConfig({ querySelectorAll: () => inputs }, defaults, false).transmit, false);
  assert.deepEqual(defaults, before);
});

test('status explains pauses and stops in plain words', () => {
  assert.match(statusText({ reason: 'transmitting', active: true }), /계기판 송신 중/);
  assert.match(statusText({ reason: 'CCNC RX stale', paused: true }), /^일시정지 \/ Paused · 카메라 CAN 수신 지연/);
  assert.match(statusText({ reason: 'vehicle moving', disarmed: true }), /중지됨[\s\S]*차량 이동[\s\S]*다시 켜세요/);
  assert.equal(statusText({ reason: 'something new' }), 'something new');
});
