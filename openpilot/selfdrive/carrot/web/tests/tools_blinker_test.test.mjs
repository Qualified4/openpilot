import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { TESTS, offValues } from '../src/features/tools/blinker_test.js';

const keys = readFileSync(new URL('../../../../common/params_keys.h', import.meta.url), 'utf8');

test('the tools test panel writes the two trial params with off / left / right', () => {
  assert.deepEqual(TESTS.map(({ param }) => param), ['CanfdBlinkerMenuTest', 'CanfdBvmView']);
  for (const { param, options } of TESTS) {
    assert.deepEqual(options.map(([value]) => value), [0, 1, 2]);
    // both are integer params cleared on every start, so a forgotten trial never survives a reboot
    assert.match(keys, new RegExp(`\\{"${param}", \\{CLEAR_ON_MANAGER_START, INT, "0"\\}\\}`));
  }
});

test('folding the panel turns off only the trials that are on', () => {
  assert.deepEqual(offValues({ CanfdBlinkerMenuTest: 0, CanfdBvmView: 2 }), [['CanfdBvmView', 0]]);
  assert.deepEqual(offValues({ CanfdBlinkerMenuTest: '1', CanfdBvmView: '0' }), [['CanfdBlinkerMenuTest', 0]]);
  assert.deepEqual(offValues({}), []);
  assert.deepEqual(offValues(undefined), []);
});
