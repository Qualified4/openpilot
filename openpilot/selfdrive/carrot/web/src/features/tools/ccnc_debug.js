// This panel has no saved state and never re-enables itself after leaving the page.
// Turning it on transmits at once; the controller still allows it only when parked.
export function collectConfig(root, defaults, transmit = true) {
  const config = structuredClone(defaults);
  for (const input of root.querySelectorAll('[data-debug-path]')) {
    const keys = input.dataset.debugPath.split('.');
    let target = config;
    for (const key of keys.slice(0, -1)) target = target[key];
    target[keys.at(-1)] = input.type === 'checkbox' ? input.checked : Number(input.value);
  }
  config.transmit = transmit;
  return config;
}

// Controller/server reasons, shown in plain words.
const REASONS = {
  transmitting: ['계기판 송신 중', 'Transmitting to cluster'],
  'no field selected': ['켜진 항목 없음 · 차량/차선 항목을 켜세요', 'Nothing selected'],
  preview: ['꺼짐', 'Off'],
  'vehicle moving': ['차량 이동', 'Vehicle moving'],
  'select P, or N with brake held': ['P단 또는 브레이크를 밟은 N이 필요합니다', 'Select P, or N with brake held'],
  'controls active': ['오픈파일럿 제어 중', 'Controls active'],
  'CAN unavailable': ['CAN 통신 이상', 'CAN unavailable'],
  'CCNC RX stale': ['카메라 CAN 수신 지연', 'Camera CAN late'],
  'pedal, blind spot or trailer warning': ['페달·사각지대·트레일러 경고', 'Pedal, blind spot or trailer warning'],
  'warning active': ['계기판 경고 표시 중', 'Cluster warning active'],
  'settings expired': ['설정 갱신 지연', 'Settings late'],
  'CCNC generator unavailable / waiting': ['차량 응답 대기 중', 'Waiting for the vehicle'],
  'HDA1 CAMERA_SCC required': ['HDA1 카메라 SCC 차량 전용', 'HDA1 CAMERA_SCC required'],
  'RX-based cluster forwarding required': ['계기판 직접 송신 옵션을 끄세요', 'Turn off direct cluster TX'],
  'invalid configuration': ['설정값 오류', 'Invalid configuration'],
};
const label = (ko, en) => `${ko} / ${en}`;
const reasonText = reason => (REASONS[reason] ? label(...REASONS[reason]) : reason || '');

export function statusText(status) {
  if (!status) return '';
  if (status.disarmed) return `${label('중지됨', 'Stopped')} · ${reasonText(status.reason)} · ${label('다시 켜세요', 'turn on again')}`;
  if (status.paused) return `${label('일시정지', 'Paused')} · ${reasonText(status.reason)}`;
  return reasonText(status.reason);
}

const decimals = step => (String(step).split('.')[1] || '').length;

export function startPanel(root, schema) {
  const check = (path, text) => `<label><input type="checkbox" data-debug-path="${path}"> ${text}</label>`;
  const slider = (path, text, min, max, step, value, unit = '') => `<div class="ccnc-debug-slider">
      <span class="ccnc-debug-slider__label">${text}</span>
      <button type="button" class="smallBtn" data-debug-nudge="${path}" data-dir="-1" aria-label="${text} −">−</button>
      <input type="range" data-debug-path="${path}" min="${min}" max="${max}" step="${step}" value="${value}" data-unit="${unit}" aria-label="${text}">
      <button type="button" class="smallBtn" data-debug-nudge="${path}" data-dir="1" aria-label="${text} +">+</button>
      <output class="val" data-debug-value="${path}">${Number(value).toFixed(decimals(step))}${unit}</output></div>`;
  const select = (path, text, options, value) => `<label>${text}<select data-debug-path="${path}">${options.map(([n, name]) => `<option value="${n}"${n === value ? ' selected' : ''}>${n} · ${name}</option>`).join('')}</select></label>`;
  const types = Array.from({ length: 32 }, (_, n) => [n, schema.detect_codes[n] || label('미문서화', 'undocumented')]);
  const codes = schema.lane_codes.map(({ value, label: name, documented }) => [value, documented ? name : label('미문서화 · 실차 확인 필요', 'undocumented · verify on cluster')]);
  root.innerHTML = `<summary>CCNC Debug</summary><div class="ccnc-debug-body">
    <label class="ccnc-debug-master"><span>${label('디버그 켜기 · 켜면 바로 계기판에 송신', 'Debug on · transmits to the cluster at once')}</span>
      <span class="c-switch"><input class="c-switch__input" type="checkbox" data-debug-master><span class="c-switch__track" aria-hidden="true"></span></span></label>
    <output data-debug-status role="status">${reasonText('preview')}</output>
    <p class="ccnc-debug-note">${label('P단 또는 브레이크를 밟은 N에서 완전 정차·제어 해제 상태일 때만 송신합니다. 경고·수신 지연은 잠시 멈췄다 이어지고, 차량이 움직이거나 기어를 바꾸면 꺼지므로 다시 켜야 합니다.', 'Transmits only when parked (P, or N with brake) and disengaged. Warnings and late frames pause it; moving or shifting stops it until turned on again.')}</p>
    <fieldset><legend>${check('vehicles_enabled', label('차량 표시', 'Vehicles'))}</legend>
      <p class="ccnc-debug-note">${label('FF 횡위치: +우측 / −좌측. LF·LR은 좌측 거리, RF·RR은 우측 거리(양수). 끈 슬롯은 정상 출력을 유지합니다.', 'FF lateral: +right / −left. LF/LR left distance, RF/RR right distance. Off slots keep normal output.')}</p>
      ${Object.entries(schema.defaults.vehicles).map(([slot, entry]) => `<fieldset><legend>${check(`vehicles.${slot}.enabled`, slot)}</legend><div class="ccnc-debug-fields">
        ${select(`vehicles.${slot}.detect`, 'DETECT', types, 6)}
        ${slider(`vehicles.${slot}.distance`, label('거리', 'Distance'), 0, ['LR', 'RR'].includes(slot) ? 25.5 : 204.7, .1, entry.distance, ' m')}
        ${slider(`vehicles.${slot}.lateral`, label('횡위치', 'Lateral'), slot === 'FF' ? -6.4 : 0, slot === 'FF' ? 6.3 : 12.7, .1, entry.lateral, ' m')}</div></fieldset>`).join('')}
    </fieldset>
    <fieldset><legend>${check('lanes_enabled', label('차선 표시', 'Lanes'))}</legend>
      <p class="ccnc-debug-note">${label('색상·종류는 표시 코드 하나로 정합니다. DBC: 1 숨김, 2 흰색, 4 주황색, 6 초록색. 8·10·14는 미문서화입니다.', 'Color/type share one display code. DBC: 1 hidden, 2 white, 4 orange, 6 green. 8/10/14 are undocumented.')}</p>
      ${['left', 'right'].map(side => `<fieldset><legend>${check(`lanes.${side}.enabled`, side === 'left' ? label('좌측', 'Left') : label('우측', 'Right'))}</legend><div class="ccnc-debug-fields">
        ${select(`lanes.${side}.display`, label('표시 코드', 'Display code'), codes, 2)}
        ${slider(`lanes.${side}.position`, label('위치 코드', 'Position code'), 0, 63, 1, 15)}</div></fieldset>`).join('')}
      <fieldset><legend>${check('curve_enabled', label('곡률·방향 (좌우 공용)', 'Curvature/direction (shared)'))}</legend><div class="ccnc-debug-fields">
        ${slider('curvature', label('곡률 코드', 'Curvature code'), 0, 15, 1, 0)}
        ${select('direction', label('방향', 'Direction'), [[0, label('좌', 'left')], [1, label('우', 'right')]], 0)}</div></fieldset>
    </fieldset>
    <details><summary>${label('DBC raw / 물리값', 'DBC raw / physical values')}</summary><pre data-debug-preview></pre></details>
    <details><summary>${label('DBC 실제 필드 구조', 'Actual DBC layout')}</summary><pre data-debug-dbc></pre></details>
    </div>`;
  root.querySelector('[data-debug-dbc]').textContent = JSON.stringify(schema.signals, null, 2);
  const master = root.querySelector('[data-debug-master]');
  const state = root.querySelector('[data-debug-status]');
  let socket = null, pending = null;

  const showValue = input => {
    const output = root.querySelector(`[data-debug-value="${input.dataset.debugPath}"]`);
    if (output) output.textContent = `${Number(input.value).toFixed(decimals(input.step))}${input.dataset.unit || ''}`;
  };
  const send = () => {
    pending = null;
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(collectConfig(root, schema.defaults)));
  };
  // Sliders fire many events; coalesce to at most one edit per 50 ms.
  const queue = () => { if (pending === null) pending = setTimeout(send, 50); };
  const stop = (message = reasonText('preview')) => {
    master.checked = false;
    clearTimeout(pending);
    pending = null;
    const closing = socket;
    socket = null;
    closing?.close();
    state.textContent = message;
  };
  const start = () => {
    stop();
    master.checked = true;
    const opened = socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/ccnc_debug`);
    state.textContent = label('연결 중', 'Connecting');
    let last = '';
    opened.onopen = send;
    opened.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.preview) root.querySelector('[data-debug-preview]').textContent = JSON.stringify(message.preview, null, 2);
      last = message.error || statusText(message.status);
      if (socket === opened) state.textContent = last;
      if (message.status?.disarmed && socket === opened) stop(last);
    };
    opened.onerror = () => { if (socket === opened) state.textContent = label('연결 실패', 'Connection failed'); };
    opened.onclose = () => { if (socket === opened) stop(last && last !== statusText({ reason: 'transmitting' }) ? last : reasonText('preview')); };
  };

  master.addEventListener('change', () => (master.checked ? start() : stop()));
  root.addEventListener('input', event => {
    if (event.target === master) return;
    if (event.target.type === 'range') showValue(event.target);
    queue();
  });
  root.addEventListener('change', event => { if (event.target !== master) queue(); });
  root.addEventListener('click', event => {
    const button = event.target.closest('[data-debug-nudge]');
    if (!button) return;
    const input = root.querySelector(`[data-debug-path="${button.dataset.debugNudge}"]`);
    if (Number(button.dataset.dir) > 0) input.stepUp(); else input.stepDown();
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  root.addEventListener('toggle', () => { if (!root.open) stop(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  window.addEventListener('pagehide', () => stop());
  new MutationObserver(() => {
    const page = document.getElementById('pageTools');
    if (page.hidden || getComputedStyle(page).display === 'none') stop();
  })
    .observe(document.getElementById('pageTools'), { attributes: true, attributeFilter: ['hidden', 'style'] });
  return stop;
}

async function mount() {
  const stack = document.querySelector('#pageTools .tools-scroll-stack');
  if (!stack) return;
  const root = document.createElement('details');
  root.className = 'section ccnc-debug';
  root.innerHTML = '<summary>CCNC Debug</summary>';
  stack.append(root);
  root.addEventListener('toggle', async () => {
    if (!root.open || root.dataset.loaded) return;
    root.dataset.loaded = 'true';
    try {
      const response = await fetch('/api/ccnc_debug', { cache: 'no-store' });
      if (!response.ok) throw new Error(await response.text());
      startPanel(root, await response.json());
    } catch (error) {
      root.append(document.createTextNode(` · ${error.message}`));
      delete root.dataset.loaded;
    }
  });
}
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, { once: true });
  else mount();
}
