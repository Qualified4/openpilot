// This panel has no saved state and never rearms after leaving the page.
export function collectConfig(root, defaults) {
  const config = structuredClone(defaults);
  for (const input of root.querySelectorAll('[data-debug-path]')) {
    const keys = input.dataset.debugPath.split('.');
    let target = config;
    for (const key of keys.slice(0, -1)) target = target[key];
    target[keys.at(-1)] = input.type === 'checkbox' ? input.checked : Number(input.value);
  }
  return config;
}

export function startPanel(root, schema) {
  const label = (ko, en) => `${ko} / ${en}`;
  const check = (path, text) => `<label><input type="checkbox" data-debug-path="${path}"> ${text}</label>`;
  const numeric = (path, text, min, max, step, value) => `<label>${text}<input type="number" data-debug-path="${path}" min="${min}" max="${max}" step="${step}" value="${value}" required></label>`;
  const select = (path, text, options, value) => `<label>${text}<select data-debug-path="${path}">${options.map(([n, name]) => `<option value="${n}"${n === value ? ' selected' : ''}>${n} · ${name}</option>`).join('')}</select></label>`;
  const types = Array.from({ length: 32 }, (_, n) => [n, schema.detect_codes[n] || label('미문서화', 'undocumented')]);
  const codes = schema.lane_codes.map(({ value, label: name, documented }) => [value, documented ? name : label('미문서화 · 실차 확인 필요', 'undocumented · verify on cluster')]);
  root.innerHTML = `<summary>CCNC Debug</summary><div class="ccnc-debug-body">
    <p>${label('초기값은 송신 없는 미리보기입니다. 계기판 송신은 P 또는 브레이크를 밟은 N, 완전 정차, 제어 해제 상태에서만 허용됩니다.', 'Starts in preview. Cluster TX requires P or N with brake held, zero speed and disengaged controls.')}</p>
    <p>${label('이동·경고·통신 단절·0.6초 만료 시 정상 출력으로 복귀하며, 다시 송신하려면 직접 켜야 합니다.', 'Motion, warnings, disconnect or 0.6s expiry restore normal output. Rearm manually.')}</p>
    <div class="ccnc-debug-actions"><button type="button" class="btn" data-debug-connect>${label('미리보기 연결', 'Connect preview')}</button>
      ${check('transmit', label('실제 계기판 CAN 송신', 'Live cluster CAN TX'))}
      <button type="button" class="btn" data-debug-stop>${label('전체 OFF', 'All OFF')}</button></div>
    <output data-debug-status role="status">OFF</output>
    <fieldset><legend>${check('vehicles_enabled', label('차량 표시 디버그', 'Vehicle display debug'))}</legend>
      <p>${label('DETECT 기본값 6 = 흰색 화물차. 각 슬롯 OFF는 해당 슬롯의 정상 출력을 유지합니다.', 'DETECT defaults to 6 = WHITE TRUCK. Disabled slots retain normal output.')}</p>
      <p>${label('FF LATERAL: +우측 / −좌측, 7비트 래핑. LF·LR: 좌측 거리, RF·RR: 우측 거리(모두 양수).', 'FF LATERAL: +right / −left, 7-bit wrap. LF/LR measure left; RF/RR measure right (positive).')}</p>
      ${Object.entries(schema.defaults.vehicles).map(([slot, entry]) => `<fieldset><legend>${check(`vehicles.${slot}.enabled`, slot)}</legend><div class="ccnc-debug-fields">
        ${select(`vehicles.${slot}.detect`, 'DETECT', types, 6)}
        ${numeric(`vehicles.${slot}.distance`, 'DISTANCE (m)', 0, ['LR', 'RR'].includes(slot) ? 25.5 : 204.7, .1, entry.distance)}
        ${numeric(`vehicles.${slot}.lateral`, 'LATERAL (m)', slot === 'FF' ? -6.4 : 0, slot === 'FF' ? 6.3 : 12.7, .1, entry.lateral)}</div></fieldset>`).join('')}
    </fieldset>
    <fieldset><legend>${check('lanes_enabled', label('차선 표시 디버그', 'Lane display debug'))}</legend>
      <p>${label('색상·종류는 별도 CAN 필드가 없습니다. 표시 코드 하나를 선택합니다. DBC: 1 숨김, 2 흰색, 4 주황색, 6 초록색. 8·10·14의 색상/점선/실선 의미는 미문서화되어 추정하지 않습니다.', 'Color/type share one display code. DBC: 1 hidden, 2 white, 4 orange, 6 green. Color/dash/solid meanings of 8/10/14 are undocumented.')}</p>
      ${['left', 'right'].map(side => `<fieldset><legend>${check(`lanes.${side}.enabled`, side === 'left' ? label('좌측', 'Left') : label('우측', 'Right'))}</legend><div class="ccnc-debug-fields">
        ${select(`lanes.${side}.display`, label('CAN 표시값 · 색상/종류', 'Display code · color/type'), codes, 2)}
        ${numeric(`lanes.${side}.position`, label('위치 코드', 'Position code'), 0, 63, 1, 15)}</div></fieldset>`).join('')}
      <p>${label('위치는 6비트 0~63(미터 아님, DBC 주석 0~15와 다름). 곡률은 4비트 0~15(DBC 주석 0~31과 다름). 곡률·방향은 좌우 공용이므로 독립 설정이 불가능합니다.', 'Position: 6-bit 0–63 (not metres; DBC annotation 0–15). Curvature: 4-bit 0–15 (annotation 0–31). Curvature/direction are shared by both sides.')}</p>
      ${check('curve_enabled', label('공용 곡률·방향 덮어쓰기', 'Override shared curvature/direction'))}
      <div class="ccnc-debug-fields">${numeric('curvature', label('곡률 코드', 'Curvature code'), 0, 15, 1, 0)}
        ${select('direction', label('방향', 'Direction'), [[0, label('좌 · 기존 생성기 기준', 'left · generator convention')], [1, label('우 · 기존 생성기 기준', 'right · generator convention')]], 0)}</div>
    </fieldset>
    <details><summary>${label('DBC raw / 물리값 미리보기', 'DBC raw / physical preview')}</summary>
      <p>${label('예시 payload는 선택 필드만 담으며 CAN으로 송신하지 않습니다. 실제 송신은 기존 메시지에서 해당 필드만 교체합니다.', 'Example payloads contain selected fields only and are never sent. Live TX replaces those fields in the existing messages.')}</p><pre data-debug-preview></pre></details>
    <details><summary>${label('DBC 실제 필드 구조', 'Actual DBC layout')}</summary><pre data-debug-dbc></pre></details>
    </div>`;
  root.querySelector('[data-debug-dbc]').textContent = JSON.stringify(schema.signals, null, 2);
  let socket = null, timer = null;
  const tx = root.querySelector('[data-debug-path="transmit"]');
  const state = root.querySelector('[data-debug-status]');
  const connect = root.querySelector('[data-debug-connect]');
  const stop = () => {
    tx.checked = false;
    clearInterval(timer);
    timer = null;
    socket?.close();
    socket = null;
    connect.disabled = false;
    state.textContent = 'OFF';
  };
  const send = () => {
    if (socket?.readyState !== WebSocket.OPEN) return;
    if (![...root.querySelectorAll('[data-debug-path]')].every(input => input.checkValidity())) {
      tx.checked = false;
      state.textContent = label('입력 범위·간격 확인', 'Check input range/step');
      // Stop renewing the lease on any invalid edit.
      socket.close();
      return;
    }
    socket.send(JSON.stringify(collectConfig(root, schema.defaults)));
  };
  connect.addEventListener('click', () => {
    stop();
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/ccnc_debug`);
    const opened = socket;
    connect.disabled = true;
    state.textContent = label('연결 중', 'Connecting');
    opened.onopen = () => { send(); timer = setInterval(send, 150); };
    opened.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.status?.disarmed) tx.checked = false;
      state.textContent = message.error || `${message.status?.active ? 'CAN TX' : 'PREVIEW'} · ${message.status?.reason || ''}`;
      root.querySelector('[data-debug-preview]').textContent = JSON.stringify(message.preview || {}, null, 2);
    };
    opened.onerror = () => { state.textContent = label('연결 실패', 'Connection failed'); };
    opened.onclose = () => { if (socket === opened) stop(); };
  });
  root.addEventListener('input', send);
  root.querySelector('[data-debug-stop]').addEventListener('click', stop);
  root.addEventListener('toggle', () => { if (!root.open) stop(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  window.addEventListener('pagehide', stop);
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
