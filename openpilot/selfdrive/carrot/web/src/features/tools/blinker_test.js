// Tools > test panel next to CCNC Debug: one-shot CAN-FD trials that write their own Params (cleared on reboot).
// Each select writes its value at once through the shared setParam; folding the panel turns every trial off.
const label = (ko, en) => `${ko} / ${en}`;
const OFF_LEFT_RIGHT = [[0, label('끔', 'Off')], [1, label('왼쪽', 'Left')], [2, label('오른쪽', 'Right')]];

export const TESTS = Object.freeze([
  Object.freeze({
    param: 'CanfdBlinkerMenuTest',
    title: label('깜빡이 명령 테스트', 'Blinker command test'),
    note: label('자동주차 ECU를 멈추고 SPAS2 깜빡이 명령을 계속 보냅니다. 속도·기어 무관, 주행 중에도 동작하니 2인 1조로 시험하세요.',
      'Pauses the parking ECU and keeps sending the SPAS2 blinker command at any speed or gear; test with a second person.'),
    options: OFF_LEFT_RIGHT,
  }),
  Object.freeze({
    param: 'CanfdBvmView',
    title: label('후측방 카메라 화면 시험', 'Blind-spot camera view test'),
    note: label('깜빡이를 켜지 않고 계기판에 후측방 카메라 화면만 띄웁니다(0x400 화면 상태, 순정 프레임 뒤에 5 Hz). 표시 전용이며 P·D·R 모두 시험할 수 있습니다.',
      'Shows the blind-spot camera on the cluster without the blinker (0x400 view status, behind the stock frame at 5 Hz). Display only; works in P, D and R.'),
    options: OFF_LEFT_RIGHT,
  }),
]);

// Values to write when the panel is folded: only the trials that are still on.
export function offValues(values) {
  return TESTS.filter(({ param }) => Number(values?.[param] || 0) !== 0).map(({ param }) => [param, 0]);
}

function render(root) {
  root.innerHTML = `<summary>${label('깜빡이·카메라 시험', 'Blinker / camera tests')}</summary><div class="ccnc-debug-body">
    <p class="ccnc-debug-note">${label('재부팅하면 모두 꺼집니다. 이 칸을 접으면 켜 둔 시험을 끕니다.', 'All reset on reboot. Folding this panel turns them off.')}</p>
    ${TESTS.map(({ param, title, note, options }) => `<fieldset><legend>${title}</legend>
      <p class="ccnc-debug-note">${note}</p>
      <label>${param}<select data-test-param="${param}" disabled>${options.map(([value, name]) => `<option value="${value}">${name}</option>`).join('')}</select></label>
    </fieldset>`).join('')}
    <output data-test-status role="status"></output></div>`;
}

function mount() {
  const stack = document.querySelector('#pageTools .tools-scroll-stack');
  if (!stack) return;
  const root = document.createElement('details');
  root.className = 'section ccnc-debug';
  root.innerHTML = `<summary>${label('깜빡이·카메라 시험', 'Blinker / camera tests')}</summary>`;
  stack.append(root);
  const values = {};
  const status = text => { const out = root.querySelector('[data-test-status]'); if (out) out.textContent = text; };
  const write = async (param, value) => {
    await globalThis.setParam(param, value, { source: 'web_ui' });
    values[param] = value;
  };
  root.addEventListener('toggle', async () => {
    if (!root.open) {
      for (const [param, value] of offValues(values)) {
        try { await write(param, value); } catch (error) { console.warn('[blinker_test] off failed', param, error); }
      }
      return;
    }
    render(root);
    try {
      Object.assign(values, await globalThis.bulkGet(TESTS.map(({ param }) => param)));
      for (const select of root.querySelectorAll('[data-test-param]')) {
        select.value = String(Number(values[select.dataset.testParam] || 0));
        select.disabled = false;
      }
    } catch (error) {
      status(error?.message || String(error));
    }
  });
  root.addEventListener('change', async event => {
    const select = event.target.closest('[data-test-param]');
    if (!select) return;
    select.disabled = true;
    try {
      await write(select.dataset.testParam, Number(select.value));
      status(label('적용됨', 'Applied'));
    } catch (error) {
      select.value = String(Number(values[select.dataset.testParam] || 0));
      status(error?.message || String(error));
    } finally {
      select.disabled = false;
    }
  });
}
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, { once: true });
  else mount();
}
