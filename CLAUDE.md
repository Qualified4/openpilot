
## 역할 분담

이 규칙은 세션 모델이 Fable일 때만 적용한다. 다른 모델이면 무시하고 직접 작업한다.

너는 설계자이자 오케스트레이터다.

- 요구사항을 정리하고, 설계하고, 작업을 단계로 나눈다.
- 코드는 직접 수정하지 않는다. 구현은 항상 `implementer` 서브에이전트에게 위임한다.
- 위임할 때는 목표, 수정할 파일, 제약 조건, 완료 기준을 명확히 적어서 넘긴다.
- implementer의 결과를 검수하고, 문제가 있으면 수정 지시를 다시 내린다.
- 각 단계가 끝나면 사용자에게 변경 파일과 테스트 결과를 짧게 보고한다.
- 되돌리기 어려운 결정(구조 변경, 의존성 추가, 데이터 삭제 등)은 진행 전에 사용자에게 확인한다.

### 작업자 선택 규칙 (사용량 기준, 2026-10-27)

위임할 작업자는 **현재 사용량 잔여분**으로 고른다. Claude 사용량은 **5시간 세션 한도** 기준이다(주간 한도가 아님). 사용량은 사용자가 알려 준 값, 세션 한도 메시지(429 "session limit", 리셋 시각), `codex` 한도 메시지로 파악한다. 에이전트는 사용량을 직접 읽을 수 없으므로, 모르면 작업 시작 전에 사용자에게 묻고, 같은 5시간 창 안에서 429가 한 번이라도 났으면 그 창이 끝날 때까지 Claude 사용량을 70% 이상으로 간주한다. 한 작업 안에서 작업자를 바꾸지 않는다(작업 단위로 결정).

1. **Claude 5시간 세션 사용량(사용한 비율)이 70% 미만** → `implementer` 서브에이전트, 모델 **Opus 5.5**(`model: "opus"`).
2. 그 외(Claude 70% 이상 사용) → **GPT(Codex CLI)**:
   - GPT 잔여량 **50% 이상** → `codex exec -m gpt-6-astra -c model_reasoning_effort=medium -s danger-full-access "<지시>" < /dev/null > <log> 2>&1`
   - GPT 잔여량 **10% 이상 50% 미만** → `codex exec -m gpt-6.1-sol -c model_reasoning_effort=high ...`
   - GPT 잔여량 **10% 미만** → 다시 `implementer`(Opus 5.5).
3. Codex에 맡길 때는 지시서를 `.analysis/ccnc-field-412-20261022/CODEX_PROMPT_<작업>_KO.md`로 쓰고 로그의 `tokens used`로 완료를 확인한다. 검증 범위·시간 상한·금지 사항은 implementer 지시와 같은 수준으로 적는다.
4. 사용량이 한도에 걸려 작업자가 멈추면(429), 리셋을 기다리지 말고 위 규칙으로 다른 작업자에게 같은 지시서로 이어 맡긴다(진행 상태·산출물 경로를 지시서에 적는다).

## 로그 분석·재생 규칙 (모든 모델·세션 공통)

- **디코드 결과는 반드시 캐시한다.** fcamera/qcamera를 디코드해 만든 StripNet 입력 띠와 프레임 메타(시각, odometer, laneLineProbs, 차선 y, vEgo, 깜빡이, laneChangeState)는 `.analysis/strip-cache/<세트>/<구간>.npz`에 저장하고, 재생·분석은 캐시에서 읽는다. 캐시가 없을 때만 디코드하되 그 결과도 캐시에 저장한다. 같은 로그를 두 번 디코드하지 않는다. 캐시는 지우지 않는다.
- 캐시 생성·재생 도구는 `.analysis/strip-cache/tools/`의 공용 도구(`build_cache.py`, 병렬 `--jobs`)를 쓰고, 라운드별 폴더에 새 디코더를 만들지 않는다. 도구를 고치면 호환을 유지한다.
- **라벨 수정은 `tools/lane_profile/label_fixes.json`에만 기록한다.** 원본 구간 파일과 기존 manifest는 고치지 않는다. 확장 어휘(복선·모름) manifest를 읽는 학습·평가 스크립트는 행을 읽은 직후 `label_fixes.apply(rows, types=dataset_io.TYPES)`를 거친다(`codex/trainfast/train_fast.py` 적용됨). 같은 파일의 `exclusions`(차선이 아닌 구간: 정지 구역·횡단보도·빗길 헤드라이트 반사 등)는 학습(train_fast 적용됨)·card 재생·검증 채점에서 `label_fixes.excluded(...)`로 뺀다(앞차에 가려짐·회전 중·주차장 포함). 보고서에 적용된 수정 행 수와 제외 구간 수를 적는다. 라벨 기준: 이중 실선(두 줄 모두 실선)은 복선 클래스가 아니라 `solid`로 단다(사용자 10-23). dbl_near = 내 차 쪽 줄이 점선, dbl_far = 내 차 쪽 줄이 실선이고 반대쪽이 점선.
- **주행 북마크는 rlog가 아니라 Home Assistant의 Road Viewer 애드온에 있다**(소스 `../roadviewer-ha`, 구간 메타데이터, API `GET /api/logs/<id>/bookmarks`, Ingress). rlog의 `userBookmark`·`bookmarkButton`에는 없다. 조회는 ha-mcp(Codex 설정 `mcp_servers.ha-mcp`)로 읽기 전용으로 하고, 결과를 `.analysis/.../bookmarks/roadviewer_bookmarks.json`처럼 저장한다. **사용자가 Road Viewer 로그를 지우기 전에 먼저 받아 둔다.** 북마크(제목 = 사용자가 본 선 종류·색, 메모)는 사람 라벨로 취급해 해당 시각 앞뒤 짧은 범위를 학습·평가 라벨이나 제외 근거로 쓰고, 새 주행 분석 때 가장 먼저 확인한다.
- 로그 이벤트(carState·modelV2·sendcan·can 일부 등)는 `.analysis/cache/rlog_events/<루트>--<구간>.capnp`(`tools/ccnc/replay_full.py`의 `events()`, rlog 크기·mtime이 바뀌면 자동 재생성)에서 읽고 rlog를 다시 풀지 않는다. 0x162 차량 재생은 `tools/ccnc/replay_par.py --jobs N`. strip-cache = 영상 띠, rlog_events = 로그 이벤트. StripNet 모델 평가(13seg/16seg/비 404·400)는 `.analysis/lane-retrain-rain-20261020/tools/eval_cached.py`.
- 무거운 작업(디코드, 캐시 생성, 26구간급 재생)은 Cowork VM(2코어)이 아니라 WSL/Windows(16스레드, GPU)에서 병렬로 돌리도록 스크립트를 만들고 실행 명령을 사용자에게 준다. VM은 캐시 위 추론·집계만 한다.
- 분석 보고서에는 사용한 캐시 경로와 새로 만든 캐시 구간을 적는다.

## 속도 규칙 (모든 모델·세션 공통)

- 작업이 느리다는 전제로 시작한다. 무거운 단계(디코드, 재생, 학습, 대량 집계)마다 **더 빠르게 돌릴 방법을 먼저 찾아 적용**한다: 캐시 재사용, 병렬(`--jobs`), 필요한 구간·프레임만 처리, WSL/GPU로 이관, 같은 계산 반복 금지.
- **작업 시작 전에 예상 소요 시간을 사용자에게 출력**한다(단계별 분 단위, 근거 한 줄: 처리 프레임 수·코어 수·이전 실측). 실제가 예상의 1.5배를 넘기면 중간에 알리고 남은 단계를 줄이거나 나눈다.
- 보고서에 실제 소요 시간과 예상 대비 차이를 적는다.
- **여러 번 학습(시드·ablation)은 스크립트 하나로 N개 동시 실행**한다(기본 `-j 4`, 사용자 PC GPU 여유 확인됨 10-22). 각 실행은 결과 폴더를 원자적으로 만들어 선점하고, 끝난 실행(model.json)은 건너뛰며, 학습은 끝났는데 내보내기만 실패한 실행은 다시 학습하지 않고 내보내기만 한다. 사용자에게 터미널을 여러 개 열게 하지 않는다. 예: `.analysis/ccnc-field-412-20261022/codex/TRAIN_SINGLE_PAR.sh`.
- **GPU가 필요한 명령(학습, GPU 프로파일·평가)은 에이전트가 직접 실행**한다. Codex는 기본 샌드박스에서 GPU(`/dev/dxg`)가 보이지 않으므로 그 명령에만 `sandbox_permissions="require_escalated"`로 권한 상승을 요청하고(사용자 승인), 파일 읽기·편집·CPU 검사는 기본 샌드박스에 둔다. "CUDA 없음"으로 학습을 사용자에게 넘기기 전에 이 방법을 먼저 쓴다.
- 차선 스트립 모델 학습은 GPU 상주 데이터 trainer(`.analysis/ccnc-field-412-20261022/codex/trainfast/`, 10-22 기준 40에포크 약 2분)를 기본으로 쓴다. DataLoader 방식의 이전 trainer(약 10분)로 새 실험을 만들지 않는다.
- **수정·검증은 한 번에 묶는다.** 먼저 원인을 전부 찾고(실패 테스트 전부, 재생에서 드러난 문제 전부를 기존 결과·캐시 위에서 확인), 모든 수정을 한 번에 넣은 뒤 테스트와 회귀 재생을 한 번에 돌린다. 한 줄 고치고 전체 테스트·재생, 또 한 줄 고치고 전체 테스트·재생을 반복하지 않는다. 원인 확인용 실행은 해당 테스트 하나(`-k`)나 해당 구간 하나로 좁힌다. 최종 회귀에서 새 문제가 나오면 원인을 모두 모은 뒤 다시 한 번만 돌린다.
- WSL/Windows에서 돌 스크립트는 VM 검증만으로 완료가 아니다. 기존에 WSL에서 통과한 스크립트(예: wsl_eval.sh)의 환경 처리(한글 경로 → 영문 symlink OPROOT/PYTHONPATH, `~/mnt/OpenpilotLogs` 링크 또는 LOGS_ROOT, 임시 HOME)를 그대로 유지하고, "VM에서만 확인함"을 보고에 명시한다.
