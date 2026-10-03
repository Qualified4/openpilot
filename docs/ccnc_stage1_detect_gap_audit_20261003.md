# CCNC 1단계 추가 detect 공백 5샘플 원인 분석 — 2026-10-03

> 종료 결정: 사용자는 아래 판단 불가 2샘플을 알려진 한계로 유지하면서 1단계만 채택했다. 아래 감사 결과는 변경하지 않는다. [연구 사이클 종료 보고서](ccnc_research_cycle_close_20261003.md)를 참조한다.

현재 1단계 accepted/pending 이력 격리 수정은 유지한다. 이번 조사에서는 생산 코드, threshold, confirmation, hold, pending 및 슬롯 규칙을 변경하지 않았다.

결론은 **A 정상 변화 3샘플, B 확정된 실제 regression 0샘플, C 판단 불가 2샘플**이다. 0건의 B는 무회귀 증명이 아니다. `395/7` 두 샘플에는 연속 횡단 후보와 실제 차량이 존재하지만, 해당 반사점의 정확한 차량 대응과 한 번의 빠른 횡관측 정답을 확정하지 못했다. 1단계 수정은 유지하되, 표시 무회귀 채택 결론에는 이 횡단 상황의 추가 근거가 필요하다. 이 결과만으로 최소 수정의 필요성이나 방법을 확정하지 않는다.

| 샘플 | 제어 frame | raw source | 수정 전 → 후 detect | 분류 | 직접 원인 |
|---|---:|---:|---:|---|---|
| 38a/16 36.676683s LF | 3644 | 34 | 1→0 | A | last-good 경과 400ms로 entry 제거, 동일 ID 새 track의 확인 미완료 |
| 38a/16 36.722006s LF | 3649 | 34 | 1→0 | A | 위 새 track을 같은 publication으로 재사용, 확인 관측 추가 없음 |
| 393/1 56.307794s RF | 5621 | 52 | 2→0 | A | 최신 수용 y=-4.75가 기존 RF 유효 경계 -4.46172 밖 |
| 395/7 7.267240s LF | 6719 | 42 | 1→0 | C | y 보류 후 추정 위치 4.20이 LF 경계 4.03160 밖; 수정 전 예측 3.80은 안 |
| 395/7 7.317228s LF | 6724 | 42 | 1→0 | C | 같은 보류 상태, 선택 이력에 따른 경계 여유도 전/후 다름 |

영상 존재 분류는 다섯 샘플 모두 **실제 차량 존재**다. 이는 영상 장면의 차량 존재를 의미하며 각 radar 반사점이 그 차량의 어떤 부분인지까지 확정한 라벨은 아니다. A는 차량 자체가 사라졌다는 뜻이 아니다. 현재 표시 범위에서 벗어난 원시 궤적을 오래된 상태로 계속 표시하던 차이가 제거됐다는 분류다. 영상은 replay/알고리즘 입력으로 사용하지 않았다.

## 재현과 시간순 자료

frozen baseline SHA256은 `3db9dcb8695980bc3615d8a69d18e49fb43ca11e2da05b0a34b10d28bf9cf984`, 현재 수정 버전은 `5cf5150797fc8227ff7b6789c546ae89bbd9bd48a752c641bad051cdc28e5760`이다. 두 버전의 차이는 accepted y로 기존 횡이력/속도 계산 블록을 보호하는 1단계 수정이다.

`38a/16`, `393/1`, 그리고 연속 segment 상태를 재현하기 위한 `395/6→7`을 원본 로그에서 실행했다. 4,801 출력 시각에서 양쪽 값이 직전 20개 replay의 값과 모두 일치했다. 진단은 frozen 함수의 Python line tracing과 실행 전후 상태 복사로 수행했다. 관측값이나 내부 결정 state를 주입하거나 조건을 바꾸지 않았다. 다섯 target은 정확히 수정 전 detect>0, 수정 후 detect=0임을 assert했다.

세 구간을 각각 35.5~37.8s, 55.1~57.5s, 6.1~8.5s에서 기록했다. 각 개별 공백은 전/후 **최소 1초**를 별도 추출했다. 실제 송신 시각은 불규칙하므로 제어 frame×10ms로 계산하는 내부 deadline과 wall-time 차이를 구분한다.

로컬 자료는 `.analysis/archive/2026-10-03/ccnc-five-gaps/`에 있다.

- `*_raw_candidates.csv`: 모든 출력 시각의 raw 후보 전체, source/track ID와 d/y/v/vLead/yvRel, raw publication 시각. 동일 publication 재사용도 명시적으로 남긴다.
- `*_focus_timeline.csv`: 각 대상의 시간순 전/후 accepted mask, 마지막 수용 axis 관측, d/y/v 추정 state, vy, accepted history, pending 세 축 전체, last-good, 실제 사용 prediction, gate limits, temporal/qualification hold 잔여시간, 선택 ID, display birth/wire ID/slot/detect, lane probability/center.
- `*_candidate_branches.csv`: lane 후보별 실제 도달한 마지막 선택 분기와 그 시점의 경계/속도/거리 값. 약한 차선 때문에 후보 loop 자체가 실행되지 않은 경우도 full trace에서 구분한다.
- `*_event.jsonl.gz`: 각 샘플 전후 최소 1초의 모든 raw 후보, 전/후 모든 entry와 pending, selection/stop memory, 모델 차선·도로 경계, 실제 분기 trace와 최종 CAN 출력.
- `*_video.jpg`, `video_index.json`: 전후 1초의 9장 평가 프레임과 요청/실제 영상 시각. 순차 decode로 인접 프레임을 고르며 이번 27장의 시각 오차는 25ms 미만이다.

`entry.point.dRel/yRel/vRel`은 축별 관측이 거부되면 prediction을 포함하는 **추정 state**다. 이를 raw accepted observation이라고 부르지 않는다. CSV의 `accepted_axis_observation`은 수용된 axis의 마지막 raw 값과 시각을 별도로 표시한다. 실제 gate가 실행되지 않고 같은 publication을 재사용하면 mask/prediction은 null로 기록한다. 음수 hold remaining은 이미 만료된 기존 state의 진단 값이며 새 track의 확인 완료를 의미하지 않는다.

## 38a/16 — 우회전 중 source34의 보존 수명 차이

| 시간(s) / frame | raw d/y/v | 전 accepted mask | 후 accepted mask | 전 state y / vy | 후 state y / vy | 전/후 last-good frame | 전/후 LF detect |
|---|---|---|---|---|---|---|---|
| 36.268781 / 3604 | 16.50 / 4.00 / 6.14 | TTT | TTT | 4.00 / 3.00 | 4.00 / 3.00 | 3604 / 3604 | 1 / 1 |
| 36.328302 / 3609 | 17.05 / 4.65 / 6.39 | TFT | TFT | 4.15 / 5.50 | 4.15 / 3.00 | 3604 / 3604 | 1 / 1 |
| 36.423804 / 3619 | 17.30 / 5.20 / 6.53 | TTT | TFT | 5.20 / 7.25 | 4.45 / 3.00 | 3619 / 3604 | 1 / 1 |
| 36.479847 / 3624 | 17.80 / 6.45 / 7.11 | TFT | TFT | 5.5625 / 0 | 4.60 / 3.00 | 3619 / 3604 | 1 / 1 |
| 36.578469 / 3634 | 17.95 / 8.80 / 8.03 | TFT | TFT | 5.5625 / 0 | 4.90 / 3.00 | 3619 / 3604 | 1 / 1 |
| 36.622056 / 3639 | 직전 publication 재사용 | — | — | 5.5625 / 0 | 4.90 / 3.00 | 3619 / 3604 | 1 / 1 |
| **36.676683 / 3644** | **18.15 / 10.40 / 8.71** | TFT | 새 track bootstrap | 5.5625 / 0 | 10.40 / 0 | 3619 / 새3644 | **1 / 0** |
| **36.722006 / 3649** | 직전 publication 재사용 | — | — | 5.5625 / 0 | 10.40 / 0 | 3619 / 새3644 | **1 / 0** |
| 36.769535 / 3654 | 18.30 / 11.35 / 8.97 | TFT, 기존 identity 종료 | 새 track 지속 | 기존5.5625, source 해제 | 11.35 / 0 | 기존3619 / 3654 | 0 / 0 |

위 T/F 순서는 d/y/v다. bootstrap은 confirmed 수용과 구분한다. 표의 일부 반복 publication 행은 fresh 관측이 아니므로 gate를 재실행하지 않는다.

1. **마지막 정상 표시 연결:** 두 버전 모두 source34/track34, display birth2979다. 후 버전의 마지막 완전 수용은 36.268781s raw `(16.5,4.0,6.14)`다. 이후 d/v는 수용해도 y가 보류되므로 `good=3604`다. 마지막 실제 LF 출력은 36.622056s, wire34다.
2. **수정 전 유지 이유:** 36.328302s rejected y=4.65가 lateral history에 들어가 vy를 3→5.50으로 바꿨다. 이 속도로 다음 prediction y=4.70을 만들었고 raw5.20과 오차0.50이 gate0.55 이내여서 36.423804s를 수용했다. `good`과 selection qualification이3619로 갱신됐다. 이후 보류 중에도 오래된 추정 y=5.5625/offset 상태를 LF로 유지했다.
3. **수정 후 정확한 OFF 조건:** 3644 frame의 fresh publication 처리 시작에서 `frame-good=40 > RECONNECT=35`여서 기존 entry34가 dictionary prune에서 제거됐다. 원시 source34는 여전히 있어 동일 숫자 track34로 새 entry가 생겼다. tracker birth3644/count1이고 stable 조건인 3관측 및 span15frame을 충족하지 못한다. saved selection도 새 entry가 unconfirmed여서 retain 실패한다. 차선 probability가0.1 미만이라 lane 후보 loop는 실행되지 않는다. 최종 LF 후보 없음→detect0이다. **이 샘플의 직접 reset은 HOLD=30 branch가 아니라 RECONNECT=35 prune이다.** 앞선 요약의 “hold 만료”는 이 distinction을 생략했다.
4. **직접 결과 여부:** 격리로 rejected y가 vy를 높이지 못해 3619의 y 수용/last-good 갱신이 사라진 결과다. 후 버전은 유효 accepted vy=3을 보존했고 임의로0으로 만든 것이 아니다.
5. **오염 덕분의 우연한 유지:** 맞다. 해당 rejected y가 vy를 높여 후속 관측을 직접 gate 안으로 가져오고 보존 수명을 연장했다. raw y=10.40이 시점의 기준인데 이전 y=5.5625를 유지한 상태다.
6. **대체 후보:** raw40 `(22.90,6.30,5.58)`, raw57 `(14.70,3.10,-5.45)` 등은 존재한다. 40은 unconfirmed이고 횡위치도 멀며, 57도 unconfirmed다. 다른 source는 source34의 작은 위치/속도 변동을 설명하는 유일한 대체 후보가 아니다. 차선 문맥도 상실되어 새 LF 선정을 하지 않는 구간이다.
7. **association 누락 여부:** 아니다. target source34 자체가 계속 있다. 기존 동일-source 경로/수명 종료이며 absent-source 재연결 실패가 아니다.
8. **신뢰할 후보:** raw34는 y가4.0→10.4로 이동하고 자차 우회전 영상과 방향이 일치한다. 물체 관측 자체의 존재와 현재 인접 LF로 계속 표시할 자격은 다르다. 이 창에서 유효한 대체 LF는 확인되지 않았다.
9. **영상:** 실제 차량 존재. 자차가 우회전하며 기존 전방 차량들이 화면 좌측으로 벗어난다. 자동차가 없어졌다고 해석하지 않는다.

36.676683s의 기존 entry 기준 hold 잔여시간은 전+50ms/후-100ms, reconnect는 전+100ms/후-50ms다. 후 새 entry의 hold300ms는 bootstrap 이후의 새 값이다. 36.722006s에는 전 hold0ms, 후 새 entry250ms지만 확인 관측은 늘지 않았다. 전 wire34/birth2979가 유지되고 후는 wire 없음이다. 전도 36.769535s에 OFF가 된다.

lane probabilities(L0/L1/L2/L3)는 첫 target에서 `[0.00374,0.01267,0.00396,0.00021]`이다. 신규 lane center 측정은 양쪽 None이다. 이 창의 raw 10.4m를 인접 LF 차선 중심 좌표의 정답으로 대체한 것은 아니다. 현재 표시 문맥에서 이미 벗어난 원시 궤적을 stale 좌표로 더 오래 남긴 차이로 **두 샘플 A**로 분류한다.

## 393/1 — 횡단 후보의 최신 y 수용과 RF 경계 탈출

| 시간(s) / frame | raw d/y/v | 전/후 accepted mask | 전 state y / vy | 후 state y / vy | 전/후 good | RF detect |
|---|---|---|---|---|---|---|
| 56.061543 / 5596 | 22.00 / -2.10 / -0.15 | TFF / TFF | -1.50 / 0 | -1.50 / 0 | 5591 / 5591 | 0 / 0 |
| 56.107227 / 5601 | 22.05 / -2.65 / -0.03 | TFT / TFT | -1.50 / 0 | -1.50 / 0 | 5591 / 5591 | 0 / 0 |
| 56.159401 / 5606 | 22.05 / -3.20 / -0.01 | TTT / TTT | -3.20 / 0 | -3.20 / 0 | 5606 / 5606 | 2 / 2 |
| 56.207645 / 5611 | 22.05 / -3.80 / +0.18 | TTT / TTT | -3.80 / -11.50 | -3.80 / 0 | 5611 / 5611 | 2 / 2 |
| 56.263464 / 5616 | 21.95 / -4.25 / -1.96 | TTF / TTF | -4.25 / 0 | -4.25 / 0 | 5616 / 5616 | 2 / 2 |
| **56.307794 / 5621** | **21.90 / -4.75 / -2.10** | **TFF / TTF** | **-4.25 / 0** | **-4.75 / 0** | **5616 / 5621** | **2 / 0** |
| 56.359806 / 5626 | 21.90 / -5.30 / -2.12 | TFT / TTT | -4.25 / 0 | -5.30 / 0 | 5616 / 5626 | 0 / 0 |

1. **마지막 정상 연결:** source/track52, birth5546, 양쪽 RF의 마지막 공통 정상 표시는 56.263464s다. 그때 d/y는 raw를 수용하고, v는 reject하여 accepted state의 v=+0.18을 사용한다.
2. **수정 전 유지 이유:** 5596/5601의 보류 y를 history에 넣은 결과5611에서 vy=-11.50을 얻었다. 5616의 prediction=-4.375는 raw-4.25와 가까워 direct y gate를 통과하고 pending을 clear했다. 반면 후는 vy0이어서5616의 y를 기존 coherent pending으로 수용하고 queue를 보존했다.
3. **수정 후 OFF 조건:** 5621에서 양쪽 prediction y=-4.25는 같지만 baseline pending은1개로 y reject, 후 pending은4개의 일관된 관측으로 y accept다. 그 결과 후보 y는 전-4.25/후-4.75다. RF 선택 조건 `y > right_effective_bound`의 경계는 양쪽-4.461719이므로 전은통과/후는탈락한다. stopped 상태여서 temporal selection hold는 bypass한다. last-good 수명이나 확인 실패가 아니다.
4. **직접 결과:** history격리→5611 vy차이→5616 direct gate/pending clear차이→5621 pending 수용차이의 연결이다. threshold는 같다.
5. **오염 덕분 유지:** 간접적으로 맞다. 공백 순간의 vy는 양쪽0이고 accepted history도 clear 상태지만, 이전의 오염이 pending 경로를 바꿔 baseline이 최신 y를 늦게 수용하도록 만들었다. 이를 “그 순간 오염된 vy로 coast했다”라고 설명하면 틀린다.
6. **대체 후보:** target raw52는 계속 존재하고 이미 identity52에 연결됐다. 다른 RF raw37은 boundary rejected, raw42는59.5m에서 정지 속도여서 측면 속도/근거리 조건을 충족하지 않으며, raw57은 y=-24.75다. 해당 횡단차를 대체할 물리적 연속 후보는 확인되지 않았다.
7. **association 누락:** 아니다. 같은 source52/track52/birth5546을 업데이트한 뒤 geometry 선택에서 탈락했다.
8. **신뢰성:** raw d가22m 부근을 유지하며 y가 연속 감소하는 횡단 궤적은 자연스럽다. 후는 그 최신 y를 더 잘 수용한다. 유지할 RF 후보 없음은 “관측 차량 없음”과 다르다.
9. **영상:** 실제 차량 존재. 회색 차량이 좌→우 횡단하며 화면 우측으로 진행한다.

target의 temporal hold는 전250ms/후300ms다. 전 wire52/birth5546이한번 더 출력되고 후 wire없음이다. lane probabilities는 `[0.04460,0.21800,0.09236,0.03126]`이고 신규 side center는 None이다. RF outer geometry는 정차 중 기존 saved width를 사용하는 경로이며 낮은 현재 probability를 높은 geometry 신뢰도로 해석하지 않는다.

저장된 표시 중심은 양쪽 `[3.00425,-2.90446]`이며 현재 geometry valid는 둘 다 false다. stop/approach 표시 memory는 양쪽 모두 비어 있어 이 공백을 메울 별도 정차 hold가 없다.

이 샘플의 변화는 최신 횡위치 수용에 따른 **기존 표시 경계 탈출**이다. 동일 관측 후보를 association하지 못하거나 더 강하게 reject한 것이 아니므로 **A**로 분류한다. 이 분류는 실제 차량 위치를 정밀 보정해 검증했다는 뜻이 아니다. 전도 다음 출력56.359806s에는 v 수용 후 기존 속도 조건 때문에 RF를 표시하지 않는다.

## 395/7 — 정상 표시 단절이 아닌 초기 LF 표시 두 샘플 차이

| 시간(s) / frame | raw d/y/v | 전/후 accepted mask | 전 state y / vy | 후 state y / vy | good 전/후 | LF detect |
|---|---|---|---|---|---|---|
| 7.112678 / 6704 | 15.80 / 5.00 / -0.34 | TFT / TFT | 5.80 / 0 | 5.80 / 0 | 6694 / 6694 | 0 / 0 |
| 7.169933 / 6709 | 15.80 / 4.50 / -0.32 | TFT / TFT | 5.80 / 0 | 5.80 / 0 | 6694 / 6694 | 0 / 0 |
| 7.215358 / 6714 | 15.80 / 4.20 / -0.33 | TTT / TTT | 4.20 / -8.00 | 4.20 / 0 | 6714 / 6714 | 0 / 0 |
| **7.267240 / 6719** | **15.80 / 3.25 / -0.22** | **TFT / TFT** | **3.80 / 0** | **4.20 / 0** | **6714 / 6714** | **1 / 0** |
| **7.317228 / 6724** | **15.75 / 2.75 / -1.76** | **TFF / TFF** | **3.80 / 0** | **4.20 / 0** | **6714 / 6714** | **1 / 0** |
| 7.367456 / 6729 | 15.70 / 2.45 / -1.28 | TFT / TFT | 3.80 / 0 | 4.20 / 0 | 6714 / 6714 | 0 / 0 |
| 7.414379 / 6734 | 15.65 / 2.10 / -1.69 | TTT / TTT | 2.10 / 0 | 2.10 / 0 | 6734 / 6734 | 0 / 0 |
| 8.016458 / 6794 | 15.80 / -2.35 / +0.53 | TTT / TTT | -2.35 / -5.67 | -2.35 / -5.67 | 6794 / 6794 | 0 / 0, RF42 양쪽 표시 |

1. **마지막 정상 차량:** source42는 두 target 직전에 LF로 표시되지 않았다. LF는 6.268~6.768s에 source54가 있었고 6.816s부터 양쪽 비어 있다. source42의 tracker birth6624는 같지만 baseline에서만 7.267240s에 LF 표시가 처음 생긴다. 따라서 두 샘플을 기존 차량 표시가 끊겼다는 의미로 설명하면 부정확하다. FF53과 RF63은 해당 두 출력에서 양쪽 같다.
2. **수정 전 표시 이유:** 6704/6709의 rejected y=5.0/4.5를 history에 넣고 6714의 pending 확정 y=4.2를 더해 vy=-8을 만들었다. 다음 prediction은 4.2-8×0.05=3.8이다. y 관측3.25는 여전히 reject지만 이 prediction이 LF 경계4.031596보다 작아 선택됐다.
3. **수정 후 OFF 조건:** 6714에서 accepted history에는 최신4.2 하나만 들어가 vy0이다. 6719 prediction/state4.2가 LF 경계4.031596보다 크므로 `y < left_effective_bound`가 false다. 6724는 수정 전이 직전 LF 선택으로 +0.3m margin을 받아 경계4.35150, 수정 후는4.05150이다. 전 state3.8 통과/후4.2 실패가 계속된다. 정차 중이므로 temporal selection hold 및 slot stabilization은 bypass이고 hold 만료가 아니다.
4. **직접 결과:** accepted history 격리가 초기 vy 생성을 막은 결과다. source/ID/birth/관측횟수/확인 기준은 같다.
5. **오염으로 우연히 표시:** baseline의 처음 선택은 확실히 rejected evidence로 생성한 vy 덕분이다. 하지만 그 evidence가 실제 횡단차의 유효 궤적일 수 있어, 이 사실만으로 표시가 잘못됐다고 판정할 수 없다. accepted state 복구와 차량 표시 품질을 구분해야 한다.
6. **후보 존재:** 같은 source42가 연속 존재한다. d는 15.8m 부근이고 y는 5.0→4.5→4.2→3.25→2.75→2.45로 감소한다. raw yvRel도 약 -6.5~-7m/s다. 전반적 횡단 방향은 자연스럽지만 4.2→3.25는 50ms에0.95m, 다른 step보다 빠르고 기존 coherent 허용0.90m도 넘는다.
7. **association 실패:** 아니다. 같은 source42/track42/birth6624에 전부 연결되어 있다. 최신 y 수용이 보류된 상태의 geometry 선택 실패다. 다른 candidate로 재연결할 문제가 아니다.
8. **신뢰할 후보 여부:** 지속성 있는 raw 후보는 있다. 다만 950mm step이 실제 차량 이동인지 차체 반사점 변화/관측 오류인지 확정하지 못했다. 6724에는 v도 -0.22→-1.76으로 튀어 reject된다. 다른 LF source33은 d25/y5.75로 별도 물체 후보이고, 나머지는 거리/횡경계/속도 조건을 탈락해 대체 LF로 선택되지 않는다.
9. **영상:** 실제 차량 존재. 흰 승용차가 좌→우 횡단하며 자전거/보행자와 다른 차량도 있다. source42 궤적과 방향은 일치하지만 그 반사점의 정확한 대응·횡좌표 정답은 독립적으로 확정하지 않았다.

6719의 d는 양쪽15.80, v는 양쪽-0.22, maskTFT, good6714, temporal hold250ms로 같다. 6724는 d15.75/v=-0.22/maskTFF/good6714/hold200ms로 같다. pending은 두 버전 모두6719에서 `[5.0,4.5,4.2,3.25]`, 6724에서 `[4.5,4.2,3.25,2.75]`다. 0.95m step이 queue에 남는 동안 coherent가 false다. 6734에서 그 step이 window에서 빠지면 y2.1을 다시 수용한다. 이때 관측 v가 기존 저속 선택 조건을 탈락하므로 LF를 표시하지 않는다. 전체 차량42를 잃은 것은 아니며 8.016458s에는 두 버전 모두 RF42를 표시한다.

baseline의 LF wire42/birth6624/출력 y3.735317은 두 샘플 같다. 후는 LF wire 없음이다. lane probabilities는 첫 target `[0.06967,0.24726,0.45018,0.09739]`; 신규 side center는 None이다. 0.5m 초과 출력 jump 문제가 아니라 초기 선택 기회를 두 번 놓친 차이다.

저장된 표시 중심은 양쪽 `[2.53431,-2.87250]`이며 current valid는 둘 다 false다. stop/approach memory도 양쪽 모두 비어 있다. 저장된 중심이나 정차 memory 차이가 LF를 지운 것은 아니다.

실제 차량과 유효 raw 궤적이 있을 가능성 때문에 **정상 변화 A로 처리하지 않는다**. 반대로 보류된0.95m 관측의 정답과 차량 identity를 확정하지 못해 **실제 regression B도 확정하지 않는다**. 두 샘플 모두 **C**다.

## 채택 판단

현재 1단계는 accepted/pending의 오염 경로를 실제로 차단하며 유지한다. 이번 5샘플 모두 raw 관측이 존재하므로 단순 미검출 hold 연장으로 원인을 설명할 수 없다. association 재설계나 새 ghost/볼라드 필터를 제안할 근거도 없다.

**표시 무회귀 채택의 추가 판단에는 더 많은 근거가 필요하다.** 우선 `395/7`의 radar 반사점↔차량 대응 및 횡단 위치 라벨, 가능하면 같은 정차 횡단 상황의 연속 주행 샘플이 필요하다. 비슷한 로그 개수만 늘리는 것보다 이 정답 정보가 중요하다. 그 정보로395의 두 샘플을 A/B 중 하나로 확정한 후에만 최소 후속 수정 필요성을 판단한다. 이번에는 accepted history에 pending 과거 값을 승격하거나 hold/threshold/confirmation을 바꾸지 않았다.
