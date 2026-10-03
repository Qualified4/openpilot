# CCNC 횡이력 연구 사이클 종료 — 2026-10-03

## 최종 결정과 범위

사용자 결정에 따라 **1단계 accepted/pending 횡이력 격리만 채택한다.** C/D 횡출력 정책과 fixed-object rejection은 채택하지 않는다. 이번 정리는 production 동작을 추가 변경하지 않으며 커밋하지 않는다. 판단 불가 사례를 해소했다거나 볼라드를 해결했다고 주장하지 않는다.

## 1단계 production 수정

파일은 `opendbc_repo/opendbc/car/hyundai/hyundaicanfd_ccnc_extension.py`, 함수는 `_CcncTemporalTracks.update()`다. 기존 횡이력/횡속도 갱신 블록을 `if accepted[1]:` 안으로 옮겼다. 신규 state와 parameter는 0개다. threshold, confirmation, hold, pending 확정 규칙, 후보 선택, slot, smoothing은 변경하지 않았다.

원래는 y가 rejected/pending이어도 d/v가 accepted이면 raw y를 횡이력에 넣고 prediction용 vy를 다시 계산했다. 따라서 거부된 관측이 다음 판단과 missing coast를 오염시킬 수 있었다. 수정 후에는 accepted y만 accepted history와 vy를 갱신한다. rejected y는 기존 pending evidence에만 남고 기존 정상 vy를 보존한다. pending이 기존 규칙으로 확정되면 최신 accepted 관측을 기록하며, 이전 pending 값들을 소급해 넣지 않는다. accepted y 상황의 기존 d/v 관련 clear/reset 처리는 유지한다.

### Synthetic 및 테스트

재현: 0.25s y=-3.4 accepted → 0.30s -2.9 pending → 0.35s -2.4 pending → 0.40s missing. 수정 전 vy=6.667m/s, coast y=-3.0667m; 수정 후 vy=0, coast y=-3.4m. 종료 시 동일 trace를 재실행해 확인했다.

`tools/ccnc/test_radar_display.py`에 4개 함수/5개 parameterized case를 추가했다. rejected 두 관측 이후 missing(정지/기존 정상 vy 각각), single-frame spike 후 복귀, 기존 pending confirmation을 통한 지속 횡이동 수용, accepted 정상 vy 이후 missing을 검증한다. 수정 전에는 4 fail/1 pass, 수정 후 전체 display **270 pass**다. 정상 accepted vy를 0으로 만드는 수정이 아니다.

종료 검증에서 Windows schema/Params 격리 wrapper와 기존 `tools.ccnc.run_tests`의 관련 전체 묶음은 **630 pass (최종 재실행 2.95s)**다. 이 묶음은 CCNC display/차량 CAN/관련 UI·문서 검사이며 openpilot 저장소의 모든 테스트를 뜻하지 않는다. display 단독은 **270 pass (0.82s)**다. 채택하지 않은 shadow 정책 테스트는 이 production 검증 묶음에 포함하지 않는다.

### 실제 로그 replay

수정 전 SHA256은 `3db9dcb8695980bc3615d8a69d18e49fb43ca11e2da05b0a34b10d28bf9cf984`, 채택 버전은 `5cf5150797fc8227ff7b6789c546ae89bbd9bd48a752c641bad051cdc28e5760`이다. 종료 정리는 이 채택 소스를 바꾸지 않았다. 아래는 해당 두 버전으로 이미 수행한 동일 입력 replay의 최종 기록이며 종료 작업에서 모든 raw 로그를 다시 실행한 결과는 아니다.

| 20개 로그 / 23,889 출력 지표 | 수정 전 | 수정 후 |
|---|---:|---:|
| 완료된 detect 공백 사건 | 232 | 231 |
| 짧은 공백 (80–650ms) | 140 | 138 |
| detect=0 slot 샘플 | 35,441 | 35,446 |
| display birth 변경 | 102 | 103 |
| slot 변경(재등장 포함) | 62 | 61 |
| 연속 표시 중 slot 변경 | 28 | 28 |
| 0.5m 초과 횡출력 step | 31 | 31 |
| 최대 횡출력 step | 0.75m | 0.75m |
| pending 진입 | 2,102 | 2,101 |
| pending 확정 | 1,448 | 1,644 |
| rejected y의 accepted history 오염 | 4,124 | 0 |
| rejected y 이후 vy 변경 | 993 | 0 |

신규 detect=0 샘플은 5개, 제거된 detect=0 샘플은 0개다. 공백 사건 수와 공백 프레임 수는 서로 다른 지표다. pending 확정 증가도 confirmation threshold 변경이 아니라 입력 이력 수정의 결과다. 추가 8개 로그에서는 detect 공백·birth·slot·0.5m 초과 step 총수가 같았다. 356/2의 최대 횡변화 0.330→0.434m와 7개 coast 차이를 물리적 개선으로 단정하지 않았다.

실제 cut-in/out 시작 시각 ground truth는 없어 물리적 반응 지연을 측정하지 못했다. 395/7 source42의 RF 최초 표시 8.016458s와 확인된 비교 구간의 표시 시작 시각은 동일했다. 39e/19에서 확인 가능한 한 reacquisition은 양쪽 93.349ms였으며 전체 재획득 통계로 일반화하지 않는다. 최초 12개 로그에서는 accepted nonzero-vy coast 383→395샘플로 정상 coast가 유지됐다. raw 로그에서 synthetic의 잘못된 coast chain 전체가 재현됐다고 주장하지 않는다.

후속 공간 데이터셋의 78개 고유 segment / 92,495 출력 검증은 **같은 1단계 코드의 frozen A/B 동등성 검사**다. 원래 버그 버전과의 78개 로그 개선 비교가 아니다. 이 검사의 rejected-history/vy 오염은 0이다. 옵션 OFF의 기존 14,290 CAN 출력 동등성 기록도 유지한다.

### 추가 공백 5샘플 감사

각 사건 전후 최소 1초, raw 전체 후보와 내부 상태·prediction·hold·선택·wire/slot·geometry를 조사했다. 4,801 출력 시각에서 기존 replay와 일치했다. [상세 감사](ccnc_stage1_detect_gap_audit_20261003.md)를 보존한다.

| 사건 | 분류 | 원인 |
|---|---|---|
| 38a/16 36.676683s LF | A 정상 변화 | corrected last-good 400ms 만료 후 동일 ID 새 track 확인 미완료 |
| 38a/16 36.722006s LF | A 정상 변화 | 같은 publication 재사용으로 새 확인 증거 없음 |
| 393/1 56.307794s RF | A 정상 변화 | 최신 accepted y=-4.75가 기존 RF 경계 -4.46172 밖 |
| 395/7 7.267240s LF | C 판단 불가 | accepted prediction 4.20은 LF 경계 4.03160 밖; 오염된 이전 vy는 3.80으로 이동 |
| 395/7 7.317228s LF | C 판단 불가 | 같은 pending 및 선택 이력에 따른 경계 여유 차이 |

**A 3 / B 확정 실제 regression 0 / C 2**다. 다섯 시각 모두 영상 장면에는 실제 차량이 존재한다. A는 실제 차량 부재를 뜻하지 않는다. 395/7은 이미 표시되던 source42의 중단이 아니라 초기 LF birth 차이다. 실제 횡단 차량과 자연스러운 진행 방향은 보이지만 빠른 반사점 이동의 정답을 확정하지 못했다. B=0을 무회귀 증명으로 해석하지 않는다. 이 한계를 유지한 채 사용자 결정으로 1단계만 채택한다.

### 성능과 알려진 한계

같은 14,430 입력에 warm-up 후 전/후 순서를 교대한 5회 PC helper wall-time 측정의 중앙값을 사용한다. 평균 **0.122218→0.121327ms**, p99 **0.297300→0.296500ms**, run별 maximum의 중앙값은 **0.6534→0.7157ms**다. 측정 중 가장 큰 maximum은 **2.5950→1.3578ms**다. maximum 변동을 알고리즘 개선/악화 효과로 해석하지 않는다. 이전 약 0.278ms 수치와 다른 실행 환경/측정은 직접 비교하지 않았다. 순차 실행의 변동을 교대 측정으로 확인했으며 유의한 성능 regression 증거는 없었다.

종료 작업에서 성능을 재측정하지 않았다. production 소스는 당시 측정과 byte-identical하다. Windows thread CPU clock의 거친 해상도는 tail 평가에 쓰지 않았다. 이 결과는 전체 controller/ARM 실차 CPU나 계기판 반응 검증이 아니다. noisy accepted 관측, geometry 불확실성, ID association, 395/7의 판단 불가 2샘플과 볼라드 오표시는 해결하지 않았다.

## 채택하지 않은 실험

| 실험 | 가설과 결과 | production 미채택 이유 | 다시 검토하려면 필요한 정보 |
|---|---|---|---|
| C transient-jump suppression | accepted prediction에서 벗어난 pending은 즉시 따르지 않고 지속 motion은 빠르게 따른다. side >0.5m step 18→28, 전체 왕복 1→5. 합성 지속 drift 반응 300→150ms지만 짧은 false drift excursion 0→0.90m. | 일부 proxy 반응 개선보다 jitter/false-motion 반례가 큼. slot 1프레임 재배치도 발생. | 실제 cut-in onset 및 source/body 대응 GT, 짧은 coherent false drift와 실제 motion의 구분 근거 |
| D motion-confirmed fast-follow | 확인된 motion evidence를 하위 보정에 재사용한다. side step 18→29, 왕복 1→5. 측정 가능한 motion proxy20개 중1개만 약50ms 빨라짐; 합성 false drift excursion 0→0.75m. | 불필요한 출력 이동 증가. 395/7 source42 최초 표시도 그대로. | 실제 횡이동의 지연 GT와 중복 confirmation이 원인이라는 근거, false-motion 반례의 독립 검증 |
| shadow fixed-object classifier | 복귀 jump/d-v/geometry/lifetime/motion을 reason bit로 조합하고 낮은 confidence는 abstain한다. 2,134 사건 중 MOTION206/STATIC0/REFLECTION2/AMBIGUOUS1,926. 기존 motion345 의심0, reflection122 중2 검출. | STATIC은 정차차량 보호 때문에 분류 불가. 영역 proxy에서 실제 차량 FP1, fixed TP1/FN2로 precision 부족; 반사 이상은 비차량을 뜻하지 않음. | source-level 영상 대응, 경계 정차차량·트럭 복수 반사·혼합 장면의 독립 라벨 |
| offline feature discovery | 117 feature 및 2/3개 조합의 반복 차이를 찾는다. 확정 장면 proxy vehicle7/fixed3, 개별 source는 모두 UNKNOWN. density 신호는 보이나 world speed AUC0.571, noise0.619; 정차차량 반례로 d/v AUC1→0.905. | 작은 표본, 장소/속도/ROI 선택 편향. 실제 접근차량의 boundary persistence가 구조물보다 높은 반례. | 조건을 맞춘 충분한 독립 source GT 및 정차/저속 실제 차량 반례 |
| spatial density/boundary | 배열 밀도+경계 지속성이 독립 로그에서 반복되는지 확인한다. 78segment를 discovery20/동일route확장34/다른route evaluation24로 분리. 장면264 및 raw제안154. 평가 density AUC0.771, boundary persistence0.526. | scene 신호는 있으나 source GT0. 비슷한 속도/거리 scene pair11, 같은 lateral region pair0. full3s fixed history는1개뿐. 독립 객체≥30/≥30 목표 미달. | 실제 source↔object 대응, 같은 lateral 영역/속도/거리/환경의 vehicle/fixed pairs, 독립 장소 holdout |

C/D의 detect 공백 총수는 동일하지만 slot 시각까지 동일하지 않았다(61→63회 변경). 원래 motion-like345 중313은 사전 표시가 없고, 나머지32 중 실제 proxy 반응 측정은20개뿐이었다. 물리적 cut-in 지연 성능으로 일반화하지 않는다. one/two-frame spike는 baseline부터 억제됐다.

shadow classifier의 395/7 source42는 MOTION_SUPPORTED로 보호했고 정차차량은 AMBIGUOUS로 남겼다. 이 결과는 production reject precision의 증명이 아니다. 공간 평가에서 frozen offline probe의 fixed scene flag 1/16, vehicle flag0/12는 **객체 recall/false rejection이 아니다**. source-level ground truth가 없으므로 두 지표는 미측정이다. 추가 dense-selected stress 집합과 평가 후 source-agnostic ROI 기록은 탐색 결과이며 독립 evaluation으로 합산하지 않는다.

## 볼라드 문제의 현재 결론

**liveTracks와 현재 model geometry만으로 fixed object와 실제 정차/저속 차량을 안정적으로 구분할 근거가 부족하다.** 낮은 속도, boundary persistence, lifetime, source 안정성에는 실제 차량 반례 또는 큰 겹침이 있다. scene-level spatial density에는 신호가 있으나 radar source ↔ 실제 object ground truth가 없고 ROI/장소 편향이 남는다.

production rejection 개발은 중단한다. source-level ground truth를 확보할 때만 재검토한다. 1단계 수정 때문에 볼라드 표시가 우연히 바뀌더라도 볼라드 해결책으로 해석하지 않는다. 이번 종료 작업은 추가 데이터 수집이나 feature 탐색을 수행하지 않았다.

## 도구·테스트·최종 diff 정리

production runtime에 shadow import/call이 없다. active tree의 `tools/ccnc/replay_shadow.py`, `shadow_classification.py`, `shadow_output_policy.py`와 두 shadow 테스트를 제거하고 byte/hash를 확인해 로컬 archive로 옮겼다. C/D 또는 fixed-object rejection을 production behavior로 전제한 테스트는 남기지 않는다. 원래 `run_tests.py`, `pytest_plugin.py`, `pytest.ini`와 display tests를 유지한다.

중복 실험 문서 5개는 이 종료 보고서로 통합하고 원문 및 기존 history append를 로컬 archive에 보존했다. 1단계 five-gap 감사는 별도 문서로 남긴다. 분석 자료는 runtime 의존성이 없는 frozen 근거이며 설치나 기본 production 테스트가 import하지 않는다. 필요시 역사적 재현을 위해 별도 scratch에 복원할 수 있으나 활성 classifier/tool로 유지하지 않는다.

로컬 근거 위치:

- `.analysis/archive/2026-10-03/ccnc-accepted-history/`: before/after source, synthetic, replay, paired timing
- `ccnc-generalization/`, `ccnc-five-gaps/`: 20개 replay, 사건 context, 5샘플 상세 감사
- `ccnc-shadow/`, `ccnc-output-policies/`, `ccnc-feature-discovery/`, `ccnc-spatial-dataset/`: 채택하지 않은 실험 기록
- `ccnc-cycle-close/`: 퇴역 파일/문서, SHA 기록, 종료 테스트와 diff 검토

최종 production diff는 위 1단계 guard만이며 regression test 70줄과 필요한 문서가 나머지다. 신규 필터, 객체 rejection, C/D output policy, threshold/hold/confirmation/slot 변경이 없다. **커밋하지 않았다.** 실차 display/CPU와 판단 불가 2사건은 알려진 한계로 남긴다.
