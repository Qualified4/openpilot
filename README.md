[원본 carrotpilot README → ajouatom/openpilot · carrot-wip](https://github.com/ajouatom/openpilot/blob/carrot-wip/README.md)

# carrotpilot · CCNC HDA1

[ajouatom/openpilot](https://github.com/ajouatom/openpilot)의 `carrot-wip`을 기반으로, CCNC HDA1 계기판 표시와 차선 인식, 깜박이 동작, 로그 업로드 등을 추가·수정한 저장소입니다. 작업 브랜치는 **`ccnc-hda1`**입니다.

공통 기능·지원 장치·차량별 하네스와 설치 조건은 [원본 README](https://github.com/ajouatom/openpilot/blob/carrot-wip/README.md), [CarrotPilot 사용 설명서](https://g4iwnl.gitbook.io/carrotpilot), [원본 Wiki](https://github.com/ajouatom/openpilot/wiki)를 참고하세요. 아래는 원본 `carrot-wip` 대비 이 브랜치의 주요 추가·변경 사항입니다.

## 추가·변경 기능

### CCNC 계기판 표시

| 기능 | 추가·변경 내용 | 안내 |
|---|---|---|
| 모델 기반 차선 표시 | 모델이 인식한 좌우 차선 위치와 곡률을 계기판에 반영합니다. 일반·정밀 모드, 차선변경 방향 강조와 전환 애니메이션, 곡률 표시 기준을 선택할 수 있습니다. | [CCNC 설정](docs/user/ko/settings.md#ccnc-display) |
| 차선 강조·배경색 | 가감속·주행 모드에 따른 차선 강조와 정지·출발 배경색을 표시합니다. 배경색은 계기판에 따라 보이지 않을 수 있습니다. | [CCNC 설정](docs/user/ko/settings.md#ccnc-display) |
| 주변 차량 표시 | HDA1에서 레이더 기반 전방·측면 차량을 표시하고, 차량 위치 보정과 선택형 레이더 지도를 제공합니다. | [표시 구조와 적용 범위](docs/ccnc_front_radar_detection.md) |
| 측면 차량의 전·후방 인계 | 옆 차량이 앞에서 뒤로 지나갈 때 표시를 이어줍니다. 후방은 전방 관측과 상대 움직임을 이용한 가상 표시이며, 후방 레이더 감지를 뜻하지 않습니다. | [인계 방식과 검증 범위](docs/ccnc_rear_handover_20261018.md) |
| 차선 종류·색상 인식 | 카메라 영상의 차선 영역을 별도 소형 모델로 분석해 점선·실선과 흰색·노란색을 CCNC에 표시합니다. 현재 포함된 모델은 **UB R5 small s0**입니다. | [기능 설정](docs/user/ko/settings.md#ccnc-display) · [모델과 비교 결과](docs/lane_model_ub_s0_20261108.md) |
| CCNC 디버그 도구 | Carrot Web에서 차량 슬롯·차선 표시 값을 조절해 계기판 표시를 확인합니다. 정차·기어·제어 상태 등의 실행 조건을 검사합니다. | [CCNC Debug](docs/ccnc_debug_20261018.md) |

### 차로 변경·깜박이

| 기능 | 추가·변경 내용 | 안내 |
|---|---|---|
| 노란 실선 차로변경 대기 | 차선 인식 기능과 `LaneChangeYellowSolid`가 켜져 있으면, 유효하게 확정된 노란 실선 쪽의 차로변경을 대기시킵니다. 판단 보류·만료된 결과로는 차단하지 않습니다. | [차로변경 조건](docs/user/ko/lane-change.md#lanechangeyellowsolid--차선변경-노란-실선-차단) |
| CAN-FD 차로변경 깜박이 | 차선 접근이나 모델 차로변경 시작에 맞춰 깜박이를 켜는 별도 옵션을 제공합니다. 순정 SPAS 프레임 뒤에 붙여 보내는 피기백 방식과 비교용 송신 방식을 선택할 수 있습니다. | [CAN-FD 설정](docs/user/ko/settings.md#차로변경-깜빡이-자동-can-fd-시험) |
| 원터치 깜박이 유지 | 운전자의 원터치 요청을 차로변경 대기·진행 중 이어주는 옵션을 제공합니다. 변경 진행도·취소·제한 시간 등에 따라 유지 요청을 끝냅니다. | [CAN-FD 설정](docs/user/ko/settings.md#차로변경-깜빡이-자동-can-fd-시험) |
| 깜박이 테스트 | Carrot Web의 도구에서 지원 경로의 깜박이 명령을 시험할 수 있습니다. | [CAN-FD 설정](docs/user/ko/settings.md#차로변경-깜빡이-자동-can-fd-시험) |

CAN-FD 깜박이 옵션은 지원 차량과 송신 방식에 따라 동작·제약이 다릅니다. 아이오닉5 PE와 쏘나타 2024의 확인 범위를 다른 차종으로 확대해 해석하지 마세요. 각 옵션의 조건과 알려진 문제는 설정 안내에서 확인할 수 있습니다.

### 로그·Road Viewer

- **Road Viewer 연결·전송**: Carrot Web에서 Road Viewer와 연결하고, 여러 주행에서 선택한 구간을 한 번에 전송합니다. 구간별 완료 처리, 중단 후 이어 올리기, 완료 응답 유실 시 수신 결과 확인을 지원합니다.
- **업로드 화면 개선**: 접힌 주행의 선택 구간도 전송 목록에 포함하고, 화면을 이동한 뒤에도 작업 진행 상태를 확인할 수 있도록 개선했습니다.
- Road Viewer는 [별도 홈 서버 프로젝트](https://github.com/Qualified4/roadviewer-ha)입니다. 영상 변환·보관·재생은 서버 측에서 처리합니다. [기기 연결·전송 및 검증 안내](openpilot/selfdrive/carrot/server/road_viewer/TESTING.md)

### 주행 응답·실행 안정성

- **선행차 가속 반응**: 선행차가 서서히 가속하거나 차간거리가 벌어지는 상황의 가속 예측과 MPC 반응 조건을 조정했습니다. [설정 안내](docs/user/ko/settings.md) · [구현](openpilot/selfdrive/controls/lib/longitudinal_preview.py)
- **정지 차간거리**: 선행차 뒤 저속 최종 정지에서 설정 차간거리에 접근하도록 MPC 거리 비용을 조정했습니다. [적용 조건](openpilot/selfdrive/controls/lib/longitudinal_stop_gap.py)
- **레이더 처리 배치**: CCNC 처리 부하가 큰 `card`와 `radard`를 서로 다른 CPU 코어에 배치했습니다. [현재 배치](openpilot/selfdrive/carrot/radar/radard_dpath.py)
- **외부 모델 응답 처리**: eGPU 작업 진행과 늦게 도착한 응답을 처리하는 로직을 보완했습니다. [구현](openpilot/selfdrive/modeld/precompiled_runner.py)
- **저장소 업데이트 상태**: 업데이트 작업의 잠금 소유자·진행 단계를 구분하고, 네트워크 실패 후 재시도 정보를 제공합니다. [구현](openpilot/selfdrive/carrot/server/services/auto_update.py)

## 지원 조건과 적용 방법

1. **차량과 구성을 확인합니다.** 이 브랜치의 CCNC 확장은 현대·기아 CAN-FD의 해당 계기판 경로를 대상으로 합니다. HDA1 레이더 차량 표시가 HDA2에 그대로 적용되지는 않습니다. 차종별 지원과 검증 범위는 각 기능 문서를 확인하세요.
2. **사용 브랜치를 확인합니다.** 이 저장소의 추가 기능은 `Qualified4/openpilot`의 **`ccnc-hda1`** 기준입니다. 원본 README의 `carrot-wip` 설치 주소는 원본 배포용이며 이 브랜치의 설치 주소가 아닙니다.
3. **Carrot Web에서 설정합니다.** 기기와 같은 네트워크에서 `http://기기-IP:7000`에 접속합니다. CCNC 옵션은 **설정 → 차량·하드웨어 → CANFD·HDA**에서 확인할 수 있습니다. [전체 설정 안내](docs/user/ko/settings.md)
4. **변경의 반영 조건을 확인합니다.** 모델·프로세스·Panda 변경은 Git 업데이트 후 전체 기기 재시작이 필요할 수 있습니다. 옵션별 기본값과 반영 시점은 설정 설명을 따릅니다.

차선 분류와 레이더 지도 등에는 실험 기능이 포함됩니다. 인식 오류와 표시 유지·지연이 생길 수 있으며, 로그 재생·단위 테스트 결과가 모든 차량의 실제 동작을 검증한 것은 아닙니다. 차선 인식은 별도 차로변경 차단 옵션을 통해 제어 조건에도 사용되므로, 계기판 표시와 차로변경 조건을 함께 확인하세요.

## 문서·출처

- [한국어 설정 안내](docs/user/ko/settings.md) · [English settings guide](docs/user/en/settings.md)
- [CCNC 표시 기술 안내](docs/ccnc_front_radar_detection.md) · [개발·검증 이력](docs/ccnc_validation_history.md)
- [원본 carrotpilot README](https://github.com/ajouatom/openpilot/blob/carrot-wip/README.md) · [사용 설명서](https://g4iwnl.gitbook.io/carrotpilot) · [Wiki](https://github.com/ajouatom/openpilot/wiki)

이 저장소는 [ajouatom의 carrotpilot](https://github.com/ajouatom/openpilot)과 [comma.ai의 openpilot](https://github.com/commaai/openpilot), 각 프로젝트 기여자의 작업을 기반으로 합니다. comma.ai의 공식 배포판이 아니며, 원본과 다른 차량 제어·표시 동작을 포함합니다. 운전자는 항상 전방을 주시하고 즉시 차량을 제어할 준비를 해야 합니다. 사용·설치 전 원본의 지원 조건과 주의사항도 확인하세요.

라이선스는 [LICENSE](LICENSE) 및 각 구성 요소의 라이선스 표시를 따릅니다.
