# 차선 인식 모델 R5 small s0 교체 — 2026-10-27

사용자 요청에 따라 `lane_strip_v4.onnx`와 메타데이터를 Claude 학습 세션의
`certlabel1026/data/training/R5_small_s0/model.{onnx,json}`에서 교체했다.
학습 원본은 `.analysis/ccnc-field-412-20261022/claude/` 아래에 보존한다.

- 이전: D1 seed2, SHA256 `23ff1bc749421977e400977e9d14e70dd2e30ab59391a0a59e890f7051f9afb7`.
- 교체: R5 small seed0, SHA256 `53fb243a2d89507c6e173d598bdd79526dadf2daec12a5678eb4249c0e8d5237`.
- ONNX 146,119바이트, 파라미터 34,079개.
- 입력: raw YCbCr 3×128×128, 반폭 1m, 오른쪽 미러링.
- 출력: 유형 5종(dashed/solid/dbl_near/dbl_far/unknown), 색상 3종(blue는 패딩).
- 임계 0.9와 기존 필터·게이트·표시 정책을 유지한다. 4종 모델 교체가 아니다.

원본 ONNX와 배포 파일의 SHA가 일치한다. 메타데이터에는 학습 원본 경로,
파라미터 수와 교체 이력을 추가했으며, 기기 검증 상태는 원본의 false를 유지한다.
모델 고정 SHA와 제품 메타데이터 검사도 새 모델에 맞췄다.

WSL OpenCV 4.13에서 모델 로딩·실제 추론과 관련 회귀 256개를 통과했다.
검사 범위는 StripModel v4, 모델 로더/daemon, 필터 변형, strip filter,
차선 변경 추론, 표시 정책이다. 한글 저장소 경로의 pycapnp 문제는 테스트에서만
스키마를 ASCII 임시 경로로 복사해 우회했다. 제품 스키마 코드는 바꾸지 않았다.

저장소 파일 교체 결과이며 콤마 설치·주행 검증·성능 재측정은 하지 않았다.
