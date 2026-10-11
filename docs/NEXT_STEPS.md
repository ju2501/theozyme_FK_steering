# Development roadmap

이 계획은 CPU v0.1.1을 실제 RFdiffusion2 효소 steering 실험으로 연결하기 위한 순서다. 아래 완료 조건을 만족한 뒤 다음 단계로 이동한다. 기간은 GPU 접근성과 실제 upstream 코드에 따라 달라진다.

## 1. CPU 실험의 신뢰성부터 보완

**첫 PR: resume checkpoint의 출처와 반복 번호 검증.**

`src/steer/cli/benchmark.py`는 run manifest를 검증하지만 반복별 JSON을 불러올 때 내부 식별자를 검사하지 않는다. `0000.json`을 `0001.json`으로 복사하면 같은 결과를 두 독립 반복으로 집계한다. 기존 정상 실행이 잘못됐다는 증거는 없으며, 파일 혼입 시 통계가 왜곡될 수 있는 검증 공백이다.

- Checkpoint에 run identity와 replicate/seed를 넣고 resume에서 확인한다.
- Rows/summaries의 replicate, arm, slot 구성과 개수, seed 일치를 검사한다.
- 정상 resume와 빠진 반복 재계산은 기존 결과를 유지해야 한다.
- 다른 반복·seed·run의 checkpoint와 손상된 JSON은 파일명을 포함한 명확한 오류로 거부한다.
- 스키마 변경 시 기존 결과와 호환 여부를 명시한다. 기존 결과의 통계나 source hash를 덮어쓰지 않는다.

완료 조건: 중복 checkpoint 반례를 재현하는 테스트가 수정 전 실패하고 수정 후 통과하며, 정상 재개 결과가 유지된다.

**그다음 PR: 실제 배포 경로 검증.** `scripts/build_release.py`로 ZIP을 만든 뒤 임시 폴더에 추출하고, 해당 소스를 설치해 score/demo를 실행하는 CI를 추가한다. 기존 packaging 단위 테스트와 source checkout 테스트를 보완하는 경로다. 첫 원격 push 이후 Actions의 Python 3.10/3.12 결과를 확인한다.

## 2. 실제 참조 task 하나를 정하고 CPU에서 채점 검증

코드 위치: `configs/theozyme/`, `src/steer/io/pdb.py`, `src/steer/geometry/`, `tests/test_reward_io.py`.

가능하면 RFdiffusion2 공식 baseline 예제와 같은 참조 task를 골라 baseline·원자 매핑·downstream 평가를 공유한다. 임의의 거리 숫자로 반응을 정의하지 않는다.

| 기록 | 완료 조건 |
|---|---|
| 출처 | 참조 구조/기하의 출처, chain/residue/atom 선택, 목표 값과 허용 오차의 근거 |
| 원자 역할 | 고정 ligand/motif와 생성 원자 구분, 각 점수가 어떤 좌표에서 계산되는지 명시 |
| 기하 대조 | 작은 손계산 또는 별도 기하 도구와 거리·각·이면각 결과 비교 |
| Perturbation | 생성 원자의 변형에 대응하는 penalty 변화, 강체 회전·이동 불변성 |
| 경계 조건 | 원자 누락, altloc, 대칭, 이면각 wrap, chirality 및 ligand-relative 평가 |

중간 구조에 필요한 원자가 없다면 좌표를 0으로 채우지 않는다. 평가 가능한 단계와 표현을 정하거나 근거 있는 재구성 방법을 검증한다. 고정 motif 자체만으로 구성된 reward가 상수라면 가중치를 키우기 전에 목표와 원자 매핑을 수정한다.

## 3. 고정한 RFdiffusion2 baseline → 실제 adapter

코드 위치: `scripts/probe_runtime.py`, `pipelines/rfd2/baseline.py`, `src/steer/hooks/rfd2_hook.py`.

1. Lab GPU에서 upstream SHA·가중치/컨테이너 식별자·GPU·환경을 기록한다. Colab만 가능하면 같은 baseline을 먼저 확인한다.
2. FK 없이 upstream 예제 하나를 끝까지 실행하고 config·로그·출력 구조를 남긴다.
3. 실제 소스에서 step 함수, 시간 방향, 예측 구조와 noisy state, cache와 self-conditioning, 원자/잔기 대응을 확인한다. 가상의 hook API를 미리 정하지 않는다.
4. `initialize / advance / predict / clone` 계약에 연결한다. 공유 가능한 모델 가중치와 입자별 mutable state를 구분한다.
5. 중간 예측을 저장해 raw geometry, invalid 비율, 입자 간 reward 분산을 확인한다. 초기 단계의 missing representation과 실제 invalid geometry를 구분한다.
6. 동일 seed의 λ=0 경로가 upstream baseline과 일치하는지 비교한다. GPU 비결정성이 있으면 비교 허용 범위와 원인을 명시한다.
7. 복제한 자식 상태를 바꿔도 부모와 형제의 cache가 변하지 않는지 확인한다. 전파가 확률적이면 RNG 분리를, 결정론적이면 복제 후 같은 경로가 유지되는 제약을 확인한다.

완료 조건: 작은 단일-task 실행에서 baseline·λ=0·양의 FK가 모두 종료되고 reward trace, ESS, 계보, 시간, peak VRAM이 저장된다. 이 단계만으로 성능 우위를 주장하지 않는다.

## 4. Downstream 기하 보존을 평가

RFdiffusion2 공식 평가 경로는 RFdiffusion2 → LigandMPNN → Chai-1이다. 공식 문서는 idealization과 LigandMPNN packing 사이에서 motif rotamer가 달라질 수 있음을 설명한다. 따라서 생성 직후의 reward만으로 설계 성공을 정의하지 않는다.

[공식 출력·평가 설명](https://rosettacommons.github.io/RFdiffusion2/readme_link.html#pipeline-metrics)

- 단계별로 stable design ID와 atom/residue mapping을 전달한다.
- 촉매 잔기 identity 및 고정 조건을 명시하고 LigandMPNN 출력에서 다시 확인한다.
- 생성 직후, packing 후, 구조 예측 후에 같은 기하 지표를 계산한다.
- Ligand-relative 평가의 정합 원자와 평가 원자를 구분한다. 활성부위만 재정합해서 ligand에 대한 위치 오차를 숨기지 않는다.
- AF2를 사용할 경우 어떤 입력과 예측 모드인지 기록한다. Ligand가 없는 예측에서 ligand-relative 기하를 주장하려면 별도로 정의하고 검증한 좌표 대응이 필요하다.
- 예측 실패와 missing atom을 결과에서 제거하지 말고 전체 시도 수와 단계별 실패율을 기록한다.

완료 조건: 같은 design을 단계별로 추적하는 지표 표와 실패 사례가 남는다. Gradient guidance와 복잡한 에너지 항의 추가는 이 경로가 작동한 뒤 검토한다.

## 5. 작은 비교 실험부터 시작

| 비교군 | 정의 |
|---|---|
| Unguided | Baseline 생성 pool에서 무작위 K개 선택 |
| Best-of-N | 동일 baseline pool을 최종 geometry reward로 선별해 K개 선택 |
| FK | 생성 중 steering 후 K개 선택 |

모든 군에 동일한 downstream 파이프라인과 설정을 적용한다. 독립 ensemble 반복을 통계 단위로 삼는다. 주요 결과는 사전 정의한 downstream 기하 통과율, 실패율, 다양성, 모델/reward/downstream 호출 수 및 총 GPU 시간으로 둔다. Score-only와 downstream 지표를 구분한다.

먼저 모델 호출 수를 맞춘 pilot로 연결 가능성을 확인한다. 비용까지 포함한 효과를 주장하려면 별도로 측정한 wall time/GPU 예산을 맞춰 비교한다. 초기 toy 결과가 실제 task에서도 반복될 것이라고 가정하지 않는다.

첫 ablation은 입자 수와 steering 강도에 한정한다. N을 늘릴 때 비용도 함께 증가함을 기록한다. Schedule, 다른 potential, gradient guidance를 동시에 바꾸지 않는다.

## 주간 업데이트 단위

한 주에 하나의 질문과 검증 결과를 남긴다. 일정이 밀려도 미완료 기능을 완료로 표시하지 않는다.

| 작업 단위 | GitHub에 남길 결과 |
|---|---|
| Checkpoint 검증 | 실패 반례 → 수정 → 회귀 테스트 |
| 실제 task | 원자 매핑·근거·기하 대조 표 |
| Baseline / adapter | 고정 버전, 실행 로그, λ=0·복제 상태 검증 |
| 비교 pilot | config, 독립 반복 수, 비용, 성공·실패와 다양성 |

PR 본문에는 해결할 문제, 선택한 방법의 이유, 실제 검증, 남은 한계를 적는다. 로그와 결과가 없는 대규모 추상화보다 작은 재현 가능한 변경을 우선한다. 한 번의 pilot이 한 주에 끝나지 않으면 상태 매핑이나 baseline 재현처럼 검증 가능한 단계로 나눈다.
