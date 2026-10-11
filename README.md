# Theozyme FK Steering

**Inference-time Feynman–Kac steering toward catalytic geometry.**

RFdiffusion2 + LigandMPNN 기반 효소 설계에서, 생성 중 촉매 활성부위의 목표 기하(*theozyme*)를 평가하고 유망한 생성 경로에 계산을 더 배분하는 연구 프로젝트입니다.

핵심 질문은 **“FK steering이 unguided 생성과 사후 선별보다, 서열 설계·구조 예측 후에도 목표 촉매 기하를 보존하는 후보를 더 잘 만드는가?”**입니다. 현재 v0.1.1은 이 질문을 실험하기 위한 **CPU 기하 reward·FK 코어·합성 benchmark**를 제공합니다. 실제 RFdiffusion2 연결과 효소 설계 실험은 앞으로 검증할 단계입니다.

[빠른 실행](#quickstart) · [현재 결과](#results) · [방법](docs/METHOD.md) · [개발 로드맵](docs/NEXT_STEPS.md) · [Chromebook 안내](docs/CHROMEBOOK.md)

## 구현 상태

| 구성 요소 | 현재 지원 범위 |
|---|---|
| 기하 평가 | 거리·각·이면각, 주기 경계 처리, Kabsch 및 anchor 기준 RMSD |
| Theozyme reward | 원자 selector, fixed/generated 역할, 허용 오차를 둔 기하 penalty |
| FK / SMC | Difference potential, ESS 기반 systematic resampling, 계보 기록 |
| CPU benchmark | Unguided · Best-of-N · FK 비교, 독립 ensemble 반복, 결과 그림 |
| RFdiffusion2 | Adapter 미구현. Baseline launcher는 제공하지만 GPU 실행 미검증 |
| LigandMPNN / 구조 예측 | 자동 연동 미구현. 실제 효소 task 및 gradient guidance도 미구현 |

## 생성 도중 어디에 개입하는가?

![Implemented CPU FK loop and planned GPU integration](docs/assets/fk-overview.svg)

각 입자는 하나의 생성 상태입니다. 지정한 단계에서 예측 좌표를 채점하고, reward 변화로 가중치를 갱신합니다. 가중치가 소수 입자에 집중되면 resampling하여 다음 단계를 진행합니다. **Best-of-N은 생성이 끝난 뒤 후보를 선별하는 비교군**입니다.

향후 RFdiffusion2 adapter는 좌표뿐 아니라 self-conditioning, cache, 원자 대응 등 진행에 필요한 상태 전체를 보존해야 합니다. 도식의 점선 영역은 향후 연결할 GPU 단계입니다.

## Quickstart

### 1. CPU 환경 설치

Python **3.10 이상**이 필요합니다. Chromebook의 Crostini에서도 CPU 경로를 실행할 수 있으며 CUDA는 필요하지 않습니다.

```bash
git clone https://github.com/ju2501/theozyme_FK_steering.git
cd theozyme_FK_steering
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,report]'
```

이미 프로젝트 폴더가 있다면 해당 폴더에서 venv 생성부터 시작하세요. Crostini에서 `venv` 또는 `ensurepip` 오류가 나면 `sudo apt update` 후 `sudo apt install python3-venv`를 실행합니다.

기본 의존성은 NumPy입니다. `dev`는 pytest, `report`는 matplotlib을 추가합니다. 상세 환경 설정은 [CHROMEBOOK.md](docs/CHROMEBOOK.md)를 참고하세요.

### 2. 설치 확인과 단일 구조 채점

```bash
python -m unittest discover -s tests -v
python -m steer.cli.main score --task configs/theozyme/synthetic.json --pdb tests/fixtures/synthetic.pdb
```

합성 PDB fixture의 기대 결과는 **reward 0, valid true, passed true**입니다. `valid`는 채점 가능 여부이고, `passed`는 모든 기하 제약을 허용 오차 안에서 만족했는지입니다. 유효하지만 제약을 통과하지 못한 구조도 `score` 명령의 종료 코드는 0이므로 `passed` 값을 확인하세요.

### 3. 세 방법 비교

```bash
python -m steer.cli.main demo --config configs/runs/toy.json --out runs/first-demo --plot
```

이 데모는 **세 고정점과 한 이동점의 거리 제약**을 다룹니다. 단백질 생성 모델이나 실제 효소 실험은 아닙니다.

| 출력 파일 | 확인할 내용 |
|---|---|
| `runs/first-demo/comparison.png` | Reward와 좌표 다양성 비교 |
| `runs/first-demo/summary.json` | 방법별 평균, 통과율, 반복 간 통계 |
| `runs/first-demo/replicate_metrics.csv` | 각 독립 ensemble의 지표와 비용 |
| `runs/first-demo/selected.csv` | 선택된 좌표·reward·계보 |
| `runs/first-demo/manifest.json` | 설정, task/source hash, 실행 버전 |
| `runs/first-demo/replicates/` | 반복별 원본 pool과 FK resampling 기록 |

새 실험에는 새로운 `--out` 경로를 사용합니다. 중단된 실행은 동일 명령에 `--resume`을 붙여 **완료된 ensemble 반복 단위**로 재개합니다. 설정·task·core source·Python/NumPy 버전이 달라지면 재개를 거부합니다. trajectory 중간 재개는 지원하지 않습니다. v0.1.1의 반복별 checkpoint 식별자 검증 보완 사항은 [개발 로드맵](docs/NEXT_STEPS.md)에 기록되어 있습니다.

## Results

![Synthetic benchmark: reward and coordinate spread](results/comparison.png)

저장된 [설정](configs/runs/toy.json)은 seed 20260924부터 **20개 독립 ensemble**, ensemble당 **16 particles × 40 steps**, 최종 선택 **4개**를 사용합니다.

| 방법 | 평균 reward ↑ | 선택 좌표 사이 평균 거리 | 전체 제약 통과 / 선택 출력 | Sampler calls / ensemble |
|---|---:|---:|---:|---:|
| Unguided | −5.3235 | 2.4686 | 0 / 80 | 640 |
| Best-of-N | −1.5335 | 1.8139 | 0 / 80 | 640 |
| FK | −0.5823 | 1.1574 | 1 / 80 | 640 |

**이 합성 문제에서 FK는 평균 reward를 높였지만 좌표 다양성은 줄었고, 전체 제약 통과는 드물었습니다.** 기하 점수 개선이 실제 효소 활성 향상을 뜻하지는 않습니다.

- Unguided와 Best-of-N은 같은 baseline pool을 공유합니다. 세 방법의 최종 선택 수는 같습니다.
- Sampler 호출 수를 맞춘 비교입니다. FK의 추가 reward 평가 비용이 있어 동일 wall time 비교는 아닙니다.
- FK 입자들은 계보를 공유합니다. 불확실성은 80개 선택 출력이 아닌 **20개 독립 ensemble**을 단위로 계산합니다.

수치와 실행 정보는 [summary.json](results/summary.json), [manifest.json](results/manifest.json)에 있습니다. 분석 정의는 [EXPERIMENTS.md](docs/EXPERIMENTS.md)를 참고하세요.

## Reward와 FK의 핵심 규칙

제약별 오차에서 허용 오차를 뺀 양수 부분을 scale로 나누고, 가중 제곱 평균에 음수를 붙입니다. **Reward는 0이 최상**입니다.

```text
violation_j = max(0, abs(error_j) - tolerance_j) / scale_j
reward = -sum(weight_j * violation_j^2) / sum(weight_j)
```

- 거리 단위는 Å, 각·이면각은 degree입니다. 이면각 오차는 주기 경계에서 wrap합니다.
- 생성 원자가 포함된 제약만 reward에 들어갑니다. 고정 원자만의 제약은 diagnostics와 전체 통과 여부에 사용합니다.
- 모든 제약이 고정 원자만 포함하면 steering을 거부합니다. 역할 태그가 있어도 실제 예측에서 reward가 변하는지는 별도 확인해야 합니다.
- Missing atom이나 정의되지 않은 기하는 invalid입니다. FK에서 한 번 invalid가 된 계보는 제거 대상이며, 모든 입자가 invalid이면 오류로 종료합니다.
- Difference potential의 log-weight를 누적하며, 중간 resampling을 생략해도 가중치를 보존합니다. 양의 FK scale에서는 마지막 단계의 채점·resampling을 반드시 수행합니다. `scale=0`은 unguided 경로를 따릅니다.

수식과 경계 조건: [METHOD.md](docs/METHOD.md). 실제 원자 선택 규칙은 [synthetic task](configs/theozyme/synthetic.json)를 참고하세요. 현재 parser는 PDB만 지원하며 mmCIF와 자동 대칭 원자 대응은 지원하지 않습니다. 거리 제약만으로 chirality나 촉매 작용이 보장되지는 않습니다.

## 설정 변경과 GPU 확장

[configs/runs/toy.json](configs/runs/toy.json)에서 입자 수(`num_particles`), 반복 수(`replicates`), steering 강도(`scale`), 평가 간격(`every`), ESS 임계값(`ess_fraction`)을 조절할 수 있습니다. `start_fraction`과 `end_fraction`은 중간 평가 구간이며 최종 평가를 끄지 않습니다. 변경한 설정은 별도 파일과 새 출력 폴더로 보관하세요.

현재 `demo`는 synthetic sampler 전용입니다. 실제 PDB의 정적 채점에는 `score`를 사용하고, RFdiffusion2의 생성 도중 채점에는 실제 adapter 구현이 필요합니다.

| 실행 장소 | 권장 작업 |
|---|---|
| Crostini / CPU | Reward·원자 대응 검증, 테스트, 합성 실험, 결과 분석 |
| 랩 GPU | 고정한 upstream 버전의 baseline, adapter 검증, 실제 생성 실험 |
| Colab GPU | 동일 baseline의 설치·실행을 먼저 확인한 뒤 작은 반복 실험 |

RFdiffusion2 공식 평가 경로는 **RFdiffusion2 → LigandMPNN → Chai-1**입니다. 이 프로젝트에서 AF2를 선택한다면 ligand-relative 기하를 어디에서 평가할지 따로 정해야 합니다. 현재 제공하는 Colab 노트북은 CPU 데모용입니다. GPU 준비는 [GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md), 연결 계약은 [RFD2_INTEGRATION.md](docs/RFD2_INTEGRATION.md)에 있습니다.

## 다음 개발 목표

1. **실험 신뢰성:** checkpoint 반복 식별자 검증, 배포본 설치·실행 CI.
2. **실제 task 1개:** 출처가 있는 참조 기하, 원자 매핑, 좌표 perturbation 검증.
3. **RFdiffusion2 adapter:** baseline 재현, 중간 reward 가변성, 완전한 상태 복제, λ=0 대조.
4. **단계별 평가:** 생성 직후·LigandMPNN 후·구조 예측 후의 기하 보존을 추적.
5. **효과 비교:** 동일 선택 수와 기록한 계산 비용으로 Unguided / Best-of-N / FK 비교.

각 단계의 완료 조건과 주간 작업 단위는 [NEXT_STEPS.md](docs/NEXT_STEPS.md)에 정리했습니다. Gradient guidance와 여러 반응으로의 확장은 단일 task 경로를 검증한 뒤 진행합니다.

## 코드와 문서 탐색

| 경로 | 역할 |
|---|---|
| [`src/steer/geometry/`](src/steer/geometry/) · [`rewards/`](src/steer/rewards/) | 기하 계산, task 정의, reward |
| [`src/steer/particles/`](src/steer/particles/) | FK 가중치·ESS·resampling |
| [`src/steer/hooks/`](src/steer/hooks/) | Sampler 계약, toy sampler, RFD2 연결 경계 |
| [`src/steer/io/`](src/steer/io/) · [`cli/`](src/steer/cli/) | PDB/JSON 입출력과 실행 명령 |
| [`tests/`](tests/) · [`configs/`](configs/) | 테스트, 합성 fixture, 실행 설정 |
| [`pipelines/rfd2/`](pipelines/rfd2/) | GPU 미검증 baseline launcher |
| [`docs/`](docs/) · [`DECISIONS.md`](DECISIONS.md) | 상세 방법, 개발 계획, 설계 결정 기록 |

검증 이력은 [BUILD_REPORT.md](docs/BUILD_REPORT.md)에 있습니다. Python 3.12에서 로컬 unittest 54개와 CPU 데모를 확인했습니다. 2026-10-11에는 [GitHub Actions](https://github.com/ju2501/theozyme_FK_steering/actions/runs/38138798702)에서도 Python 3.10/3.12의 설치·pytest·score·CPU demo가 통과했습니다. OS·NumPy 버전 간 bitwise 동일성은 보장하지 않습니다.

## References and license

- [FK-RFdiffusion](https://github.com/ErikHartman/FK-RFdiffusion): RFdiffusion을 대상으로 한 FK steering 선행 구현. 본 프로젝트는 RFdiffusion2의 촉매 기하 평가와 연결을 연구합니다.
- [RFdiffusion2 공식 문서](https://rosettacommons.github.io/RFdiffusion2/readme_link.html): baseline, 출력 표현, LigandMPNN·Chai-1 평가 경로.
- [LigandMPNN](https://github.com/dauparas/LigandMPNN): ligand-aware 서열 설계와 side-chain packing.

이 저장소에는 upstream 코드·가중치·컨테이너를 포함하지 않습니다. 자체 코드의 라이선스는 아직 선택하지 않았습니다.
