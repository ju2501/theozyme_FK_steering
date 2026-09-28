# Theozyme-Steered Enzyme Design

**Status: v0.1.1 CPU research core. Real RFdiffusion2 steering is not implemented or GPU-validated.**

RFdiffusion2 + LigandMPNN 계열 효소 설계에서, 재학습 없이 촉매 기하 reward로 샘플링을 유도하는 연구 프로젝트입니다. 현재 릴리스는 기하 계산·reward·FK 엔진·합성 benchmark를 실행할 수 있습니다. **아래 그림은 단백질이나 효소의 실험 결과가 아닙니다.**

![Synthetic geometry benchmark, not enzyme results](results/comparison.png)

## 1. Motivation

연구 질문: 동일 계산 예산에서 inference-time FK가 unguided 생성 및 사후 선별보다 **downstream에서 촉매 기하를 보존하는 설계**를 더 많이 만드는가?

기여 후보는 효소용 원자 대응·기하 reward, RFdiffusion2 중간 상태와의 연결, downstream 검증입니다. 기존 FK 구현은 RFdiffusion을 감싸며 ProteinMPNN과 packing/relaxation으로 reward를 계산합니다. 이를 RFdiffusion2에 바로 호환된다고 가정하지 않습니다. [FK 논문](https://arxiv.org/html/2511.09216), [공식 구현](https://github.com/ErikHartman/FK-RFdiffusion).

기하 개선을 효소 활성 향상으로 해석하지 않으며, 최초 적용이라는 우선권 주장도 하지 않습니다.

## 2. Method

- 거리 단위: Å, 각도·이면각: degree. 이면각 오차는 주기 경계에서 wrap합니다.
- 제약별 정규화 위반량: `v = max(0, abs(error) - tolerance) / scale`.
- Reward: 생성 원자가 포함된 제약의 `-sum(weight * v**2) / sum(weight)`; 0이 최상입니다.
- 고정 원자만 포함된 항은 diagnostics에 남기고 reward에서는 제외합니다. 모든 항이 고정이면 steering을 거부합니다.
- FK는 **difference potential만** 구현합니다. 초기 log-potential은 0이며, adaptive ESS 때문에 resampling을 건너뛰어도 log-weight를 보존합니다.
- 중간 스케줄과 별개로 마지막 단계에는 terminal reward와 최종 resampling을 적용합니다. `scale=0`이면 모든 steering을 생략합니다.
- `Sampler`의 initialize/advance/predict/clone 계약으로 모델 의존성을 격리합니다. 모델 객체는 공유해도 각 입자의 mutable cache는 공유하면 안 됩니다.

수식·경계 조건은 [docs/METHOD.md](docs/METHOD.md), 실제 결합 요건은 [docs/RFD2_INTEGRATION.md](docs/RFD2_INTEGRATION.md)를 참조하세요.

## 3. Installation

Crostini에서 Python 3.10 이상을 사용합니다. 기본 의존성은 NumPy뿐이며 Torch·CUDA·Apptainer 없이 실행됩니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,report]'
python -m unittest discover -s tests -v
```

`pytest -q`로도 같은 테스트를 수집할 수 있습니다. 이번 작업 환경에서는 Python 3.12로 **unittest 54개 통과**를 확인했습니다. Pytest 자체 실행 및 GitHub Actions 실행은 아직 확인하지 않았습니다. 새 venv에서의 온라인 의존성 설치는 사용자 환경에서 확인해야 합니다. 빌드 환경에서는 기존 NumPy를 사용한 editable 설치를 검증했습니다.

conda를 선호하면 `env/steer-cpu.yml`을 사용할 수 있습니다. 해당 conda 설치 경로는 아직 검증하지 않았습니다. 상세 안내: [Chromebook·Codex·GitHub](docs/CHROMEBOOK.md). 저장소 구성 방향: [참고 저장소와 적용 계획](docs/REPOSITORY_STYLE.md).

## 4. Quickstart

프로젝트 최상위에서:

```bash
python -m steer.cli.main score --task configs/theozyme/synthetic.json --pdb tests/fixtures/synthetic.pdb
python -m steer.cli.main demo --config configs/runs/toy.json --out runs/my-demo --plot
```

첫 명령은 작은 **합성 PDB**의 reward 0과 `passed: true`를 출력합니다. 두 번째는 약간의 확률적 이동을 하는 점 하나를 세 고정점과의 거리 제약으로 유도합니다. 이것은 RFdiffusion2를 모사하는 생물학적 모델이 아닙니다.

중단 후에는 같은 명령에 `--resume`을 붙입니다. 완료된 **ensemble 반복 단위**부터 이어갑니다. config·task·core source·Python/NumPy 버전이 달라지면 재개를 거부합니다. trajectory 중간 재개는 미구현입니다. 기존 출력 디렉터리를 덮어쓰지 않습니다.

## 5. Usage

### Task 정의

`configs/theozyme/synthetic.json`처럼 원자 selector와 fixed/generated 역할을 지정합니다. 실제 효소용 숫자를 임의로 제공하지 않습니다. 선택한 반응·참조 구조·원자 매핑에서 기하를 확인해 새 task를 작성해야 합니다.

PDB parser는 첫 MODEL을 사용하고, 원자별로 occupancy가 높은 altloc을 선택합니다. 중복·모호한 매핑은 오류입니다. 실제 실험 입력은 하나의 일관된 conformer로 정리하세요. mmCIF·hybrid-36 번호·자동 대칭 원자 매핑은 지원하지 않습니다.

### 랩 GPU / Colab

```bash
python scripts/probe_runtime.py --out runtime_probe.json
```

RFdiffusion2 checkout과 SIF가 있으면 `--upstream /actual/RFdiffusion2 --sif /actual/image.sif`를 추가합니다. 이 도구는 환경을 읽고 JSON을 남기며 모델을 설치하거나 실행하지 않습니다.

`pipelines/rfd2/baseline.py`는 공식 baseline 호출을 구성하는 **미검증 GPU launcher**입니다. 기본은 dry-run이며 pinned commit을 요구합니다. `--execute`를 지정해야 실행합니다. 실제 steering 명령인 `scripts/run_steered.sh`는 현재 미지원 오류로 종료합니다. [GPU 안내](docs/GPU_RUNBOOK.md).

## 6. Results

현재 확인한 결과는 CPU synthetic benchmark뿐입니다. config: `configs/runs/toy.json`, 20개 독립 ensemble, 입자 16개, 40 steps, 최종 선택 4개입니다.

| 방법 | 평균 reward ↑ | 선택 점 사이 평균 거리 | sampler 호출 / ensemble |
|---|---:|---:|---:|
| Unguided | -5.3235 | 2.4686 | 640 |
| Best-of-N | -1.5335 | 1.8139 | 640 |
| FK | -0.5823 | 1.1574 | 640 |

FK가 이 toy reward를 높이는 동시에 다양성이 줄어드는 동작을 확인했습니다. **효소에 대한 유효성은 이 결과로 판단할 수 없습니다.** 제약 전체 통과는 FK에서 80개 출력 중 1개뿐이며, 실제 효소 성공률을 뜻하지 않습니다. 입자들은 계보를 공유하므로 독립 표본이 아닙니다.

세 방법은 sampler 호출 수를 맞췄습니다. FK의 reward 평가는 더 많으므로 동일 wall time을 의미하지 않습니다. Unguided와 best-of-N은 같은 baseline pool을 공유합니다. 불확실성은 입자 대신 독립 ensemble을 단위로 계산합니다. [결과·비용 해석](docs/EXPERIMENTS.md).

## 7. Repository Layout

```text
src/steer/geometry/    거리·각·이면각·정합·task schema
src/steer/rewards/     촉매 기하 reward와 invalid 처리
src/steer/particles/   difference potential·ESS·systematic resampling
src/steer/hooks/       sampler 계약·synthetic sampler·미구현 RFD2 경계
src/steer/io/          JSON·PDB·원자 매핑
src/steer/cli/         score·demo·report 명령
pipelines/rfd2/       upstream baseline launcher (GPU 미검증)
scripts/              테스트·데모·환경 probe
tests/                CPU 검증과 합성 fixture
configs/              재현 가능한 task·run 설정
results/              이번 합성 benchmark의 소규모 결과
docs/                 실험·연결·환경·다음 작업 안내
notebooks/            Colab에서 CPU 데모를 실행하는 보조 노트북
```

## 8. Reproducibility

`results/manifest.json`에 config, task SHA256, core source SHA256, 패키지·Python·NumPy 버전을 기록했습니다. `selected.csv`는 선택 출력, `replicate_metrics.csv`는 독립 반복별 평가, `summary.json`은 집계입니다. 새 실행에는 `replicates/*.json`으로 원본 pool과 resampling 계보도 남습니다.

서로 다른 NumPy/BLAS/운영체제 사이의 bitwise 일치까지 보장하지 않습니다. 모델 실험에서는 추가로 upstream SHA·weights/SIF hash·runtime·GPU 종류·총 비용을 기록해야 합니다.

## 9. Limitations & Future Work

- **미구현:** 실제 RFdiffusion2 adapter, LigandMPNN/AF2 자동 실행, 실제 효소 task, GPU 실험, gradient guidance.
- Synthetic sampler는 stochastic coordinate process입니다. RFdiffusion2 전파가 결정론적이면 입자 복제만으로 새 trajectory가 생기지 않습니다.
- fixed/generated 태그만으로 reward 가변성이 보장되지는 않습니다. 실제 중간 예측에서 확인해야 합니다.
- 거리만으로는 chirality·방향성·촉매 작용을 보장할 수 없습니다. 현재 synthetic task에는 의도적으로 거리 항만 있습니다.
- Kabsch RMSD와 ligand-anchor RMSD를 구분합니다. 기하 평가가 곧 activity 평가가 아닙니다.
- 실험에서 missing atom·invalid 결과를 조용히 삭제해 통과율을 높여서는 안 됩니다.
- 공개 benchmark의 결론을 내려면 실제 downstream·독립 반복·compute-matched control이 필요합니다.

진행 순서: [NEXT_STEPS.md](docs/NEXT_STEPS.md).

## 10. Citation & Acknowledgements

- [RFdiffusion2 공식 문서](https://rosettacommons.github.io/RFdiffusion2/readme_link.html)
- [RFdiffusion2 환경 설치](https://rosettacommons.github.io/RFdiffusion2/installation.html)
- [FK steering 논문](https://arxiv.org/html/2511.09216), [구현](https://github.com/ErikHartman/FK-RFdiffusion)
- [LigandMPNN](https://github.com/dauparas/LigandMPNN)

이 배포본에는 upstream 코드·가중치·컨테이너를 포함하지 않습니다. 자체 코드의 공개 라이선스는 아직 지정하지 않았습니다. 공개 배포 전 저장소 소유자가 라이선스를 선택하고, 실제 포함할 외부 자산의 조건을 확인합니다.
