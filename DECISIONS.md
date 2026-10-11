# DECISIONS.md — Theozyme-Steered Enzyme Design (스캐폴딩 설계 결정)

> 상태: **확정 초안 (code 세션 인계용)**
> 범위: 코드 작성 이전의 구조/환경/모듈/문서 골격 결정만 다룬다.
> 프로젝트 한 줄 정의: RFdiffusion2 + LigandMPNN 계보의 효소 설계 파이프라인에
> **inference-time steering (Feynman–Kac / gradient guidance)** 를 얹어,
> 샘플링을 촉매 기하(theozyme) reward 쪽으로 유도한다.
> 기여 포인트: binder 설계에서 검증된 FK steering을 **효소 활성부위**로 확장.

---

## 0. 설계를 지배하는 제약 (모든 결정의 근거)

| # | 제약 | 결과 |
|---|---|---|
| C1 | 로컬 = Crostini **CPU**, 무거운 실행 = 랩 **GPU** | 코드는 "CPU에서 개발·테스트 → GPU로 전송" 가능해야 함. GPU 없이 import/테스트 가능한 경계가 필요 |
| C2 | RFdiffusion2 / RFdiffusionAA / heme_binder 는 **apptainer 이미지** | 이 코드베이스들의 python 환경은 **내가 통제할 수 없다**. conda로 재현 불가 |
| C3 | steering 은 **디노이징 루프 내부**에 들어간다 | steering 코드가 컨테이너 **안**에서 import 돼야 함 |
| C4 | eval / 분석은 순수 python | 컨테이너 밖, conda로 자유롭게 |
| C5 | 최종 산출물은 GitHub README | 재현 경로가 문서로 설명 가능해야 함 |

**C2 + C3 의 충돌이 이 프로젝트 구조의 핵심 문제다.**
→ 결정 D2-1 (아래) 이 그 답이고, 나머지 구조는 대부분 거기서 파생된다.

---

## 1. 레포 폴더 구조

### D1-1. 단일 레포 + `src/` 설치형 패키지 + 실행층 분리 (확정)

```
theozyme-steer/
├── README.md
├── DECISIONS.md              # 이 파일 (세션마다 append-only 로 갱신)
├── pyproject.toml            # src/steer 를 설치 가능 패키지로
├── .gitignore
│
├── env/
│   ├── steer-cpu.yml         # 개발/steering 로직 (Crostini)
│   ├── eval.yml              # 구조 평가·분석 (랩 서버 CPU)
│   ├── containers.md         # .sif 출처/버전/빌드 명령 기록
│   └── containers/           # .sif 심볼릭 링크만. 실제 파일은 git 제외
│
├── configs/                  # 코드 아닌 모든 파라미터
│   ├── base.yaml
│   ├── theozyme/             # 촉매 기하 정의 (반응별 1파일)
│   │   └── ser_his_asp.yaml
│   ├── steering/             # FK/guidance 하이퍼파라미터
│   │   ├── fk_default.yaml
│   │   └── guidance_default.yaml
│   └── runs/                 # 실험 1회 = config 1개 (재현 단위)
│
├── src/steer/                # ★ 순수 python 라이브러리 (의존성 최소)
│   ├── geometry/             # theozyme 기하 계산
│   ├── rewards/              # 기하 → 스칼라 reward
│   ├── particles/            # FK / SMC resampling
│   ├── guidance/             # gradient-based guidance
│   ├── hooks/                # RFd2 디노이징 루프 삽입 지점 어댑터
│   ├── io/                   # PDB/모티프/트래젝토리 I/O
│   └── cli/                  # 엔트리포인트
│
├── pipelines/                # ★ 컨테이너 호출 래퍼 (bash + 얇은 python)
│   ├── rfd2/
│   ├── ligandmpnn/
│   ├── af2/
│   └── common/               # bind-mount 경로, 공통 apptainer exec 조립
│
├── eval/                     # 설계 결과 평가 (컨테이너 밖, 순수 python)
│   ├── metrics/
│   ├── filters/
│   └── report/
│
├── scripts/                  # 사람이 직접 치는 진입점
│   ├── run_baseline.sh
│   ├── run_steered.sh
│   ├── slurm/
│   └── sync_to_gpu.sh
│
├── notebooks/                # 탐색·그림. 재현 경로에는 포함하지 않음
├── tests/                    # pytest, GPU/컨테이너 없이 전부 통과해야 함
├── data/                     # (gitignore) 입력 모티프, 레퍼런스 PDB
└── runs/                     # (gitignore) 실행 산출물
```

**근거**

- `src/steer` 를 **설치 가능한 순수 라이브러리**로 떼어낸 것이 전부의 핵심이다.
  컨테이너 안(GPU)에서도, 컨테이너 밖(CPU)에서도, 노트북에서도 같은 코드를 import 한다.
  "steering 로직"이 이 프로젝트의 **유일한 학술적 기여**이므로, 가장 이식성 높고
  가장 테스트하기 쉬운 자리에 둔다.
- `pipelines/` 는 "남의 코드를 부르는 껍데기"만 모은다. 여기에 로직이 새어 들어가면
  컨테이너 버전이 바뀔 때마다 기여 코드가 깨진다. **경계를 물리적으로 강제**한다.
- `eval/` 을 `src/` 안에 넣지 않은 이유: eval 은 무거운 의존성(PyRosetta, MDAnalysis,
  pymol 등)을 끌어올 가능성이 높다. `src/steer` 의 의존성 최소화(D2-1)를 오염시키면 안 된다.
- `configs/runs/` 1파일 = 실험 1회. 논문/README 에서 "이 결과는 이 config" 로 인용 가능.

**고려했으나 버린 대안**

| 대안 | 버린 이유 |
|---|---|
| A. flat 구조 (`steering.py`, `eval.py`, …) | 초반엔 빠르지만 컨테이너 안/밖 의존성 경계를 표현할 수 없다. C2·C3 와 정면 충돌 |
| B. 멀티 레포 (steer-core / pipelines 분리) | 경계는 가장 깨끗하지만 1인 사이드 프로젝트에 CI·버전 핀 오버헤드가 과하다. README 하나로 끝내려는 목표(C5)와도 안 맞음 |
| C. RFdiffusion2 fork 안에 직접 구현 | 가장 빨리 돌아가지만 upstream 업데이트 시 병합 지옥이고, "내 기여가 뭔지" GitHub 에서 보이지 않는다 — 이 프로젝트 목적상 치명적 |
| D. `src/` 없이 top-level `steer/` | 사소한 차이지만 `src/` 레이아웃이 "설치 안 한 채로 우연히 import 되는" 사고를 막아준다. 컨테이너 안에서 PYTHONPATH 로 붙일 때 특히 |

---

## 2. 환경 전략

### D2-1. `src/steer` 는 **numpy 외 하드 의존성을 갖지 않는다** (확정 · 가장 중요)

- 허용: `numpy` (그리고 런타임에 이미 있는 `torch` 는 **선택적** import)
- 금지: hydra, pandas, biotite, MDAnalysis 등을 `src/steer` 최상위에서 import 하는 것
- torch 는 `guidance/` 서브패키지 안에서만, 지연 import 로 사용

**근거**: steering 코드는 내가 통제하지 못하는 컨테이너 python(C2) 안에서 import 돼야 한다(C3).
거기엔 numpy 와 torch 는 반드시 있지만 그 외에는 아무것도 보장되지 않는다.
의존성 1개 추가가 곧 "컨테이너 안에서 안 돌아감" 이다.
config 파싱·로깅 같은 편의 기능은 전부 `cli/` 와 `pipelines/` 쪽 바깥 레이어로 밀어낸다.

**대안**: 컨테이너를 내가 다시 빌드해서 의존성을 넣는다 → 가능하지만 이미지 재빌드 비용이 크고,
랩 GPU 환경에서 이미지 교체 권한/디스크가 매번 문제가 된다. 의존성을 줄이는 쪽이 훨씬 싸다.

### D2-2. conda env 는 **2개**, 컨테이너는 **conda로 감싸지 않는다** (확정)

| env | 어디서 | 무엇 |
|---|---|---|
| `steer-cpu` | Crostini (로컬) | python, numpy, torch(cpu), pytest, hydra/omegaconf, matplotlib. `pip install -e .` |
| `eval` | 랩 서버 (CPU 노드) | biopython/biotite, MDAnalysis, pandas, (필요시 PyRosetta), 리포트 생성 |
| — | 랩 GPU | **conda env 없음.** apptainer 이미지가 곧 환경 |

**근거**
- env 를 3개 이상으로 쪼개면 1인 프로젝트에서 관리 비용만 늘고 실익이 없다.
- 반대로 1개로 합치면 PyRosetta/MDAnalysis 류가 로컬 Crostini 설치를 망가뜨린다. 2개가 균형점.
- GPU 쪽에 conda env 를 두면 "컨테이너 python 과 conda python 중 뭐가 실행됐나" 를
  디버깅하는 데 시간을 다 쓰게 된다. **GPU에서는 conda를 아예 쓰지 않는다**는 규칙이 더 싸다.

### D2-3. apptainer 이미지는 **레포 밖**, 레포에는 경로·해시·빌드 레시피만 (확정)

- `.sif` 는 git 에 절대 들어가지 않는다 (수 GB).
- 실제 위치: 랩 공용 경로 (예: `/opt/containers/` 또는 스크래치). 환경변수 `STEER_SIF_DIR` 로 참조.
- `env/containers.md` 에 이미지별 **출처 URL / 커밋 / 빌드일 / sha256** 을 표로 기록 → 재현성의 핵심.
- `env/containers/` 에는 심볼릭 링크만 두어 로컬에서 경로를 통일한다.

**근거**: 이미지 자체는 버전 고정만 되면 되고, 문서화된 해시가 실제 재현성을 만든다(C5).

**대안**: Docker/Dockerfile 로 직접 관리 → 랩 GPU 클러스터에서 root 권한이 없는 경우가 대부분이라 비현실적.

### D2-4. 컨테이너 안에서 `src/steer` 는 **bind-mount + PYTHONPATH** 로 주입 (확정)

```
apptainer exec --nv \
  -B $REPO/src:/opt/steer_src:ro \
  -B $RUNS:/runs \
  --env PYTHONPATH=/opt/steer_src \
  $SIF python /opt/rfd2/run_inference.py ...
```

**근거**: 이미지를 재빌드하지 않고 steering 코드만 매번 최신으로 갈아끼울 수 있다.
개발 사이클(로컬에서 고치고 → GPU에서 바로 실행)이 이미지 빌드 없이 닫힌다.
`:ro` 로 마운트해 컨테이너가 소스를 오염시키지 못하게 한다.

**대안**: `pip install` 을 컨테이너 안에서 매 실행 → 네트워크/권한 의존, 실행마다 느려짐. 기각.

---

## 3. 앞으로 만들 스크립트 / 모듈 목록

> 각 줄 = 파일 1개 = 대체로 code 세션 1개 단위. 우선순위는 P0 > P1 > P2.

### 3.1 `src/steer/geometry/` — 촉매 기하

| 파일 | 역할 | P |
|---|---|---|
| `theozyme.py` | theozyme 정의(잔기 조합 + 목표 거리/각도/이면각)를 담는 데이터클래스 | P0 |
| `measure.py` | 좌표 → 기하량(거리/각/이면각) 계산. numpy만 사용 | P0 |
| `align.py` | 설계 구조에 theozyme 템플릿을 정합(Kabsch)시켜 RMSD 산출 | P0 |
| `catalytic_frames.py` | 잔기 국소 좌표계 정의 — 회전 자유도를 다룰 때 필요 | P1 |

### 3.2 `src/steer/rewards/` — 기하 → 스칼라

| 파일 | 역할 | P |
|---|---|---|
| `base.py` | `Reward` 인터페이스 (`__call__(coords, ctx) -> float`), 합성·가중 규약 | P0 |
| `theozyme_reward.py` | theozyme 기하 위배량을 매끄러운 감쇠 함수로 변환한 주 reward | P0 |
| `composite.py` | 여러 reward 가중합·스케줄링(노이즈 레벨별 가중 변화) | P1 |
| `penalties.py` | 클래시, 매립도, 비현실적 루프 등 보조 페널티 | P1 |

### 3.3 `src/steer/particles/` — FK steering

| 파일 | 역할 | P |
|---|---|---|
| `fk_steering.py` | FK 입자 집합 관리: 전파 → 가중 → resample 루프 | P0 |
| `resample.py` | systematic/multinomial resampling, ESS 계산 | P0 |
| `potentials.py` | reward → FK potential 변환 (difference / max / sum 방식) 및 온도 λ | P0 |
| `state.py` | 입자 상태 직렬화 — 중단·재개 및 사후 분석용 | P1 |

### 3.4 `src/steer/guidance/` — gradient guidance

| 파일 | 역할 | P |
|---|---|---|
| `grad_guidance.py` | 미분가능 reward 로 디노이징 방향 보정 (torch, 지연 import) | P1 |
| `diff_geometry.py` | `measure.py` 의 미분가능 torch 버전 | P1 |

### 3.5 `src/steer/hooks/` — RFd2 결합부 (가장 리스크 높음)

| 파일 | 역할 | P |
|---|---|---|
| `rfd2_hook.py` | RFdiffusion2 디노이징 루프에 콜백을 꽂는 어댑터. **upstream 의존 지점을 여기 한 곳에 격리** | P0 |
| `trajectory.py` | 스텝별 좌표/가중/ESS 기록 → jsonl 로 덤프 | P0 |
| `schedule.py` | 어느 timestep 구간에서 steering 을 켤지 결정 | P1 |

### 3.6 `src/steer/io/` · `cli/`

| 파일 | 역할 | P |
|---|---|---|
| `io/pdb.py` | 의존성 없는 최소 PDB 파서/라이터 (컨테이너 안에서 써야 하므로 biopython 불가) | P0 |
| `io/motif.py` | 모티프/활성부위 스펙 파싱, 잔기 인덱스 매핑 | P0 |
| `cli/run_steered_rfd2.py` | 컨테이너 안에서 실행되는 메인 엔트리 | P0 |
| `cli/score_only.py` | 기존 PDB 에 reward 만 매겨보는 디버그용 | P0 |

### 3.7 `pipelines/`

| 파일 | 역할 | P |
|---|---|---|
| `common/apptainer.py` | bind-mount·env·sif 경로를 한 곳에서 조립 | P0 |
| `rfd2/run.sh` | RFdiffusion2 호출 (baseline / steered 양쪽) | P0 |
| `ligandmpnn/run.sh` | 백본 → 서열 설계 | P0 |
| `af2/run.sh` | 설계 서열 재예측 (self-consistency 용) | P0 |
| `common/stage_io.py` | 스테이지 간 파일 규약(디렉토리 레이아웃, 네이밍) 고정 | P1 |

### 3.8 `eval/`

| 파일 | 역할 | P |
|---|---|---|
| `metrics/self_consistency.py` | scRMSD / pLDDT / pAE — 표준 성공 기준 | P0 |
| `metrics/theozyme_rmsd.py` | 최종 구조의 촉매 기하 정확도 (이 프로젝트의 주 지표) | P0 |
| `metrics/diversity.py` | 설계 다양성 — steering 이 모드 붕괴를 일으키는지 검사 | P0 |
| `filters/pipeline.py` | 필터 체인 적용 후 통과율 집계 | P1 |
| `report/make_report.py` | baseline vs steered 비교표·그림 생성 | P1 |

### 3.9 `scripts/` · `tests/`

| 파일 | 역할 | P |
|---|---|---|
| `scripts/run_baseline.sh` | steering 없는 대조군 1커맨드 실행 | P0 |
| `scripts/run_steered.sh` | steering 실험 1커맨드 실행 | P0 |
| `scripts/sync_to_gpu.sh` | 로컬 → 랩 GPU rsync (runs/ 제외) | P0 |
| `scripts/slurm/*.sbatch` | 배치 제출 템플릿 | P1 |
| `tests/test_geometry.py` | 합성 좌표로 기하 계산 검증 | P0 |
| `tests/test_resample.py` | resampling 통계적 정합성 (가중 → 기대 빈도) | P0 |
| `tests/test_reward_smoke.py` | 실제 PDB 1개로 reward 파이프 end-to-end | P0 |
| `tests/test_hook_mock.py` | 가짜 디노이징 루프로 hook 계약 검증 — **GPU 없이 FK 로직 전체를 테스트** | P0 |

> `test_hook_mock.py` 가 사실상 이 프로젝트의 개발 속도를 결정한다.
> 가짜 디퓨전 루프(노이즈에서 목표 기하로 수렴하는 장난감 문제)를 하나 만들어 두면
> Crostini CPU 에서 FK steering 의 대부분을 검증할 수 있다. **P0 중에서도 먼저 만들 것.**

---

## 4. README 목차 (뼈대)

> 세션이 진행될 때마다 해당 섹션만 채운다. 섹션 번호는 고정.

```markdown
# Theozyme-Steered Enzyme Design

> 한 줄 요약 + 대표 그림 1장 (baseline vs steered 촉매 기하 분포)

## 1. Motivation
   1.1 효소 de novo 설계에서 촉매 기하가 병목인 이유
   1.2 재학습 없이 샘플링만 바꾸는 접근(inference-time steering)의 매력
   1.3 기존 binder steering 연구와 이 작업의 차이

## 2. Method
   2.1 배경: RFdiffusion2 / LigandMPNN 파이프라인
   2.2 Theozyme reward 정의
   2.3 FK steering (입자·가중·resampling)
   2.4 Gradient guidance (선택적)
   2.5 스케줄: 어느 timestep 에 개입하는가

## 3. Installation
   3.1 요구사항 (GPU, apptainer)
   3.2 conda 환경 2종
   3.3 컨테이너 이미지 준비 (출처·해시 표)
   3.4 설치 검증 (`pytest`)

## 4. Quickstart
   4.1 최소 예제 1커맨드
   4.2 출력물 읽는 법

## 5. Usage
   5.1 theozyme config 작성법
   5.2 steering 하이퍼파라미터
   5.3 baseline 재현
   5.4 랩 GPU / SLURM 에서 돌리기

## 6. Results
   6.1 평가 지표 정의
   6.2 baseline vs steered
   6.3 ablation (입자 수, λ, 스케줄)
   6.4 실패 사례와 한계

## 7. Repository Layout
## 8. Reproducibility  (config → 결과 대응표, 시드, 이미지 해시)
## 9. Limitations & Future Work
## 10. Citation & Acknowledgements  (upstream 라이선스 명시)
```

**근거**: Motivation → Method → 설치 → 실행 → 결과 순서는 연구 코드 레포의 사실상 표준이고,
대학원 지원 시 평가자가 **30초 안에 기여를 파악**할 수 있어야 한다.
그래서 (a) 대표 그림을 맨 위에, (b) Method 를 Installation 보다 앞에 둔다
(라이브러리가 아니라 연구 산출물이므로).

**대안**: 설치를 맨 앞에 두는 툴 스타일 README → 사용자가 실행하는 도구라면 맞지만,
이 레포의 1차 독자는 "무엇을 했는지 보는 사람"이므로 기각.

---

## 5. 다음 세션 진입 순서 (code 세션 인계 사항)

1. **세션 A**: `pyproject.toml` + 디렉토리 골격 + `io/pdb.py` + `geometry/` + `tests/test_geometry.py`
2. **세션 B**: `rewards/` + `tests/test_reward_smoke.py` — 기존 효소 PDB 로 reward 가 말이 되는지 확인
3. **세션 C**: 가짜 디퓨전 루프 + `particles/` + `tests/test_hook_mock.py` — **CPU에서 FK 검증 완료**
4. **세션 D**: `pipelines/` + `hooks/rfd2_hook.py` — 여기서 처음 GPU 필요
5. **세션 E**: `eval/` + baseline 실행
6. **세션 F**: 결과 정리 + README 채우기

> 세션 C 까지는 GPU 없이 진행된다. 이 순서를 지키면 GPU 시간을 디버깅에 낭비하지 않는다.

---

## 6. 아직 열려 있는 질문 (결정하지 않음)

- 대상 반응/theozyme 을 무엇으로 할 것인가 (Ser-His-Asp 가 제일 무난하지만 차별성은 낮음)
- FK potential 형태: difference vs max — 논문마다 갈리고, 효소 reward 의 희소성 때문에 실험 필요
- gradient guidance 를 끝까지 할지, FK 만으로 마무리할지 (스코프 관리 차원)
- LigandMPNN 단계에도 steering 을 걸지, diffusion 단계에만 걸지


---

## 7. 2026-09-24 구현 결정 (기존 초안에 대한 append-only 갱신)

- D3-1: 첫 릴리스 범위는 CPU core와 합성 benchmark. 실제 RFdiffusion2/MPNN/AF2 실행은 미구현이며 GPU 검증을 주장하지 않는다.
- D3-2: FK difference potential만 구현. adaptive ESS에서 log-weight 누적을 유지하고 terminal correction/resampling을 명시한다. max/sum/guidance는 후속 과제.
- D3-3: 각 원자에 fixed/generated 역할을 요구한다. 고정 원자만의 항은 diagnostics이며 모든 항이 고정이면 steering을 거부한다.
- D3-4: YAML/Hydra 대신 표준 JSON을 첫 config 형식으로 사용한다. NumPy 외 core 의존성이 없다.
- D3-5: torch 없이 stdlib unittest로도 검증한다. 현재 eval은 외부 무거운 환경이 필요하지 않으므로 별도 eval conda 환경 생성은 유보한다.
- D3-6: CPU 환경은 venv 또는 conda 선택. 랩은 Apptainer 우선, Colab은 실제 호환성 확인 후 별도 native profile 가능. 랩 GPU의 conda 금지를 모든 Colab 실행에 확대하지 않는다.
- D3-7: 기존 세션 A/B/C 구현을 완료했으나 RFd2 연결 D는 source/runtime 확보 전까지 완료로 표시하지 않는다. 임의 hook API를 만들어 GPU 지원으로 포장하지 않는다.
- D3-8: 사후 선별 비교군 추가. 같은 최종 출력 수 외에 model/reward/downstream 호출과 GPU 시간을 기록한다. 입자 대신 독립 ensemble 반복을 통계 단위로 한다.
- D3-9: 중단 복구는 ensemble 반복 단위로 지원. config/task/source/runtime identity mismatch를 거부한다. trajectory 중간 재개는 미구현.
- D3-10: 합성 fixture는 생물학적 구조가 아니며 Ser-His-Asp 참조 숫자를 임의로 작성하지 않는다. 실제 task는 upstream baseline에서 검증 후 확정한다.
- D3-11: gradient 및 일반적인 단백질 평가에는 라이브러리를 무리하게 늘리지 않는다. 실제 downstream 도구 선택은 lab/Colab runtime에서 확인한다.
- D3-12: 원본 README는 docs/README_original.md로 보존. 자체 코드의 공개 라이선스는 아직 지정하지 않았다.


## 8. 2026-09-27 배포 수정

- D4-1: v0.1.0 ZIP은 이름이 runs인 모든 폴더를 제외해 configs/runs/toy.json도 누락했다. v0.1.1은 최상위 /runs만 제외하는 재현 가능한 release builder를 사용한다.
- D4-2: Git에도 설정 파일이 빠지지 않도록 .gitignore의 runs/를 /runs/로 고친다.
- D4-3: source-tree 테스트 외에 배포 ZIP을 별도 폴더에 추출하고 그 소스를 설치해 테스트·데모를 확인한다. 이를 Crostini 자체 검증으로 부르지 않는다.
- D4-4: UvA의 튜토리얼, heme의 단계별 파이프라인, RFdiffusion의 실행 예제를 참고하되 실제 지원 수준을 문서에 표시한다.


## 9. 2026-10-08 README 및 개발 순서 정리

- D5-1: README를 연구 질문, 구현 상태, 개요 그림, CPU quickstart, 결과 해석, 다음 단계 순서로 정리한다. 실행 위치와 결과 파일, valid/passed의 차이를 명시한다.
- D5-2: GPU 성능이나 새 효소 실험을 주장하지 않는다. CPU v0.1.1 결과는 기존 값을 유지한다.
- D5-3: 다음 코드 변경은 반복별 checkpoint identity 검증으로 시작한다. 실제 task의 원자 매핑·reward 가변성과 upstream baseline을 확인한 뒤 adapter를 구현한다.
- D5-4: 공식 RFdiffusion2 평가 경로인 LigandMPNN/Chai-1과 프로젝트에서 고려하는 AF2의 평가 범위를 구분한다. 단계별 ligand-relative 기하 보존을 연구 지표로 삼는다.
- D5-5: 이 문서 갱신은 보관된 로컬 v0.1.1 기준이다. 확인한 GitHub 주소는 2026-10-08 API에서 빈 저장소로 응답해 원격 최신 코드에 적용한 것으로 표시하지 않는다.


## 10. 2026-10-11 원격 코드 확인 및 문서 반영

- D6-1: 사용자의 최초 push 이후 원격 main의 1128c81 커밋을 확인했다. README 수정안의 기준 문서와 실제 원격 파일이 일치함을 검증했다.
- D6-2: 준비한 README, FK 개요 그림, 개발 로드맵을 별도 문서 브랜치에서 검토한다. 코어와 저장된 결과는 변경하지 않는다.
- D6-3: 원격 main의 GitHub Actions에서 Python 3.10/3.12 설치, pytest, score, CPU demo 통과를 확인하고 README와 BUILD_REPORT에 검증 이력을 추가한다.
