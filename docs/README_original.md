# Theozyme-Steered Enzyme Design

> **한 줄 요약**: RFdiffusion2 + LigandMPNN 효소 설계 파이프라인에 inference-time steering
> (Feynman–Kac / gradient guidance) 을 얹어, 재학습 없이 샘플링을 촉매 기하(theozyme)
> reward 쪽으로 유도한다.

<!-- TODO: 대표 그림 1장 — baseline vs steered 촉매 기하 분포.
     eval/report/make_report.py 의 plot_geometry_distribution 산출물을 여기 삽입.
     평가자가 30초 안에 기여를 파악하는 자리이므로 그림 없이 공개하지 말 것. -->

<!-- TODO: 배지 (라이선스, python 버전) — 공개 직전에 추가 -->

> **상태**: 개발 중 (스캐폴딩 단계). 설계 결정 근거는 [`DECISIONS.md`](DECISIONS.md) 참조.

---

## 1. Motivation

### 1.1 효소 de novo 설계에서 촉매 기하가 병목인 이유
<!-- TODO: 서열·폴드는 맞지만 활성부위 기하가 Å 단위로 어긋나 활성이 안 나오는 문제.
     기존 보고들의 성공률 수치를 인용하며 병목을 정량적으로 제시. -->

### 1.2 재학습 없이 샘플링만 바꾸는 접근(inference-time steering)의 매력
<!-- TODO: 파인튜닝 대비 비용·데이터 요구량 비교. reward 만 바꿔 목표를 교체할 수 있는 유연성. -->

### 1.3 기존 binder steering 연구와 이 작업의 차이
<!-- TODO: 이 레포의 기여 문장. binder(계면 면적·접촉 같은 비교적 매끄러운 reward)와 달리
     효소 활성부위 reward 는 소수 원자의 Å/도 단위 구속이라 훨씬 희소하다는 점을 명시. -->

---

## 2. Method

### 2.1 배경: RFdiffusion2 / LigandMPNN 파이프라인
<!-- TODO: 백본 생성 → 서열 설계 → AF2 재예측 흐름 다이어그램 -->

### 2.2 Theozyme reward 정의
<!-- TODO: 거리/각/이면각 구속 → 정규화 위배량 → 매끄러운 감쇠 → 가중합.
     수식과 configs/theozyme/ser_his_asp.yaml 예시를 나란히. -->

### 2.3 FK steering (입자·가중·resampling)
<!-- TODO: 입자 N, log potential(difference/max/sum), ESS 기반 resampling 조건.
     potential 형태 선택은 실험 결과(6.3)와 연결. -->

### 2.4 Gradient guidance (선택적)
<!-- TODO: 미분가능 reward 로 디노이징 방향 보정. 스코프상 생략될 수 있음을 명시. -->

### 2.5 스케줄: 어느 timestep 에 개입하는가
<!-- TODO: 너무 이른 개입은 구조 미형성, 너무 늦으면 되돌릴 자유도 없음.
     start_frac/end_frac ablation 결과와 함께. -->

---

## 3. Installation

<!-- TODO: 아래 자리는 세션 A 이후, 실제로 설치가 되는 것을 확인한 뒤 채운다.
     "돌아가지 않는 설치 안내"가 레포 신뢰도를 가장 빨리 깎는다. -->

### 3.1 요구사항 (GPU, apptainer)
<!-- TODO: GPU 메모리 요구량(입자 수 N 에 비례), apptainer 버전, 디스크 요구량 -->

### 3.2 conda 환경 2종
<!-- TODO: env/steer-cpu.yml (로컬 개발) / env/eval.yml (평가) 생성 명령과 용도 구분.
     GPU 노드에는 conda env 를 만들지 않는다는 점을 눈에 띄게 적을 것. -->

```bash
# TODO: 검증 후 주석 해제
# conda env create -f env/steer-cpu.yml && conda activate steer-cpu && pip install -e .
```

### 3.3 컨테이너 이미지 준비 (출처·해시 표)
<!-- TODO: env/containers.md 의 표를 여기로 요약 전재 + STEER_SIF_DIR 설정 안내 -->

### 3.4 설치 검증 (`pytest`)
<!-- TODO: GPU·컨테이너 없이 전체 테스트가 통과함을 보여주는 것이 이 절의 목적 -->

```bash
# TODO: 검증 후 주석 해제
# pytest -q
```

---

## 4. Quickstart

### 4.1 최소 예제 1커맨드
<!-- TODO: scripts/run_steered.sh configs/runs/<...>.yaml 한 줄 + 예상 소요 시간/GPU -->

### 4.2 출력물 읽는 법
<!-- TODO: runs/<run_name>/ 의 00_config ~ 04_eval 레이아웃과 각 파일의 의미 -->

---

## 5. Usage

### 5.1 theozyme config 작성법
<!-- TODO: 새 반응을 추가하는 절차 — 참조 구조에서 기하 실측 → yaml 작성 → score_only 로 검증 -->

### 5.2 steering 하이퍼파라미터
<!-- TODO: num_particles / lambda_temp / potential / ESS 임계값 / 스케줄 각각의 실전 감각 -->

### 5.3 baseline 재현
<!-- TODO: seed·num_designs 를 steered run 과 맞추는 것이 공정 비교의 전제임을 명시 -->

### 5.4 랩 GPU / SLURM 에서 돌리기
<!-- TODO: sync_to_gpu.sh → sbatch 흐름, bind-mount 경로 주의사항 -->

---

## 6. Results

<!-- TODO: 실험 완료 후 채운다. 숫자 없이 주장만 쓰지 말 것. -->

### 6.1 평가 지표 정의
<!-- TODO: theozyme RMSD(주 지표), scRMSD/pLDDT(설계 가능성), 다양성(모드 붕괴).
     성공 cutoff 값을 사전에 못 박고 그 이유를 쓸 것. -->

### 6.2 baseline vs steered
<!-- TODO: 성공률 표 + 분포 그림 + 필터 통과 funnel -->

### 6.3 ablation (입자 수, λ, 스케줄)
<!-- TODO: 파라미터별 성공률 곡선. potential 형태(difference vs max) 비교도 여기 -->

### 6.4 실패 사례와 한계
<!-- TODO: 기하는 맞았으나 구조가 무너진 사례, 다양성 붕괴 사례를 그림과 함께.
     이 절이 있어야 6.2 의 수치가 신뢰를 얻는다. -->

---

## 7. Repository Layout

<!-- TODO: DECISIONS.md D1-1 의 트리를 요약해 전재. 각 층의 "왜 여기 있는가" 한 줄씩. -->

```
src/steer/     순수 python steering 라이브러리 (의존성: numpy만) — 이 프로젝트의 기여
pipelines/     apptainer 컨테이너 호출 래퍼 (로직 없음)
eval/          설계 결과 평가 (컨테이너 밖, 무거운 의존성 허용)
configs/       코드가 아닌 모든 파라미터. configs/runs/ 1파일 = 실험 1회
scripts/       사람이 직접 치는 진입점
tests/         GPU·컨테이너 없이 전부 통과해야 함
```

---

## 8. Reproducibility

<!-- TODO: config → 결과 대응표, 시드 고정 방식, 컨테이너 이미지 해시.
     "이 그림은 이 config" 로 1:1 인용 가능해야 한다. -->

| run config | 이미지 해시 | 커밋 | 결과 |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

---

## 9. Limitations & Future Work

<!-- TODO: 실험 후 채운다. 미리 예상되는 것들:
     - FK resampling 의 다양성 감소
     - reward 가 근사하는 것은 기하일 뿐 활성이 아니라는 본질적 격차
     - LigandMPNN 단계에는 steering 을 걸지 않았음
     - theozyme 정의가 참조 구조에 의존 -->

---

## 10. Citation & Acknowledgements

<!-- TODO: upstream 라이선스 명시 (RFdiffusion2 / RFdiffusionAA / LigandMPNN / AF2 각각).
     라이선스 조건을 확인하기 전에 레포를 공개하지 말 것.
     FK steering 선행 연구 인용. -->
