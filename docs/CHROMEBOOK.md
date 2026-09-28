# Pixel Chromebook: first working session

## 1. Crostini에서 CPU 실행 확인

2026-09-27 수정본 v0.1.1을 사용하세요. v0.1.0 ZIP에는 실행 설정 파일이
누락돼 있었습니다. 기존 작업 폴더를 덮어쓰지 않고 버전별 새 폴더에 풉니다.

ChromeOS 파일 앱에서 다운로드한 `theozyme-steer-v0.1.1.zip`을 **Linux 파일**
최상위로 복사하세요. 그러면 터미널에서 `~/theozyme-steer-v0.1.1.zip`으로 보입니다.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git unzip
python3 --version
```

Python이 3.10 이상인지 확인합니다. 3.9 이하라면 다음 설치를 진행하지 말고
Python 버전과 `cat /etc/os-release` 출력을 공유하세요.

```bash
ls -lh ~/theozyme-steer-v0.1.1.zip
mkdir -p ~/projects/theozyme-steer-v0.1.1
unzip -n ~/theozyme-steer-v0.1.1.zip -d ~/projects/theozyme-steer-v0.1.1
cd ~/projects/theozyme-steer-v0.1.1/theozyme-steer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,report]'
python -m unittest discover -s tests -v
```

정상이라면 마지막에 `Ran 54 tests`와 `OK`가 나옵니다. 에러가 나면 이후
명령을 진행하지 말고 실패한 명령과 출력 마지막 20줄 정도를 공유하세요.

```bash
python -m steer.cli.main demo --config configs/runs/toy.json --out runs/chromebook-check --plot
```

그림: `runs/chromebook-check/comparison.png`, 집계: `runs/chromebook-check/summary.json`.
ChromeOS 파일 앱에서 압축을 푼 프로젝트 폴더를 찾아 그림을 열 수 있습니다.
이 결과는 합성 좌표 데모이며 실제 효소 설계 결과가 아닙니다.

같은 데모를 이어서 실행할 때만 `--resume`을 붙입니다. 새 실험은 새로운
`--out` 폴더를 사용합니다.

터미널을 다시 열면 다음 두 줄로 작업 위치와 환경을 복구하세요.

```bash
cd ~/projects/theozyme-steer-v0.1.1/theozyme-steer
source .venv/bin/activate
```

## 2. Codex CLI

Use the current official Linux installer:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

After installation, open a new terminal if the installer requests a PATH refresh,
enter this project directory, activate `.venv`, and run `codex`. Sign in using
the available account method. Account/workspace access is checked during login.

The repository includes AGENTS.md so a coding session can see the tests,
dependency boundary and genuine GPU blockers. The next concrete task is to use
the lab/Colab runtime probe and pinned upstream code to build the real adapter.

This browser conversation does not automatically gain access to Crostini files
when the CLI is installed. Keep project changes in Git and transfer the actual
files/commits between environments.

[Official Codex CLI instructions](https://learn.chatgpt.com/docs/codex/cli).

## 3. GitHub

No repository or account is connected to this bundle. After checking the files,
create a repository under your own account and add its URL as the remote. The
following initializes local Git without embedding any authentication token:

```bash
git init -b main
git add .
git status --short
git commit -m "Add CPU geometry and FK steering core"
```

If Git asks for author name/email, configure your own identity. Then, after
creating the intended GitHub repository:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Replace the placeholder with the actual URL. Complete login through your local
GitHub authentication flow. Do not paste tokens into this README or the chat.
The archive excludes models, SIFs and run folders. Choose the public license
before treating the project as an open-source release. Until real GPU results
exist, keep the README's CPU-only status statement intact.
