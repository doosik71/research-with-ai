# Research with AI

## 논문 정보 관리

Codex 등의 AI 에이전트를 이용하여 논문 목록을 수집 논문 요약, 발표 자료 작성을 할 수 있다.
관련 논문 정보는 `static/docs` 폴더에 저장된다.

AI 에이전트를 이용하여 논문 검색 및 논문 분석 등을 수행하는 방법은 [docs/README.md](./static/docs/README.md)를 참고하라.

## Paper Viewer

이 웹앱은 `static/docs` 폴더에 정리된 논문 메타데이터, 요약 문서, 슬라이드를 탐색하고 읽기 위한 도구이다.
토픽 목록, 논문 목록, 요약 Markdown, Marp 슬라이드, 원문 PDF를 한 화면에서 확인할 수 있다.

논문 문서 저장소의 구조와 관리 규칙은 [docs/README.md](./static/docs/README.md) 를 참고하면 된다.

### 주요 기능

- 연구 분야별 토픽 목록 조회
- 토픽별 논문 목록 조회
- 논문 요약 Markdown 렌더링
- Marp 슬라이드 렌더링
- arXiv / PDF 바로 열기
- 3단 컬럼 레이아웃과 패널 토글

### 실행 방법

#### 요구 사항

- Node.js
- `npm install` 로 설치된 의존성

#### 실행

```bash
npm start -- --dir docs --port 8080
```

브라우저에서 `http://127.0.0.1:8080` 을 연다.

Windows에서는 아래 배치 파일로도 실행할 수 있다.

```bat
paper_viewer.bat
```

### 개발용 문서 서버

`static/docs` 아래의 마크다운 파일을 직접 만들고 수정하는 로컬 전용 서버도 사용할 수 있다.
이 서버는 Cloudflare Pages 배포 경로에 포함되지 않는다.

```bash
npm run docs:dev
```

기본 실행 주소는 `http://127.0.0.1:8787` 이다.

### 데이터 소스

뷰어는 기본적으로 `static/docs` 폴더를 읽는다.
각 토픽 폴더에는 아래 파일이 있어야 한다.

- `metadata.json`
- `paper_list.jsonl`

논문별 요약 Markdown과 슬라이드 Markdown은 `paper_list.jsonl` 에 연결된 파일명을 기준으로 로드한다.

### 프로젝트 구조

```text
.
├─ src/           # 클라이언트 UI
├─ static/docs/   # 논문 문서 저장소
├─ scripts/docs-dev-server.js  # 로컬 전용 문서 관리 서버
└─ paper_viewer.bat
```

### 비고

- 기본 바인딩 주소는 `127.0.0.1` 이다.
- 수식 렌더링은 MathJax를 사용한다.

## Script Usage

Added utility scripts can be run with `npm run`:

```bash
# Generate static/docs/manifest.json
npm run manifest

# Generate all missing paper analysis reports
npm run update-docs

# Generate index.md for one topic from metadata.json and paper_list.jsonl
npm run update-index -- activation_function

# Initialize missing summary file names interactively for a topic
npm run init-summary -- semantic_segmentation

# Generate one paper analysis report
npm run update-summary -- speech_recognition/adapting_whisper_for_streaming_speech_recognition_via_twopass_decoding

# Generate topic-level report artifacts for all 12 stages for one topic
npm run generate-report -- activation_function

# Generate topic-level report artifacts for multiple topics in the given order
npm run generate-report -- activation_function anomaly_detection

# Generate report artifacts for every topic under static/docs that has metadata.json
npm run generate-report

# Generate only one stage for a topic
npm run generate-report -- --4 activation_function

# Add an arXiv paper entry to a topic
npm run add-doc -- instance_segmentation http://arxiv.org/abs/2210.12852v3

# Convert one arXiv abs URL to a JSON line for paper_list.jsonl
npm run arxiv2json -- https://arxiv.org/abs/2004.06632v1
```

`init-summary` reads titles from standard input, updates `static/docs/<topic_id>/paper_list.jsonl`, and exits when you enter `/quit` or `/exit`.

`update-index` reads `static/docs/<topic_id>/metadata.json` and `paper_list.jsonl`, then writes `static/docs/<topic_id>/index.md` sorted by newest year first and title within the same year.

`generate-report` creates topic report artifacts under `static/docs/<topic_id>/report-01` through `report-12`. With no arguments it scans every subdirectory under `static/docs` and runs all stages for each topic that contains `metadata.json`, continuing past failures and summarizing failed topics at the end. With one or more `<topic_id>` arguments, it runs all stages for those topics in the given argument order and skips any topic missing `metadata.json`. With `--1` to `--12` plus a single `<topic_id>`, it runs only that stage for that topic. For example, `npm run generate-report -- --4 activation_function` runs only stage 4 for the `activation_function` topic.

`arxiv2json` fetches an arXiv abs page and prints a single-line JSON object with `title`, `author`, `year`, `url`, `summary`, and `slide` fields.

For `init-summary`, `update-index`, `update-summary`, `generate-report`, `arxiv2json`, and `add-doc`, pass script arguments after `--`.
