# Research with AI

## 논문 정보 관리

Codex 등의 AI 에이전트를 이용하여 논문 목록을 수집 논문 요약, 발표 자료 작성을 할 수 있다.
관련 논문 정보는 `docs` 폴더에 저장된다.

AI 에이전트를 이용하여 논문 검색 및 논문 분석 등을 수행하는 방법은 [docs/README.md](./docs/README.md)를 참고하라.

## Paper Viewer

이 웹앱은 `docs` 폴더에 정리된 논문 메타데이터, 요약 문서, 슬라이드를 탐색하고 읽기 위한 도구이다.
토픽 목록, 논문 목록, 요약 Markdown, Marp 슬라이드, 원문 PDF를 한 화면에서 확인할 수 있다.

논문 문서 저장소의 구조와 관리 규칙은 [docs/README.md](./docs/README.md) 를 참고하면 된다.

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

### 데이터 소스

뷰어는 기본적으로 `docs` 폴더를 읽는다.
각 토픽 폴더에는 아래 파일이 있어야 한다.

- `metadata.json`
- `paper_list.jsonl`

논문별 요약 Markdown과 슬라이드 Markdown은 `paper_list.jsonl` 에 연결된 파일명을 기준으로 로드한다.

### 프로젝트 구조

```text
.
├─ public/        # 클라이언트 UI
├─ docs/          # 논문 문서 저장소
├─ server.js      # Express 서버 및 렌더링 API
└─ paper_viewer.bat
```

### API 개요

- `GET /api/topics`: 토픽 목록
- `GET /api/papers?topic=...`: 토픽별 논문 목록
- `GET /api/render-summary?topic=...&file=...`: 요약 HTML 렌더링
- `GET /api/render-slide?topic=...&file=...`: 슬라이드 HTML 렌더링
- `GET /raw/:topic/:file`: 원본 파일 제공

### 비고

- 기본 바인딩 주소는 `127.0.0.1` 이다.
- 수식 렌더링은 MathJax를 사용한다.
- 슬라이드 렌더링은 `@marp-team/marp-core` 를 사용한다.
