# CONTEXT.md

> **이 파일은 이 저장소의 작업·수정 내역 로그입니다.**
>
> **규칙 (Claude 및 모든 작업자가 지킬 것):**
> 1. 작업을 시작하기 전에 이 `CONTEXT.md`를 먼저 읽는다.
> 2. 모든 작업 및 수정 내역(무엇을, 어디서, 왜)을 아래 changelog에 기록한다.
> 3. **다른 폴더를 탐색/작업해야 하는 경우, 그 폴더에 별도의 `CONTEXT.md`가 있는지 먼저 확인하고, 있으면 읽은 뒤에 작업을 진행한다.** (예: `blog-backup/CONTEXT.md`, `CV_LATEX/CONTEXT.md` — 해당 폴더 범위의 작업은 그 폴더의 CONTEXT.md를 우선한다.)
> 4. 최신 항목을 맨 위에 추가한다 (역순).

## 저장소 개요

`loggerjk.github.io` — Jiwon Kang의 학술 개인 홈페이지. 빌드 단계가 없는 **순수 정적 사이트**(Bulma 기반)이며 GitHub Pages가 그대로 서빙한다. 구성:
- **루트** = 라이브 정적 사이트 (`index.html`, `cv.html`, `css/`, `js/`, `images/`)
- **`blog-backup/`** = 이전 Jekyll(Minimal Mistakes) 기술 블로그 아카이브 (비노출)
- **`CV_LATEX/`** = CV의 LaTeX 소스 (`CV.tex` → `CV.pdf`)

아키텍처·명령어·주의사항 상세는 `CLAUDE.md` 참고.

---

## Changelog

### 2026-06-04 — 전체 변경사항 커밋 & 푸시 (GitHub Pages 배포)
- 아래 모든 변경(블로그 백업, 정적 홈페이지 전환, CLAUDE/CONTEXT 도입)을 `master`에 커밋 후 `origin/master`로 푸시 → GitHub Pages 자동 배포(`https://loggerjk.github.io`).

### 2026-06-04 — CLAUDE.md / CONTEXT.md 도입
- `CLAUDE.md` 신규 작성: 저장소 구조, 아키텍처(정적 사이트·`.nojekyll`·toggle-by-comment·CV 3중 동기화), 명령어(로컬 프리뷰, CV 빌드, 배포), 주의사항 정리.
- `CONTEXT.md`(이 파일) 신규 작성: 작업 로그 + "폴더 탐색 전 CONTEXT.md 확인" 규칙 명시.
- `CV_LATEX/` 폴더 확인됨 (사용자가 추가): Jiwon Kang Academic CV의 LaTeX 소스(`CV.tex`, 246줄, latexmk 사용). 루트 `JiwonCV.pdf`와 동일 내용의 CV를 생성.

### 2026-06-04 — Jekyll 블로그 → 정적 학술 홈페이지 전환
**배경:** 기존 `loggerJK.github.io`(Jekyll Minimal Mistakes 한국어 기술 블로그)를 Jaewoo Jung의 홈페이지(crepejung00.github.io, Bulma 정적 사이트) 스타일의 학술 홈페이지로 교체. 기존 블로그 글은 모두 아카이빙.

- **블로그 백업:** 루트의 모든 기존 Jekyll 파일을 `git mv`로 `blog-backup/`에 이동(히스토리 보존). 글 18개 + `backup_posts/` 4개 + `_config.yml` 등 보존. 검색엔진 인증 파일(`google…html`, `naver…html`)은 루트 유지.
- **`.nojekyll`** 루트 추가 → GitHub Pages가 Jekyll 빌드 없이 정적 서빙(`blog-backup/`도 빌드 안 됨).
- **정적 사이트 미러링:** Jaewoo 사이트(비공개 repo라 라이브에서 미러링)의 `css/`·`js/`·구조를 가져와 내용만 Jiwon으로 교체.
  - `index.html`: 프로필(기존 `assets/images/bio-photo.jpg` → `images/portrait.jpg` 재사용) + bio + 링크(CV PDF / Scholar `user=A2PurdIAAAAJ` / GitHub `loggerJK`) + News 5건 + Publications 7편(2026 / 2025 / Preprints).
  - `cv.html`: Research Interests · Education · Publications · Extracurricular(AIKU). LaTeX CV 내용 기반.
  - `images/SE-NeRF.png`, `images/RAIN-GS.png`은 실제 teaser 재사용(Jiwon 공저자), 나머지는 `images/placeholder.svg`.
  - `JiwonCV.pdf`(루트) = Jiwon CV PDF 사본, 사이트 "CV" 버튼이 링크.
  - `css/index.css`에 `.cv-subsection-title` 규칙 추가(CV 페이지 소분류용).
- **주석 처리(추후 손쉽게 활성화):** Experience 섹션, X/LinkedIn 버튼, Google Analytics, favicon 링크.
- **검증:** 로컬 `python3 -m http.server`로 모든 URL 200 확인, 두 페이지 정상 렌더, Jaewoo 식별정보 잔존 0건(공저자 표기만 유지).
- **남은 TODO:** 논문 teaser 썸네일 5개(현재 placeholder) — APPLE / A Noise / Where&How / Whisper / Transferability; 위 논문들의 arXiv·Project Page 링크; 선택: favicon, GA id, X/LinkedIn.
