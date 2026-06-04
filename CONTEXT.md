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

**레퍼런스(템플릿·디자인 출처): https://crepejung00.github.io/ (Jaewoo Jung, 같은 CVLAB 동료).** 이 사이트의 레이아웃·CSS·구조는 해당 사이트를 미러링해 내용만 교체한 것.

아키텍처·명령어·주의사항 상세는 `CLAUDE.md` 참고.

---

## Changelog

### 2026-06-04 — 커밋 & 푸시 (배포) `c818fec2`
- 아래 세션 변경 전체(Kakao 경력, 티저/링크, LinkedIn, Kakao,Seoul, CV 간격, 티저 세로중앙, Research Interests 주석, 티저 고화질, LaTeX 임시파일 추적제외)를 `master`에 커밋 후 `origin/master` 푸시 → GitHub Pages 배포.
- `assets/kakao.png`(사용자 추가, 사이트 미참조)는 사용자 요청으로 **삭제**(빈 `assets/` 디렉토리도 제거). 홈 Kakao 로고는 공식 `images/kakao-logo.svg` 사용.

### 2026-06-04 — 티저 이미지 고화질 교체 (700px → ~3000px)
- 이전 sips 700px 리사이즈가 확대 시 저화질이라, 원본 재수집 후 **긴 변 3000px**로 다운스케일(고화질, 라이트박스 확대에도 선명). 원본은 8342×1833(APPLE)·9900×3120(NoiseRefine)로 과대 → 11MB/4MB라 그대로는 부적합.
- `images/APPLE`: PNG(3MB) → **고화질 JPEG로 변환**(`APPLE.jpg`, 544KB, q92) 후 `index.html` src를 `APPLE.png`→`APPLE.jpg`로 수정, `APPLE.png` 삭제.
- 최종: `APPLE.jpg` 544KB(3000×659), `NoiseRefine.jpg` 564KB(3000×945), `HeadHunter.jpg` 448KB(1194×1398, 원본 유지). 합 ~1.5MB.

### 2026-06-04 — cv.html Research Interests 섹션 주석 처리
- HTML CV(`cv.html`)의 Research Interests 섹션을 `<!-- ... -->`로 주석 처리(홈페이지에 이미 동일 내용 존재). 렌더 섹션: Education → Experience → Publications → Extracurricular. 되살리려면 주석 래퍼 제거.

### 2026-06-04 — CV_LATEX LaTeX 임시파일 git 추적 제외 (staged, 미커밋)
- `CV_LATEX/.gitignore` 신규: `*.aux *.fdb_latexmk *.fls *.log *.out *.synctex.gz *.toc *.bbl *.bcf *.run.xml` 등 LaTeX 중간 산출물 무시. `CV.tex`(소스)·`CV.pdf`(결과물)는 계속 추적.
- 기존 추적 중이던 6개 임시파일(`CV.aux, CV.fdb_latexmk, CV.fls, CV.log, CV.out, CV.synctex.gz`)을 `git rm --cached`로 추적 해제(디스크엔 유지). → 다음 커밋 시 반영.

### 2026-06-04 — 회사명 Kakao,Seoul / CV 제목-링크 간격 / LinkedIn / 티저 세로중앙 (※ 아직 커밋/푸시 안 함)
- 회사명 `Kakao` → `Kakao, Seoul` (index.html Experience, cv.html, CV.tex).
- **CV PDF 제목-링크 겹침 수정**: `CV.tex` `\pubitem` tabularx 컬럼 스펙에 `@{\hspace{1.5em}}` 추가 → 긴 제목(APPLE)과 `Paper|Project Page|Code` 링크가 붙던 문제 해소.
- **LinkedIn 추가** (https://www.linkedin.com/in/jiwon-kang-b02b0911b/): `CV.tex` 헤더(주석 해제), `index.html` hero 버튼 + footer, `cv.html` 헤더 링크 + footer.
- **홈페이지 티저 세로 중앙 정렬**: `css/index.css`에 `.publication-block.columns { align-items: center; }` (7개 블록 일괄, Bulma 기본 stretch → center).
- `JiwonCV.pdf` 재빌드.

### 2026-06-04 — Kakao 인턴 경력 + 3개 논문 티저/링크 반영 (※ 아직 커밋/푸시 안 함)
- **레퍼런스 출처 명시**(요청1): 저장소 개요에 https://crepejung00.github.io/ 추가.
- **Kakao 연구 인턴(Mar–May 2026) 반영**(요청2):
  - `index.html`: 주석 처리됐던 Experience 섹션을 노출 + Kakao 항목(공식 CI 로고 `images/kakao-logo.svg`), News에 "started a research internship at Kakao" 추가, Whisper 논문에 'Work done at Kakao' 표기(로고 포함).
  - `cv.html`: Education 다음에 Experience(Kakao + 로고) 섹션 추가, Whisper 베뉴 `Under Review · Kakao`.
  - `CV_LATEX/CV.tex`: Education↔Publications 사이 `\section{Experience}`(Kakao, Research Intern, Whisper bullet) 추가, `\lastupdate` June 4 2026 → `latexmk`로 빌드 → `JiwonCV.pdf` 갱신.
  - `css/index.css`: `.title-logo.logo-kakao { height:15px }` 추가. 로고는 Wikimedia "Kakao CI yellow.svg"(공식, #ffcd00) 사용(첨부 이미지 파일 접근 불가하여 동일 CI 다운로드).
- **3개 논문 티저 이미지 + 링크**(요청, 별도): 프로젝트 페이지에서 매핑·수집.
  - APPLE → `cvlab-kaist.github.io/APPLE/` (arXiv 2601.15288, Code cvlab-kaist/APPLE), teaser `images/APPLE.png`.
  - "A Noise is Worth Diffusion Guidance" → `cvlab-kaist.github.io/NoiseRefine/` (arXiv 2412.03895, Code cvlab-kaist/NoiseRefine), teaser `images/NoiseRefine.jpg`.
  - "Where and How to Perturb" → `cvlab-kaist.github.io/HeadHunter/` (arXiv 2506.10978, Code cvlab-kaist/HeadHunter), teaser `images/HeadHunter.jpg`.
  - 티저는 `sips -Z 700`으로 리사이즈(원본 최대 10.7MB → 50–220KB).
  - `index.html`: placeholder→실제 티저 교체 + Project Page/PDF/arXiv/Code 버튼 추가. `cv.html`: 제목에 프로젝트 페이지 링크. `CV.tex`: `\publinks{arXiv}{Project}{Code}` 채움 → PDF 재빌드.

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
