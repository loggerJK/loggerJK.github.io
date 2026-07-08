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

### 2026-07-08 — 전체 문서 압축(3페이지 → 2페이지 복귀): `\largesection` 간격 축소
- 배경: SAMSUNG Experience 항목·KATUSA MOS 불릿 등 콘텐츠가 늘면서 CV.pdf가 2페이지에서 3페이지로 늘어남. 사용자가 "한 줄씩만 위로 올리고 싶다"고 요청 — 확인 결과 실제로 페이지 3에는 AIKU 항목의 불릿 한 줄("Co-founder & Vice President...")만 넘어가 있었고, 페이지 2는 바닥까지 꽉 찬 상태였음(ghostscript로 각 페이지 PNG 렌더해서 확인).
- Plan mode에서 "전체 문서 압축(2페이지 복귀 목표)" vs "특정 섹션만" 중 사용자가 전자를 선택.
- **`CV_LATEX/CV.tex`** (L143): `\largesection` 매크로(모든 `\section` 앞뒤에 붙는 공용 간격, `Research Interests/Education/Experience/Publications/Extracurricular Activities` 5개 섹션에 전부 적용)의 `\vspace{5pt} ... \vspace{5pt}` → `\vspace{3pt} ... \vspace{3pt}`로 축소. 섹션당 앞뒤 각 2pt씩, Extracurricular 이전에 이미 4개 섹션(8회 적용)을 지나므로 필요한 한 줄(~12pt) 분량을 넉넉히 확보 — 개별 항목이 아니라 이미 존재하는 단일 공용 매크로를 살짝 조정하는 방식이라 전체 문서에 고르게, 눈에 띄지 않게 분산 적용됨.
- **검증**: `latexmk -pdf CV.tex` 재빌드 → 3페이지 → **2페이지**로 복귀 확인. ghostscript로 1·2페이지 렌더 → AIKU 불릿이 2페이지 안에 들어옴, 전체 여백감도 부자연스럽게 좁아지지 않음 확인.

### 2026-07-08 — `\cventry` 매크로: 2번째 줄 생략 시 줄간격 이중 적용 버그 수정
- 배경: 사용자가 KATUSA/AIKU(Extracurricular) 항목과 Kakao/SAMSUNG(Experience) 항목이 같은 `\cventry` 매크로를 쓰는데도 불릿 리스트와의 줄간격이 확연히 다르다고 리포트(스크린샷 비교) — KATUSA/AIKU는 타이틀 줄과 불릿이 거의 겹칠 정도로 붙어 있었음.
- 원인: `\cventry`(`CV_LATEX/CV.tex:87-93`, 사용자가 직전에 직접 추가한 `\vspace{-7pt}`/`\vspace{-5pt}` 튜닝)에서 두 `\vspace`가 2번째 줄(`#3`/`#4`) 존재 여부와 무관하게 항상 실행됨. Kakao/SAMSUNG처럼 2번째 줄이 있으면 `-7pt`는 줄1↔줄2 사이, `-5pt`는 줄2↔불릿 사이로 각각 분산되어 정상으로 보이지만, KATUSA/AIKU처럼 2번째 줄을 `{}{}`로 생략하면(`L280`, `L286-287`) `\ifx...\else...\fi` 분기가 스킵되면서 두 `\vspace`가 연달아 그대로 실행되어 줄1↔불릿 사이에 합계 -12pt가 몰빵됨 — 이게 겹쳐 보이는 원인.
- **`CV_LATEX/CV.tex`** (L87-93): `\vspace{-7pt}`를 `\ifx\cv@second\@empty\else ... \fi` 분기 안으로 옮겨 2번째 줄이 실제로 렌더링될 때만 실행되도록 수정. 마지막 `\vspace{-5pt}`는 그대로 무조건 실행(2번째 줄이 있든 없든 그다음 콘텐츠와의 간격을 담당). 결과: 2번째 줄이 있는 항목(Kakao/SAMSUNG/Education)은 기존과 동일, 2번째 줄이 없는 항목(KATUSA/AIKU)은 `-5pt` 하나만 적용되어 정상적인 간격으로 보임.
- **검증**: `latexmk -pdf CV.tex` 재빌드(3페이지, 기존 30pt Overfull 경고만 잔존) → ghostscript(`gs`)로 Extracurricular 페이지 PNG 렌더 → KATUSA/AIKU 불릿이 더 이상 타이틀과 겹치지 않고 Kakao 스타일과 일관된 간격으로 나오는 것 확인.

### 2026-07-08 — CV.tex Experience(Kakao) 항목: 지역명을 우측 하단으로 이동
- 배경: 사용자가 다른 CV의 "PROJECT" 섹션 스크린샷을 참고로, 지역명(예: "Seoul, Korea")을 항목 2번째 줄 우측에 이탤릭체로 배치하는 형식을 요청. `CV_LATEX/CV.tex`에서 지역명이 타이틀에 섞여 있던 곳은 Experience 섹션의 Kakao 항목("Kakao, Seoul")이 유일(Education 항목들은 기관명만 있고 별도 지역명 없음).
- **`CV_LATEX/CV.tex`** (L180): `\cventry{Kakao, Seoul}{Mar. 2026 -- May 2026}{Research Intern}{}` → `\cventry{Kakao}{Mar. 2026 -- May 2026}{Research Intern}{\textit{Seoul}}`. 기존 `\cventry{Title}{Date}{2nd-line-left}{2nd-line-right}` 매크로의 4번째 인자(이미 Education/KAIST 항목의 "Advisor: ..."에 쓰이던 우측 정렬 슬롯)를 재사용 — 신규 매크로 없이 "Seoul"을 이동 후 `\textit{}`로 이탤릭 처리.
- **PDF 동기화**: `latexmk -pdf CV.tex` 재빌드(2페이지, 기존 30pt Overfull 경고만 잔존·무해).
- **스코프 제외**: `cv.html`·`index.html`의 Experience 항목은 여전히 "Kakao, Seoul"로 결합되어 있음 — 사용자가 이번 요청을 CV.tex로 한정했으므로 HTML은 미변경.

### 2026-07-08 — CV.tex 헤더에 Personal Website 링크 추가
- 배경: 사용자가 `CV_LATEX/CV.tex` 헤더에 개인 홈페이지 링크(`https://loggerjk.github.io/`)를 아이콘과 함께 Google Scholar 왼쪽에 추가해달라고 요청.
- **`CV_LATEX/CV.tex`** (`\cvheader` 매크로, L69): 이미 존재하던 주석 처리된 placeholder `% \href{https://TODO.github.io}{\faGlobe\ Personal Website}\cvgap % TODO: personal website`를 주석 해제하고 실제 URL로 교체 → `\href{https://loggerjk.github.io/}{\faHome\ Personal Website}\cvgap`. 위치상 Google Scholar 바로 앞이라 요청대로 왼쪽에 배치됨. 아이콘은 처음엔 기존 placeholder의 `\faGlobe`를 그대로 썼으나, 사용자가 곧바로 집 아이콘으로 바꿔달라고 요청하여 `\faHome`으로 교체(둘 다 fontawesome5 fallback 블록(L35-38)에 이미 정의되어 있어 신규 의존성 없음).
- **PDF 동기화**: `latexmk -pdf CV.tex` 재빌드 2회(아이콘 변경 반영, 최종 2페이지, 기존 30pt Overfull 경고만 잔존·무해).

### 2026-07-08 — cv.html Education 항목의 Leave of Absence/Research Intern 줄 불릿 처리
- 배경: 직전 수정(바로 아래 항목)에서 `cv.html` Education의 Korea University desc에 `<br>`로만 줄바꿈해 추가했던 "Leave of Absence..."·"Undergraduate Research Intern..." 두 줄을 사용자가 불릿 포인트로 바꿔달라고 요청.
- 조사: `.cv-entry-desc`에는 리스트 스타일 override가 없고, Bulma 기본 리셋(`ul{margin:0;padding:0} ul{list-style:none}`)이 전역 적용되어 있어 raw `<ul><li>`는 불릿이 안 보임. 동일한 문제를 이미 `.experience-desc-list`(`css/index.css` L94)·`.publication-desc-list`(L160)에서 override 클래스로 해결한 전례가 있어 같은 패턴 재사용.
- **`css/index.css`**: `.cv-entry-desc` 규칙 블록 바로 뒤에 `.cv-entry-desc-list` 신규 추가 — `list-style: disc; margin: 0.15em 0 0 1.1em; padding: 0;` (색상·글자크기는 `.cv-entry-desc`에서 상속, `.experience-desc-list`와 동일한 최소 override 방식).
- **`cv.html`**: Korea University 항목의 desc를 "B.S. ... GPA: 4.24/4.5" 평문 첫 줄 + `<ul class="cv-entry-desc-list">` 안에 Leave of Absence·Undergraduate Research Intern 두 `<li>`로 재구성.
- 참고: 숨겨진(주석 처리된) Research Interests 블록도 동일한 raw `<ul>` 문제를 갖고 있으나 현재 미노출 상태라 이번 스코프에서는 손대지 않음.

### 2026-07-08 — CV.tex 변경사항 cv.html/index.html 동기화 + JiwonCV.pdf 중복 제거
- 배경: 사용자 요청 — (1) 직전 `CV_LATEX/CV.tex` 변경사항(KATUSA 불릿화, 사용자가 직접 수정한 Research Interests 문단/불릿 순서, Education "Leave of Absence" 불릿)을 `cv.html`·`index.html`에도 반영. (2) 루트의 `JiwonCV.pdf`는 중복 파일이니 삭제하고 `CV_LATEX/CV.pdf`로 연결되게 정리. 조사 결과 `index.html`의 "CV" 버튼은 이미 `./CV_LATEX/CV.pdf`를 직접 링크하고 있었고(과거 커밋 `da693669`에서 `JiwonCV.pdf`가 git 추적에서 제외됨), `JiwonCV.pdf`를 참조하는 HTML/JS가 전혀 없어 완전히 죽은 파일이었음. index.html 소개 문구를 CV.tex 새 문단으로 교체할지는 Plan mode에서 사용자에게 확인 후 "교체" 선택.
- **`cv.html`**: (a) Education의 Korea University 항목 desc에 `Leave of Absence for R.O.K. Military Service (Dec. 2020 – Jun. 2022)` 줄 추가(GPA/Double Major와 Research Intern 사이, CV.tex 불릿 순서와 동일). (b) Extracurricular Activities에 KATUSA 항목 신규 추가(기존엔 AIKU만 있었음) — 형제 항목(AIKU, Kakao Experience)과 동일한 plain-text `.cv-entry-desc` 패턴 사용(`.cv-entry-desc`에 리스트 스타일이 없어 `<ul>` 대신 단일 문장으로). (c) 주석 처리된(숨김) Research Interests 블록의 문단/불릿 순서를 CV.tex 최신 내용으로 갱신(계속 숨김 유지, 재활성화 안 함).
- **`index.html`**: 소개 문구(intro bio, 두 번째 문장)를 CV.tex의 새 Research Interests 문단("Starting my research career in 3D vision...")으로 교체.
- **`JiwonCV.pdf` 삭제**: `rm JiwonCV.pdf`(비추적 파일이라 git 상 영향 없음).
- **`CLAUDE.md`**: "CV exists in three places" 섹션을 "두 곳"으로 수정(`JiwonCV.pdf` 단계 제거, `CV_LATEX/CV.pdf`가 사이트 CV 버튼이 직접 링크하는 대상임을 명시). Commands 섹션에서 `cp CV_LATEX/CV.pdf JiwonCV.pdf` 단계 제거.

### 2026-07-08 — CV.tex: KATUSA 소속 정보를 불릿 포인트로 변경
- 배경: 사용자가 `CV_LATEX/CV.tex`의 Extracurricular Activities 섹션에서 KATUSA 항목의 부대 정보("194th DSSB, 2nd Infantry Division, Eighth U.S. Army, Camp Humphreys")를 불릿 포인트로 바꿔달라고 요청. 기존에는 `\cventry`의 3번째 인자(plain 2nd-line)로 렌더링되고 있었음.
- **`CV_LATEX/CV.tex`** (L273): `\cventry{KATUSA...}{Dec. 2020 -- Jun. 2022}{194th DSSB, ...}{}` → 3/4번째 인자를 빈 값으로 바꾸고, 부대 정보를 `\begin{cvbullets}\item ...\end{cvbullets}`로 이동. Education/Experience 섹션에서 이미 쓰이던 동일한 `cvbullets` 패턴을 재사용(신규 마크업 없음). AIKU 항목은 변경하지 않음(사용자가 KATUSA만 지정).
- **PDF 동기화**: `latexmk -pdf CV.tex` 재빌드(2페이지, 기존 30pt Overfull 경고만 잔존·무해) → `cp CV_LATEX/CV.pdf JiwonCV.pdf` → md5 일치 확인.
- 참고: 이 작업 도중 사용자가 Research Interests 문단 재작성(3D vision → Master's 확장 스토리), 해당 불릿 순서 변경(Image Generation을 맨 앞으로), Education 섹션에 "Leave of Absence for R.O.K. Military Service" 불릿 추가를 에디터에서 직접 수행한 것을 확인. 이 변경들은 사용자가 외부에서 직접 수정한 것이며 그대로 유지, PDF 재빌드에 함께 반영됨.

### 2026-07-07 — Experience 항목 role/desc 구분용 불릿 추가
- 배경: 직전 수정(바로 아래 항목)에서 `<br>`로만 줄바꿈했더니 "Research Intern"과 "Worked on..." 두 줄이 시각적으로 구분되지 않는다는 피드백. 원본 `<li>` 시도의 의도(불릿으로 구분)는 맞았으나 `<ul>` 없이 썼던 게 문제였으므로, 이번엔 제대로 된 `<ul>`로 감싸 disc 불릿을 정상 렌더링.
- **`index.html`**: `.experience-desc` 안에서 `Research Intern<br>Worked on...` → `Research Intern` 텍스트 다음에 `<ul class="experience-desc-list"><li>Worked on...</li></ul>` 중첩 구조로 변경.
- **`css/index.css`**: `.experience-entry .experience-main .experience-desc a:hover` 규칙 바로 뒤에 `.experience-desc-list`(disc, margin-left 1.1em로 들여쓰기) / `.experience-desc-list li`(margin 0) 규칙 신규 추가. 색상·글자크기는 `.experience-desc`(gray-500, 0.9em)에서 상속받아 "Research Intern"과 동일 톤 유지 — 논문 카드용 `.publication-desc-list`(gray-700)와는 별도 클래스로 분리.

### 2026-07-07 — Experience 섹션 레이아웃 깨짐(불릿 점) 수정
- 배경: 사용자가 `index.html`의 Experience 카드(Kakao)에서 불릿 점(dot) 때문에 레이아웃이 깨진다고 리포트. 원인은 `<div class="experience-desc">` 안에 `<ul>`/`<ol>` 부모 없이 `<li>`가 단독으로 들어가 있었던 것 — 브라우저는 부모 리스트 유무와 무관하게 `<li>`에 UA 스타일시트로 `display: list-item`(기본 disc 마커)을 적용하는데, `.experience-desc`(`css/index.css` L91-93)에는 이를 리셋하는 list-style 규칙이 전혀 없어 마커가 기본 들여쓰기/마진과 함께 튀어나와 레이아웃이 깨짐.
- **`index.html`**: Experience 섹션의 `.experience-desc` 블록(Kakao, Research Intern) — 깨진 `<li>...</li>` 마크업을 제거하고 `cv.html` L123의 동일 항목이 이미 쓰고 있는 관례(줄바꿈 `<br>` + 플레인 텍스트)로 통일. `Research Intern  \n  <li>Worked on...</li>` → `Research Intern<br>Worked on Text-to-Speech research, repurposing Whisper features as a continuous tokenizer for TTS`. CSS 변경 없음(`.experience-desc`는 원래 플레인 텍스트 흐름용으로 스타일되어 있었음).

### 2026-07-07 — SE-NeRF·RAIN-GS 실제 링크 채움 (Paper/Project Page/Code)
- 배경: 사용자가 두 논문의 실제 Paper/Project Page/Code URL을 제공 — SE-NeRF(arXiv 2312.01003, cvlab-kaist.github.io/SE-NeRF, github.com/cvlab-kaist/SE-NeRF), RAIN-GS(arXiv 2403.09413, cvlab-kaist.github.io/RAIN-GS, github.com/cvlab-kaist/RAIN-GS). 확인해보니 `index.html`·`cv.html`은 이미 대부분 채워져 있었고(둘 다 Project Page 링크가 타이틀에 있었음; RAIN-GS는 index.html에 4버튼 전부 존재), 빠진 건 (a) `CV_LATEX/CV.tex`의 두 `\pubitem` 링크 슬롯(둘 다 `{}` placeholder였음)과 (b) `index.html` SE-NeRF 카드의 Code 버튼뿐이었음.
- **`CV_LATEX/CV.tex`**: `[W1]`(SE-NeRF) 링크 슬롯 `{} % TODO: add links` → `{\publinks{https://arxiv.org/abs/2312.01003}{https://cvlab-kaist.github.io/SE-NeRF/}{https://github.com/cvlab-kaist/SE-NeRF}}`. `[P1]`(RAIN-GS) 링크 슬롯(빈 `{}` + 죽은 주석 `% {...Code}`) → `{\publinks{https://arxiv.org/abs/2403.09413}{https://cvlab-kaist.github.io/RAIN-GS/}{https://github.com/cvlab-kaist/RAIN-GS}}`(죽은 주석 줄 제거).
- **`index.html`**: SE-NeRF 카드의 `.publication-links` 블록(Project Page/PDF/arXiv 3버튼만 있었음)에 Code 버튼(`github.com/cvlab-kaist/SE-NeRF`) 추가. RAIN-GS는 이미 4버튼 전부 사용자 제공 URL과 일치 — 변경 없음.
- **`cv.html`**: 두 항목 모두 이미 `.pub-title`이 Project Page로 링크되어 있어(다른 항목들과 동일한 단일-링크 관례) 변경 없음.
- **PDF 동기화**: `latexmk -pdf CV.tex` 재빌드(2페이지) → `cp CV_LATEX/CV.pdf JiwonCV.pdf` → `md5` 일치 확인.

### 2026-07-07 — Transferability 논문 Project Page 링크 추가 + Whisper(Kakao) preprint 카드 3곳 모두 주석 처리
- 배경: 사용자 요청 — (1) `https://cvlab-kaist.github.io/UMM_Transferability/` 프로젝트 페이지 개설 → Paper/Project Page/Code 3개 링크 모두 이 URL로 임시 연결(실제 arXiv/코드 저장소 아직 없음). CV·HTML 모두 반영. (2) HTML의 Preprint 섹션에서 Kakao 관련 카드(Whisper) 1개 주석 처리. Plan mode에서 두 가지 확정: (a) Whisper 항목은 `CV_LATEX/CV.tex`(이미 이전 세션에 주석 처리됨`[P2]`, L227-233)와 동기화되도록 `index.html`·`cv.html` 양쪽 모두에서 주석 처리. (b) `index.html` 버튼 라벨/순서는 사용자 지정대로 **Paper, Project Page, Code** 순(=`\publinks` 매크로가 렌더링하는 문구와 동일하게, 기존 다른 카드들의 "Project Page/PDF/arXiv/Code" 4버튼 관례 대신 3버튼 사용 — arXiv 미등재 상태를 오인시키지 않기 위함).
- **`CV_LATEX/CV.tex`**: `[C4]`(Transferability) `\pubitem`의 3번째 인자(links)를 `{} % TODO: add links` → `{\publinks{https://cvlab-kaist.github.io/UMM_Transferability/}{https://cvlab-kaist.github.io/UMM_Transferability/}{https://cvlab-kaist.github.io/UMM_Transferability/}}`로 채움(3개 URL 모두 동일, 임시).
- **`index.html`**: Transferability 카드의 `.publication-desc-list` 뒤에 `.publication-links.buttons.field.has-addons` 블록 신규 추가 — Paper(fa-file-pdf) / Project Page(fa-globe-asia) / Code(fab fa-github) 3버튼, href 전부 위 프로젝트 페이지 URL. Whisper 카드(`id="Whisper"`, 기존 L492-529)는 전체를 `<!-- -->`로 주석 처리(다음 RAIN-GS 카드는 그대로 유지).
- **`cv.html`**: Transferability 항목의 `.pub-title` 텍스트를 `<a href="https://cvlab-kaist.github.io/UMM_Transferability/">…</a>`로 감쌈. Whisper `.publication-compact` 항목(기존 L269-288)을 전체 `<!-- -->`로 주석 처리(다음 RAIN-GS 컴팩트 항목은 유지).
- **PDF 동기화**: `latexmk -pdf CV.tex` 재빌드(3페이지, 기존 30pt Overfull 경고만 잔존·무해) → `cp CV_LATEX/CV.pdf JiwonCV.pdf` → `md5` 일치 확인.
- **후속(사용자 결정 대기)**: Transferability의 실제 arXiv/PDF·GitHub 코드 저장소가 공개되면 위 3개 링크(및 `\publinks` 3개 인자)를 실제 URL로 교체 필요 — 현재는 전부 프로젝트 페이지로 임시 연결된 placeholder 상태.

### 2026-07-06 — 논문 설명 bullet 시제 통일(과거형)·문법 수정 + HTML CV 동기화 (커밋 예정)
- 배경: 사용자가 `CV.tex`의 bullet 설명을 직접 더 상세한 버전(논문당 2불릿)으로 재작성 → 현재형/과거형 혼재. 요청: 시제 통일 추천 + 문법 검사. 이후 "HTML에도 이 상세 버전 반영" 승인.
- **`CV_LATEX/CV.tex` 시제·문법**: 모든 논문 bullet의 리드 동사를 **과거형**으로 통일(성과 서술 관례). 문법/오타 수정: [C4] `Investigate…can be transferred`→`Investigated how…transfer`, `UMM`→`unified multimodal models`, `proposed practical`→`proposed a practical`; [C3] `focus`→`focused`, `pareto-optimal`→`Pareto-optimal`; [C2] `Proposedapproach enables…`(오타)→`Enabled…, highlighting…`; [C1] `via proposed framework`→`via the proposed framework`(이전 라운드 `Emprically`→`Empirically`, `systemically`→`systematically`, `layer/heads`→`layers/heads` 유지); [W1] `Self-training framework that…`(동사 없음)→`Proposed a self-training framework…`; [P1] `Relax`→`Relaxed`. (사용자가 병렬 편집 중이라 매 편집 전 파일 재-read로 최신 상태 확인.)
- **HTML CV 동기화**: `index.html`·`cv.html`의 `.publication-desc-list`에 있던 이전 라운드의 짧은 1불릿 초안 6건을 CV.tex의 최신 2불릿 상세 버전으로 교체([C4]/[C3]/[C2]/[C1] 2불릿, [W1]/[P1] 1불릿). Whisper[P2]는 CV.tex에서 주석 처리(PDF 미포함)되어 있으나 HTML 카드는 유지 — 기존 tokenizer 설명 + "Work done…at Kakao." 2불릿 그대로 보존. 옛 초안 잔존 0건(grep), 양쪽 `publication-desc-list` 각 7건.
- **검증**: 로컬 서버(`:8123`)+헤드리스 Chrome로 `index.html`/`cv.html` 재렌더 — 모든 논문에 2불릿 상세 설명 표시, venue 배지·링크·썸네일 레이아웃 정상.
- **미완/후속(사용자 결정 대기)**: (a) CV.pdf가 사용자의 `\largesection`(\LARGE 헤더) 추가로 **3페이지**가 됨 — 2페이지 복원 여부. (b) `JiwonCV.pdf`(사이트 사본) 아직 미동기화 — CV.pdf 확정 후 `cp`. HTML 동기화만 이번에 선행 진행.

### 2026-07-06 — 각 논문에 Work 설명 bullet 추가 (`\pubitem` 6번째 인자) + CV 3중 동기화 (커밋 예정)
- 배경: 사용자 요청 — CV의 각 논문(Work)에 "무엇을 했는지" 간단한 bullet 설명을 붙이고 싶다. 사용자 제안대로 `\pubitem` 매크로에 **6번째 인자(bullets)** 를 추가하는 방향. 문구는 Claude가 초안 작성, HTML CV에도 미러링(사용자 선택). Plan mode에서 확정.
- **`CV_LATEX/CV.tex` 매크로 확장**: `\pubitem`를 5→6 인자로(`{label}{title}{links}{authors}{venue}{bullets}`). `\pub@desc`/`\@empty` 빈-인자 테스트를 위해 정의를 `\makeatletter…\makeatother`로 감쌈(파일 내 `\cventry`와 동일 패턴 재사용) — `#6`가 비면 리스트를 그리지 않음. `#6`는 기존 `cvbullets` 환경에 그대로 주입(신규 리스트 정의 없음). 7개 `\pubitem` 호출 각각에 `\item …` 설명 추가(Whisper[P2]는 tokenizer 설명 + "Work done…at Kakao." 2개 bullet). `\lastupdate` → `July 6, 2026`.
- **페이지 수 복원(2p 유지)**: bullet 8줄 추가로 Extracurricular 섹션이 3페이지로 밀림 → 스페이싱 미세조정으로 2페이지 복원. `cvbullets` topsep 2→1·itemsep 1→0, `\pubitem` 말미 `\vspace` 6pt→1pt(항목 간 간격은 `parskip`가 유지하므로 안전), `\titlespacing*{\section}` 10/6→8/4pt, `\pubgroup` 선두 `\vspace` 4→2pt, Education 두 학력 사이 `\vspace` 4→2pt. gs 렌더로 각 논문 bullet 위치·전체 2페이지 육안 확인.
- **`index.html`**: 각 논문 카드의 빈 `<p class="publication-description"></p>` 7개를 `<ul class="publication-desc-list"><li>…</li></ul>`로 교체(내용 삽입). Whisper는 기존 "Work done during a research internship at Kakao." 문구를 2번째 `<li>`로 보존(+ tokenizer 설명 `<li>` 추가). 빈 description 슬롯 잔존 0건(grep 확인).
- **`cv.html`**: 7개 `.publication-compact`의 `.pub-main` 안, `.pub-authors` 다음에 동일 `<ul class="publication-desc-list">` 삽입(문구는 index.html과 동일하게 동기화).
- **`css/index.css`**: `.publication-desc-list` 규칙 신규(em 기반 크기 → 홈 카드/CV 컴팩트 양쪽에서 자연 축소, `list-style:disc`로 Bulma 기본 마진/불릿 명시 재설정). `.publication-compact .publication-desc-list`로 CV 페이지만 약간 더 작게(0.85em, `--gray-500`).
- **PDF 동기화**: `latexmk -pdf CV.tex` 재빌드(2페이지, 기존 30pt Overfull 경고만 잔존·무해) → `cp CV_LATEX/CV.pdf JiwonCV.pdf` → `shasum` 일치 확인.
- **검증**: gs로 CV.pdf 2페이지 렌더(각 논문 venue 바로 밑 bullet·Extracurricular 2페이지 내 수용 확인), 로컬 서버(`:8123`)+헤드리스 Chrome로 `index.html`/`cv.html` 렌더 스크린샷(7개 논문 모두 설명 리스트 표시, Whisper Kakao 문구 보존, 레이아웃 비-cramped 확인). index/cv 각 `publication-desc-list` 7건.
- **미완/후속**: bullet 문구는 Claude 초안이므로 **사용자 사실관계 검토 필요**(부정확 시 수정). 논문당 bullet 개수(현재 대부분 1개, Whisper 2개)는 다중 `\item`으로 언제든 확장 가능.

### 2026-06-23 — 본문 폰트를 Lato로 교체 (커밋 예정)
- 사용자 요청: "지금 상태에서 font Lato로 바꿔봐줄수 있니?" — Noto Sans 유지로 확정했던 직전 결정을 뒤집고 Lato로 변경.
- `index.html` / `cv.html` Google Fonts import에서 `Noto+Sans:wght@400;500;600;700`를 `Lato:wght@400;700`로 교체. Lato는 Google Fonts에서 100/300/400/700/900 weight만 제공(500/600 없음) — 본문에서 강조용으로 쓰던 `font-weight:600` 규칙(`.venue-badge`, `.award-badge`, `.publication-title`, `.author-me`, `.experience-title`, `.cv-entry-title`, `.cv-subsection-title` 등)은 브라우저가 로드된 weight 중 가장 가까운 700으로 매칭해 렌더링됨 — 별도 CSS 수정 없이 자연스럽게 처리됨.
- `css/index.css`: `body`와 `.cv-page`의 `font-family`를 `"Noto Sans"` → `"Lato"`로 교체(이 두 곳이 Noto Sans를 참조하던 유일한 셀렉터였음, grep으로 확인).
- 헤드리스 Chrome으로 Lato 400/700 weight가 실제로 로드되는지(`document.fonts.check`/`document.fonts.load`) 검증, 두 페이지 스크린샷으로 렌더링 확인.
- 콘텐츠 불변 검증 통과(`index.html`/`cv.html` 텍스트 diff 0줄 — `<link>` href 속성과 CSS font-family 값만 바뀌어서 본문 텍스트엔 영향 없음).
- `STYLESEED.md` Type scale 섹션 갱신(Lato로 명시 + weight 매칭 설명 추가).

### 2026-06-23 — Hero 섹션 가로폭 정렬 + 미사용 폰트 import 정리 (커밋 예정)
- 배경: 사용자가 (1) "Hello, I'm Jiwon Kang" hero 섹션의 가로 크기가 다른 섹션(News/Experience/Publications)이랑 다르다, (2) Noto Sans가 안 먹히는 것 같다(영문 본문 폰트를 더 깔끔한 걸로 추천해달라)고 요청.
- **원인 1 (실제 CSS 버그)**: `index.html`의 hero가 `<section class="hero"><div class="hero-body container is-max-desktop">` 구조였음. Bulma `.hero-body`는 자체 `padding:1.5rem`(상하좌우 24px)을 갖는데, 이게 이미 `max-width:1152px`로 제한된 `.container.is-max-desktop`과 같은 박스 안에 적용되어 있었음. 반면 News/Experience/Publications는 `<section class="section">`(Bulma `padding:3rem 1.5rem`→태블릿 이상 `3rem 3rem`)이 패딩을 담당하고, 그 안의 `.container.is-max-desktop`은 자체 패딩이 없음 — 그 결과 hero 콘텐츠 폭이 다른 섹션보다 약 48px(24px×2) 좁았음. `cv.html`은 원래 `hero`/`hero-body` 없이 `<main class="cv-page container is-max-desktop">` 하나만 쓰고 있어 이 버그가 없었음.
  - 수정: `index.html`의 hero 래퍼를 `<section class="section"><div class="container is-max-desktop">`로 교체(News/Pub과 동일한 Bulma 클래스 조합). `.card-hero` 자체 CSS는 변경 없음 — 래퍼만 맞추면 자동으로 폭이 일치.
- **원인 2 (폰트 버그 아님, 보는 방식 문제)**: 헤드리스 Chrome으로 직접 검증 — `index.html`을 `file://`로 더블클릭해서 열면, 루트 절대경로(`/css/...`)가 저장소 폴더가 아니라 파일시스템 루트(`/css/index.css`)를 가리켜서 **`bulma.min.css`/`index.css`/`fontawesome.all.min.css`가 전부 404** → 커스텀 CSS가 하나도 안 먹혀서 `body{font-family:"Noto Sans"}` 규칙조차 적용 안 됨 → 브라우저 기본 폰트로 렌더링되어 "화려하다"고 느껴진 것. `document.fonts.check()`로 재확인한 결과, `python3 -m http.server`로 정상 서빙하면 Noto Sans가 정확히 로드/적용됨 — 폰트 선택 자체는 문제 없었음. 사용자 확인 후 Noto Sans 유지로 결정.
  - 정리: 그 과정에서 발견된, 어디서도 참조되지 않는 미사용 Google Fonts import `Cormorant+Garamond`/`Crimson+Pro`를 `index.html`에서 제거(`cv.html`은 원래부터 Source Sans 3/Google Sans/Noto Sans만 import해서 이미 깨끗했음 — 이제 두 페이지 폰트 import가 동일해짐).
- 콘텐츠 불변 검증 통과(텍스트 diff: 기존 Round 1/2 diff와 동일, 신규 drift 0건). `css/index.css`는 변경 없음(`.hero`/`.hero-body` 셀렉터는 원래 거기 존재하지 않았음 — 순수 HTML 마크업 수정).

### 2026-06-23 — StyleSeed Round 3: 카드 가시성 + 논문 카드 줄간격 리듬 수정 (커밋 예정)
- 배경: 사용자가 논문 카드(ICLR "A Noise is Worth Diffusion Guidance") 스크린샷으로 피드백 — "1) line 간격이 너무 좁거나 불균형, 2) Card Design이 조금 더 눈에 띄면 좋겠어".
- **원인 1 (버그)**: Round 1에서 `--surface-page: #FAFAFA`를 정의했지만 실제로 `body` 배경에 적용한 적이 없었음 — page와 card(`--surface-card: #FFFFFF`)가 둘 다 흰색이라 카드 경계가 거의 안 보였음(있는 건 거의 안 보이는 그림자뿐). `body { background: var(--surface-page); }` 적용으로 해결.
- **원인 2 (불균형)**: `.publication-title`은 `margin:0`, `.publication-authors`는 `line-height` 자체가 없어 타이트한 기본값 상속, `.publication-venue`는 `margin-top:0`만 명시, 게다가 논문 7개 중 6개가 빈 `<p class="publication-description"></p>`를 갖고 있어 보이지 않는 한 줄 높이가 카드마다 일관성 없이 끼어듦. → `.publication-title`(`margin-bottom:0.5em`+`line-height:1.4`), `.publication-authors`(`line-height:1.65`+`margin-bottom:0.5em`), `.publication-venue`(`margin:0 0 0.5em 0`), `.publication-description:empty{display:none}`(보이지 않는 빈 단락 제거), `.publication-links`(중복 margin-top 제거)로 일관된 리듬 확립.
- **카드 강조**: `--shadow-card`를 더 또렷한 2-layer 그림자로 강화(여전히 StyleSeed 15% 상한 내), `--shadow-hover/elevated/modal`도 card<hover<elevated<modal로 단계적으로 재조정. `.card-hero`/`.card-section`/`.publication-card`에 `1px solid var(--border-subtle)` 테두리 추가(그림자만으론 약했던 경계를 보강). `.publication-card`에 hover 시 `translateY(-2px)`+`--shadow-hover` 살짝 뜨는 효과 추가("Snap" 모션 토큰 재사용, `prefers-reduced-motion`에서 자동 비활성).
- 전부 `css/index.css`만 수정(HTML 변경 없음) — 콘텐츠 불변 검증 통과(텍스트 diff Round 1/2와 동일, 신규 drift 0건).
- **`STYLESEED.md`** Shadow language 섹션을 새 값으로 갱신 + 배경/테두리/hover 변경 사항 기록.

### 2026-06-23 — StyleSeed Round 2: 토스(Toss) 스타일 컬러 복원 (커밋 예정)
- 배경: Round 1(StyleSeed 전면 적용) 결과에 대해 사용자 피드백 — "색이 너무 단조롭고 구분이 명확하지 않다". 구체적으로 (1) 논문 venue 배지가 전부 회색이라 한눈에 구별 안 됨, (2) 전체적인 톤이 너무 절제되어 있음. Toss 디자인(비비드한 토스 블루 + venue별 파스텔 카테고리 칩)을 참고 방향으로 제시. Plan mode에서 3가지 lock: **accent를 Toss Blue `#3182F6`로 교체**, **venue 배지는 파스텔 톤으로 색 구별 복원**(과거의 채도 높은 rainbow가 아닌 연한 배경+진한 텍스트), **"Under Review"는 그대로 회색 유지**(확정 venue=카테고리는 색을 갖지만, 심사중 상태는 StyleSeed의 "status=심각도, 평상시는 회색" 원칙을 그대로 따름).
- **`css/index.css` accent 토큰 교체**: `--brand` `#1A73E8`→`#3182F6`, `--brand-tint`→`#EAF2FE`, `--brand-hover`→`#1B6FE0`, `--brand-active`→`#1559B8`. Round 1에서 모든 accent 사용처가 이미 이 변수로 중앙화되어 있어 토큰 값만 교체(구조 변경 없음) — 링크·버튼·NEW 태그·focus ring·`.new-publication` 테두리·`:target` 하이라이트·nav active 등 전부 자동 반영.
- **venue 배지 파스텔 팔레트 복원**: 기존에 삭제했던 `.venue-badge.venue-{cvpr,iccv,iccvw,eccv,neurips,iclr,iclrw,icml,aaai,wacv,ipiu,arxiv}` 10규칙을 연한 배경+진한 텍스트 페어로 재추가(예: CVPR `#E3EEFD`/`#1454B8`, ECCV `#DFF5F1`/`#0E7A68` 등, 전체 표는 `STYLESEED.md` 참고). 베이스 `.venue-badge`(무수정자, 즉 "Under Review")는 그대로 중립 회색 유지. `index.html`의 `venue-{name}` 클래스는 Round 1부터 마크업에 그대로 있어 HTML 변경 없이 CSS만 추가.
- **News 아이콘 배지 신규**: Round 1에서 이모지→FA 아이콘으로 바꾼 뒤 단색 회색이었던 3개 아이콘(briefcase/trophy/graduation-cap)에 토스식 원형 틴트 배지(`.news-icon-badge--{blue,amber,violet}`) 추가 — venue 팔레트와 동일 색상군 재사용(새 색 추가 없음), "전체 톤이 단조롭다" 피드백 보강.
- **`STYLESEED.md`**: Key Color를 Toss Blue로 갱신, "Venue taxonomy palette" 섹션 신규 추가(파스텔 표 + "Under Review는 회색 유지" 근거 명시 — accent=인터랙션, venue 팔레트=분류라는 두 색 체계가 공존함을 문서화).
- `cv.html`은 변경 없음(논문 목록이 배지 없는 텍스트 행이라 영향 없음; accent 토큰 변경만 사이트 전체에 자동 반영).
- **검증**: 헤드리스 Chrome 데스크톱/모바일 재스크린샷으로 venue별 색 구분·"Under Review" 회색 유지·NEW 태그/링크 토스 블루 확인. `grep`으로 팔레트 표 외 잔존 hex 0건. 텍스트 콘텐츠 diff로 논문 데이터 무변경 재확인(Round 1과 동일 절차).

### 2026-06-23 — StyleSeed 디자인 규칙 전면 적용 (구조까지) (커밋 예정)
- 배경: 사용자가 외부 디자인 시스템 "StyleSeed"(`https://styleseed-demo.vercel.app/llms-full.txt`) 규칙을 사이트 전체에 적용 요청. StyleSeed는 모바일 대시보드/SaaS 앱(카드 전부, KPI 그리드, 도넛 차트, bottom nav)을 가정하므로, 차트/KPI/폼/430px 모바일 프레임 등 해당 없는 항목은 의식적으로 스킵하고 명시(`STYLESEED.md` 참고). Plan mode에서 사용자와 핵심 색상·모션 스타일을 먼저 lock: **accent `#1A73E8`**(Academic Blue, 기존 CVPR 배지 색)·**venue 배지 전부 그레이스케일 통일**(rainbow 금지 규칙 엄격 준수)·**모션 "Snap"**(100–150ms, ease-out, 최소 움직임)·**구조까지 전면 적용**(모든 콘텐츠를 카드 안에).
- **`css/index.css`**: `:root`에 토큰 추가(accent/tint, 5단계 그레이스케일 `--gray-900~300`, surface, radius `--radius-pill/sm/lg/card`, shadow `--shadow-card/hover/elevated/modal`, motion `--dur-fast/normal` + `--ease-snap`). 기존 `#000/#111/#171717/#333/#404040/#555/#666/#888/#4a4a4a` 등 산발적 회색·검정 리터럴을 전부 토큰으로 교체("no pure black" 준수). 라이트박스 모달 그림자가 50% 투명도였던 것을 `--shadow-modal`(12%)로 완화. venue 배지 12개 컨퍼런스별 색상 규칙 삭제 → 단일 회색 pill(`--surface-inactive` bg + `--gray-700` text)로 통일, award 배지도 동일하게 중립화. `.new-publication`의 노란 배경(`#fffbea`/`#fff3cd`)을 accent-border-only로 교체(accent 채움 금지 규칙 준수), `:target` 스크롤 하이라이트도 노랑→`--brand-tint`로. `NEW !` Bulma 빨간 태그 → `.tag-new`(accent 텍스트+tint 배경, 채움 없음).
- **카드 시스템**: `.card-hero`/`.card-section`/`.publication-card`(StyleSeed Four Section Types A/B/D) 추가. `index.html`의 Hero·News·Experience를 각각 카드로 감싸고, 7개 논문 항목(`.publication-block`→`.publication-card`로 rename)을 각각 카드화. News 앞 `<hr>` 구분선 삭제(섹션 간 구분자 금지 규칙, 카드 margin이 대신함). `cv.html`도 헤더(`.card-hero`, CV는 24px 압축 패딩)·Education/Experience/Publications/Extracurricular(각각 `.card-section`)에 동일 적용. **`CLAUDE.md`의 "재사용" 가이드와 충돌하는 유일한 의도적 예외**로 명시(`.publication-block`→`.publication-card` rename).
- **접근성/모션**: `:focus-visible` 전역 규칙 신규(이전엔 전무). `.publication-image`(라이트박스 트리거)에 `tabindex="0" role="button" aria-label`+`js/index.js`에 Enter/Space 키다운 핸들러 추가(이전엔 마우스 전용). 터치 타겟 44×44px 확보(라이트박스 닫기 버튼, news 토글, footer 아이콘). `prefers-reduced-motion` 전역 CSS 오버라이드 신규 + News "Show All" jQuery `slideDown/slideUp`은 CSS 미디어쿼리가 안 먹으므로 `js/index.js`에 `matchMedia` 분기 추가.
- **이미지 에러 상태**: 7개 논문 썸네일에 `onerror` 추가(깨지면 기존 `images/placeholder.svg`로 폴백 + `.img-fallback` 클래스로 살짝 dim). 라이트박스에 주입되는 `<video>`에 `error` 핸들러 추가("Video unavailable." 폴백, 비동기 로딩이 없는 정적 사이트에 맞는 유일한 실질적 에러 상태).
- **이모지 → 아이콘**: News 항목의 💼/🎉/🏫 이모지를 Font Awesome 아이콘(`fa-briefcase`/`fa-trophy`/`fa-graduation-cap`, `currentColor`)으로 교체 — StyleSeed Golden Rule #11(UI 아이콘에 이모지 금지) 준수.
- **기타 정리**: APPLE 항목에 남아있던 `.new-publication`(ECCV 추가 시 Transferability로 NEW 이전했는데 클래스만 잔존) 제거. Whisper venue의 인라인 `style="background:#607d8b"` 제거(전역 회색 pill로 통일).
- **`STYLESEED.md`**(신규, 루트): 실제 적용된 값(accent/grayscale/radius/shadow/motion/섹션타입 매핑/스코프 제외 항목/콘텐츠 불변 조건)을 StyleSeed 템플릿대로 기록.
- **`CLAUDE.md`**: `.publication-block`→`.publication-card` rename 및 신규 토큰/카드 시스템 도입을 반영하는 문장 추가.
- **검증**: 헤드리스 Chrome으로 데스크톱(1280px)·모바일(390px) 스크린샷 렌더 확인(카드·배지·NEW 태그·아이콘 정상). `grep`으로 pure-black/rainbow 잔존 0건, `--gray-400`(AA 경계) 사용처 점검 후 본문/네비 링크 색을 `--gray-500`로 교정(AA 통과). **콘텐츠 불변 검증**: HTML 태그 제거 후 텍스트 diff — `cv.html`은 byte-for-byte 동일, `index.html`은 의도된 변경(이모지 제거, "NEW !"→"NEW")만 차이 확인, 논문 제목/저자/링크/venue 텍스트는 전혀 변경 없음.
- **스코프 제외(StyleSeed 규칙 중 미적용, `STYLESEED.md`에 명시)**: KPI 그리드, 도넛/바 차트, bottom nav, input 필드 규칙(폼 없음), 430px 모바일 프레임 가정, 로딩 스켈레톤(비동기 데이터 없음). `CV_LATEX/CV.tex`/`JiwonCV.pdf`는 LaTeX 고유 디자인이라 이번 작업 범위 밖(미수정).

### 2026-06-23 — Transferability 논문 ECCV 2026 accept 반영 (커밋 예정)
- 배경: "Transferability Between Understanding and Generation in Unified Multimodal Models" 논문이 **ECCV 2026**에 accept. 기존엔 3곳(`index.html`, `cv.html`, `CV_LATEX/CV.tex`) 모두 "Under Review" preprint 취급 → accepted-conference 섹션으로 승격.
- `images/image.png`(사용자가 채팅에 첨부한 실제 티저, 1954×928·934KB) → `images/Transferability.png`로 rename, 기존 `placeholder.svg` 대체.
- `index.html`: Transferability 블록을 "Preprints & Under Review"에서 `2026` 섹션 최상단(APPLE보다 위)으로 이동. venue를 `Under Review`(인라인 회색) → `<span class="venue-badge venue-eccv">ECCV</span>`로 교체. **`NEW !` 태그를 APPLE → Transferability로 이전**(가장 최근 accept이므로). 링크 버튼은 아직 arXiv/project page 없어 미추가.
- `cv.html`: 동일 블록을 "Preprints & Under Review" → `International Conference` 서브섹션 최상단(APPLE보다 위)으로 이동, `pub-venue`를 `Under Review` → `ECCV, 2026`로 변경.
- `CV_LATEX/CV.tex`: `\pubitem{[P2]}`(Preprints 그룹) → `\pubitem{[C4]}`로 relabel, `\pubgroup{International Conference}` 최상단(`[C3]` APPLE보다 위)으로 이동. venue를 `Under Review.` → `European Conference on Computer Vision (ECCV 2026).`로. Preprints 잔여 항목 재번호화(`[P3]` Whisper → `[P2]`, `[P1]` RAIN-GS 유지)로 번호 공백 제거. `\lastupdate`를 `June 4, 2026` → `June 23, 2026`으로 갱신.
- `latexmk -pdf CV.tex` 재빌드(2p, 기존 7건 Overfull 30pt 경고만 잔존·무해) → `JiwonCV.pdf` 재동기화(shasum 일치).
- 검증: ghostscript로 CV.pdf p1/p2 렌더 확인(`[C4]` 최상단·ECCV 텍스트·`[P2]`/`[P1]` 공백 없음), 로컬 서버로 `index.html`/`cv.html`/`images/Transferability.png` 200 확인, grep으로 NEW! 태그 1개·venue-eccv 1건·순서 확인.

### 2026-06-06 — 검색 노출(SEO) 필수 메타데이터 추가 (커밋 예정)
- 배경: 기본 메타(title/description/OG/Twitter)·Google·Naver 검증파일은 이미 있으나, 색인·발견에 핵심인 4종이 누락. 사용자 요청으로 **필수 4종만** 추가(favicon·GA 제외).
- **`robots.txt`**(신규): 전체 크롤 허용 + `/blog-backup/`(아카이브 블로그) 제외 + `Sitemap:` 위치 명시.
- **`sitemap.xml`**(신규): `/`(priority 1.0)·`/cv.html`(0.8) 2개 URL, `lastmod 2026-06-06`(고정). `.nojekyll`라 루트에서 그대로 서빙됨.
- **canonical**: `index.html`→`https://loggerjk.github.io/`, `cv.html`→`…/cv.html` (중복 URL 정리, og:url와 일치).
- **JSON-LD schema.org `Person`**(index.html `<head>`): name·jobTitle·affiliation(KAIST)·alumniOf(Korea University)·knowsAbout·sameAs(Scholar/GitHub/LinkedIn). 값은 모두 페이지에 이미 있던 것 사용, 한글 별칭은 미표기(추측 회피).
- 검증: 로컬 서빙 robots/sitemap 200, sitemap XML well-formed, JSON-LD 유효(json.loads), canonical grep 각 1건 확인.
- **수동 후속(사용자)**: Google Search Console(검증 완료 상태)에서 `https://loggerjk.github.io/sitemap.xml` 제출 + 홈 "색인 요청". 실제 색인은 며칠 소요.

### 2026-06-06 — 프로필 사진 교체+다운스케일, CSS scale 조정, CV Education 불릿 재정렬 (커밋 완료 `1d11b149`)
- **프로필 사진**(`images/portrait.jpg`): 사용자가 새 사진(설산 배경)으로 교체. 원본 3024×4032 / 1.87MB로 과대(이전 110KB, 표시폭 ~290px 대비 17배) → 긴 변 1000px(750×1000)·q85로 **다운스케일 → 189KB**(2x retina ~580px에 충분히 선명, repo 이미지 경량화 컨벤션 유지). og:image·twitter:image는 동일 파일 참조라 자동 갱신.
- **`css/index.css`**: `.portrait img { transform: scale(0.75) → scale(1.0) }` (사용자 변경) — 새 사진을 프레임에 꽉 맞춤. 헤드리스 Chrome 렌더로 hero 정상 확인.
- **CV.tex Education 불릿 재정렬**(사용자 변경): "Double Major \quad GPA" 합쳐진 줄 → `GPA: 4.24/4.5` · `Double Major in Statistics` · `Undergraduate Research Intern @ CVLAB …` 3개 불릿으로 분리. IDE LaTeX 확장이 CV.pdf 자동 재컴파일 → `JiwonCV.pdf` 재동기화(shasum 일치). gs 렌더로 불릿 분리 + `\null` 정렬 동시 확인.

### 2026-06-05 — CV PDF `\cv@line` 빈 우측칸 줄 과대정렬(over-justify) 수정 (커밋 예정)
- 증상: Extracurricular의 "Co-founder and Vice President" 줄이 양쪽정렬로 단어 사이가 벌어지고 President가 우측 끝에 붙음(사용자 캡처). 같은 결함이 Education "Korea University, Seoul", Experience "Research Intern" 줄에도 잠복(모두 `\cventry` 4번째 인자=빈칸).
- 원인: `\cv@line`(CV.tex:83)이 `#1\hfill#2\par` 구조인데, `#2`가 비면 `텍스트\hfill\par`이 되고 TeX `line_break`가 **문단 끝 trailing glue 노드를 제거**(penalty로 변환)하여 매달린 `\hfill` 삭제 → `\parfillskip=0pt`라 끝 신축이 없어 단어 사이 glue가 늘어나 과대정렬. 우측칸이 채워진 줄(날짜·Advisor)은 마지막 노드가 텍스트라 정상.
- 수정: `\cv@line` 끝에 `\null`(폭0 박스) 추가 → 마지막 노드가 box라 `\hfill`이 삭제되지 않음. 빈 `#2`면 `\hfill`이 신축 흡수→좌측정렬, 채워진 `#2`면 날짜 우측 flush 그대로. 인접 주석에 이유 명시.
- 검증: `latexmk` 재빌드(2p, 신규 경고 없음; 기존 `\pubitem` 30pt Overfull 3건만 잔존, 무관). ghostscript로 p1/p2 PNG 렌더 → 3개 줄 좌측정렬·날짜 우측 flush 확인. `JiwonCV.pdf` 동기화(shasum 일치).

### 2026-06-04 — CV PDF Education/Experience 날짜 우측 정렬 수정 (미커밋)
- 증상: `CV.tex` Education·Experience의 오른쪽 날짜가 우측 여백까지 안 붙고 안쪽에서 끝남.
- 원인: `\cventry`의 `\textbf{#1}\hfill{#2}\par`에서 줄 끝 `\parfillskip`(fil)이 `\hfill`(fil)과 늘림을 나눠 가져 flush 안 됨.
- 수정: `\cv@line` 헬퍼 도입 — `{\parfillskip=0pt\relax\noindent#1\hfill#2\par}`로 `\hfill`이 전체 늘림을 가져 날짜가 우측 끝에 flush. `latexmk` 빌드 + `qlmanage` PNG 렌더로 정렬 확인. `JiwonCV.pdf` 갱신.
- 참고: `\makebox[\linewidth]`는 정렬은 되나 30pt Overfull 경고 → `\parfillskip=0` 채택(경고 없음). Publications `\pubitem` 30pt 경고 7건은 tabularx 트라이얼 산물(무해, 렌더 정상, 기존부터 존재).

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
