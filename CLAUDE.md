# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working convention: CONTEXT.md (read this first)

This repo uses a `CONTEXT.md` running log. **At the start of any task, read `/CONTEXT.md`.** Record every change you make (what, where, why) in `CONTEXT.md` as you go.

**Before exploring or working inside another folder, first check whether that folder contains its own `CONTEXT.md`; if so, read it before doing anything else, then proceed.** Folder-level `CONTEXT.md` files (e.g. `blog-backup/CONTEXT.md`, `CV_LATEX/CONTEXT.md`) take precedence for work scoped to that folder.

## What this repository is

`loggerjk.github.io` — Jiwon Kang's academic personal homepage. It is a **hand-authored static site** (no site generator) served by GitHub Pages, modeled on the Bulma-based academic template at crepejung00.github.io. There is **no build step for the website**: the HTML you edit is exactly what ships.

The repo holds three independent parts:

1. **Static homepage (repo root)** — the live site.
2. **`blog-backup/`** — the previous Jekyll (Minimal Mistakes) tech blog, archived. Not served (see `.nojekyll` below). Leave it untouched unless explicitly restoring/migrating old posts.
3. **`CV_LATEX/`** — LaTeX source (`CV.tex`) that compiles to the CV PDF.

## Architecture / big picture

- **`.nojekyll`** at the root disables GitHub Pages' Jekyll build, so everything is served verbatim and `blog-backup/` is *not* processed. Consequence: edits to `index.html` / `cv.html` are the source of truth; there is nothing to "build" for the site.
- **Two pages, shared assets:** `index.html` (home: intro, News, Publications) and `cv.html` (full CV). Both pull from `/css` and `/js` and `/images` using **root-absolute paths** (`/css/...`), which only work because the site is deployed at a `*.github.io` root. Keep paths root-absolute.
  - `css/`: `bulma.min.css` + `fontawesome.all.min.css` are vendored; **`index.css` is the only file to edit for styling.** Academicons (scholar icon) loads from a jsDelivr CDN.
  - `js/`: `fontawesome.all.min.js` (vendored, injects SVG icons — this is why no `webfonts/` dir is needed); **`index.js`** is the custom script (News "Show All" toggle + publication teaser lightbox).
- **Publication entries** follow a fixed markup contract that `index.css` styles by class: `.publication-block` rows, venue pills `.venue-badge .venue-{cvpr,iclr,neurips,iccvw,arxiv,…}`, award pills `.award-badge`, and `.author-me` to bold Jiwon's name. The CV page uses the lighter `.publication-compact` markup. Reuse these classes rather than inventing new ones.
- **Toggle-by-comment pattern:** sections that don't apply yet are kept in the HTML inside `<!-- ... -->` so they can be switched on later. Currently commented out: the **Experience** section, **X/LinkedIn** buttons, **Google Analytics** snippet, and **favicon** links. Re-enable by removing the comment wrapper (and filling the `G-XXXXXXX` / URL placeholders).
- **Thumbnails:** `images/SE-NeRF.png` and `images/RAIN-GS.png` are real teasers (Jiwon co-authored those); every other publication uses `images/placeholder.svg`. Swap in real teasers by replacing the `src`.

## The CV exists in three places — keep them in sync

A change to publications/education must be mirrored across all three:
1. `CV_LATEX/CV.tex` → compiles to `CV_LATEX/CV.pdf` (the canonical CV).
2. `JiwonCV.pdf` (repo root) — the copy the site's "CV" button links to. Update it from the LaTeX output (see commands).
3. `cv.html` and the Publications block in `index.html` — the HTML rendering.

## Commands

```bash
# Preview the site locally (from repo root) — open http://localhost:8000
python3 -m http.server 8000

# Rebuild the CV PDF from LaTeX
cd CV_LATEX && latexmk -pdf CV.tex      # (latexmk is already in use here; or: pdflatex CV.tex)

# After rebuilding, update the site's linked CV copy
cp CV_LATEX/CV.pdf JiwonCV.pdf

# Deploy: GitHub Pages serves master directly — just commit & push, no build.
git add -A && git commit && git push origin master
```

There is no test/lint setup; this is a static site.

## Gotchas

- `google6fa06dcad1e159f1.html` and `naver…html` at the root are search-engine site-verification files — keep them at the root.
- Don't reintroduce Jekyll front matter or `_config.yml` at the root; `.nojekyll` means they won't be processed and would just be served as raw text.
- The string "Jaewoo Jung" / `crepejung00.github.io` appears in the site **only** as legitimate co-author credits/links (SE-NeRF, RAIN-GS, Transferability). It must not reappear as site identity (title, photo, analytics, contact).
