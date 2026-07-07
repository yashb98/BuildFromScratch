# Submission dossier -- qwen3-0.6b-study

Prepared 2026-07-07 by `/manuscript`. This package is **submission-ready source**, not a
submitted paper. The skill never creates accounts, obtains endorsement, or clicks submit
(contracts §C16). **A human performs every submission step -- nothing here auto-submits.**
Everything below tagged **[HUMAN]** is yours to do; **[SKILL]** is done; **[AUTOMATIC]**
happens on its own after you submit.

## What is in this package

```
qwen3-0.6b-study.tex     # master file: single-column article, 11pt, natbib/plainnat
sections/*.tex           # 18 sections (abstract ... conclusion), all required ones present
tables/*.tex             # 8 booktabs tables, \input from the sections
figures/                 # 7 referenced PDFs + their generator .py scripts + PNG previews
refs.bib                 # 26 references, every one fetched + verified (refs.bib.provenance.json)
qwen3-0.6b-study.bbl     # generated bibliography (26 bibitems) -- SHIPS WITH THE SOURCE
qwen3-0.6b-study.pdf     # compiled preprint, 34 pages (tectonic, 2026-07-07)
evidence_manifest.json   # every body number -> on-disk results file
claims_manifest.json     # claim <-> evidence audit
arxiv_package.tar.gz     # the upload artifact (contents below)
```

### Exact contents of `arxiv_package.tar.gz` (35 files, verified 2026-07-07)

- `qwen3-0.6b-study.tex` -- the master file
- `qwen3-0.6b-study.bbl` -- the generated bibliography (**required**, see note below)
- `refs.bib`
- `sections/` (18 files): `abstract.tex`, `introduction.tex`, `related.tex`,
  `methodology.tex`, `setup.tex`, `reproduction.tex`, `pretraining.tex`, `data.tex`,
  `midtraining.tex`, `sft.tex`, `rlvr.tex`, `failures.tex`, `discussion.tex`,
  `limitations.tex`, `reproducibility.tex`, `broader_impacts.tex`, `conclusion.tex`
- `tables/` (8 files): `tab_arms.tex`, `tab_attrib.tex`, `tab_data.tex`,
  `tab_hparams.tex`, `tab_midtrain.tex`, `tab_repro.tex`, `tab_rlvr.tex`, `tab_sft.tex`
- `figures/` -- ONLY the 7 PDFs the paper actually references
  (`grep -rn includegraphics sections/` matches exactly these):
  `fig_ppl_curves.pdf`, `fig_repro_gap.pdf` (reproduction), `fig_deconfound.pdf`
  (pretraining), `fig_data_anneal.pdf`, `fig_ecl_ladder.pdf` (midtraining),
  `fig_sft_confound.pdf` (sft), `fig_rlvr.pdf` (rlvr)

The figures directory on disk also holds `.py` generators, `.png` previews, and
legacy phase-1 figures (`fig1_*`, `fig2_*`, `fig4_*`, `fig5_*`,
`exploratory_prope_val_ppl.pdf`, `vibethinker_sft_reasoning_ppl.pdf`) -- none of those
ship; only the 7 referenced PDFs go in the tarball.

### Why the `.bbl` is in the tarball (do not remove it)

arXiv's AutoTeX does **not** run BibTeX -- it compiles your source as-is, so a package
without the pre-generated `.bbl` produces `[?]` for every citation. The shipped
`qwen3-0.6b-study.bbl` is the machine-generated output of the local
pdflatex/bibtex pass (26 bibitems; regenerated 2026-07-07 alongside the PDF, i.e.
*after* the last `refs.bib` edit -- verified 26/26 cited keys resolve). The
bibliography comment at the bottom of `qwen3-0.6b-study.tex` states this correctly
(fixed 2026-07-07; it previously claimed arXiv runs BibTeX). If you ever edit
`refs.bib`, rebuild so the `.bbl` regenerates, then re-pack.

Rebuild the tarball after any source change:

```bash
cd research/papers/qwen3-0.6b-study
tar -czf arxiv_package.tar.gz \
    qwen3-0.6b-study.tex qwen3-0.6b-study.bbl refs.bib \
    sections/*.tex tables/*.tex \
    figures/fig_ppl_curves.pdf figures/fig_repro_gap.pdf figures/fig_deconfound.pdf \
    figures/fig_data_anneal.pdf figures/fig_ecl_ladder.pdf \
    figures/fig_sft_confound.pdf figures/fig_rlvr.pdf
```

## Metadata block (paste into the arXiv form)

**Title**

> Reproduce, Then Attribute: A Controlled Study of the LLM Training Lifecycle on a
> Bit-Exact Qwen3-0.6B Reproduction

**Authors**

> Yash Bishnoi

(solo, no affiliation, no AI/tool credit -- carried over from the confirmed
qwen3-imu1-matched-compute decision; matches `\paperAuthor` in the master file)

**Abstract** (plain-text rendering of `sections/abstract.tex`, no content changes)

> Published improvements to language-model training rarely arrive with attribution:
> gains are reported for bundles of changes, at loosely matched compute, often from
> single runs, so a reader learns that a recipe won without learning why. Small-scale
> reproductions, where controlled re-testing is affordable, are rarely held to a
> controlled standard. This paper carries a bit-exact reproduction of Qwen3-0.6B
> (fp32 max |Δlogits| = 0.0 against the reference on the recorded probe) through the
> full training lifecycle -- pre-training, data composition, mid-training, supervised
> fine-tuning, and reinforcement learning with verifiable rewards -- on a single GB10
> machine, admitting a comparison only when exactly one variable differs, training
> FLOPs match within 5%, and the effect is a paired three-seed bits-per-byte (BPB)
> delta with a 95% confidence interval on two corpora. Under these gates, a
> modernization bundle's single-run, in-loop -17.9% validation-perplexity headline
> decomposes into three individually significant architecture effects plus an
> early-training optimizer effect, while its schedule and z-loss axes contribute
> nothing measurable; and a premium-data anneal improves code BPB by +0.2716 (95% CI
> [+0.2641, +0.2792], 3 seeds, paired) over an iso-token control that absorbs the
> learning-rate confound. Three nulls are reported at the same standard: SFT
> response-masking does not separate from its iso-FLOP control; GRPO at 0.6B confirms
> a pre-registered null against a random-reward gate (single seed, directional by
> design); and context extension is gated off by a step-0 diagnostic. The same gates
> caught the project's own errors, including an in-loop 0.68-perplexity "win" that
> collapsed to +0.009 on a fixed held-out set.

**Primary category**: `cs.LG` (empirical LLM-training study)
**Cross-list**: `cs.CL`
**License**: **CC BY 4.0** (Creative Commons Attribution 4.0 International) --
recommended: maximally reusable, required by many indexers, and consistent with the
public GitHub repo.
**Comments field**:

> 34 pages, 7 figures, 8 tables. Code, training logs, and evidence manifests:
> https://github.com/yashb98/BuildFromScratch

(Page count verified against the compiled `qwen3-0.6b-study.pdf` -- 34 pages.)

## [SKILL] Done for you (headless)

- [x] Bit-exact reproduction baseline (fp32 max |Δlogits| = 0.0 on the recorded probe).
- [x] All body numbers traced to on-disk files (`evidence_manifest.json`).
- [x] Claim↔evidence audit in `claims_manifest.json`; nulls reported as nulls
      (SFT masking, GRPO-at-0.6B, context extension); the in-loop 0.68-PPL confound is
      in the paper as a failure-catalogue entry, not a win.
- [x] 26 references assembled ONLY from fetch-verified APIs (provenance in
      `refs.bib.provenance.json`); the 4 transient API failures logged in
      `refs.unresolved.json` (2026-07-06, DBLP 500 / S2 429) were all subsequently
      resolved -- all 4 titles are present in the current `refs.bib`; 0 model-written
      entries remain.
- [x] Citation closure verified: 26 `\citation` keys in the `.aux` == 26 `\bibitem`
      keys in the `.bbl`, zero missing.
- [x] 7 figures regenerated from the actual results files by the `figures/*.py`
      scripts; tarball carries only the referenced PDFs.
- [x] PDF built locally with tectonic (2026-07-07), 34 pages.
- [x] `arxiv_package.tar.gz` packed (35 files, listing verified above).

## [HUMAN] Pre-submission checklist

Run all of this yourself before uploading -- the skill cannot self-certify.

1. **Clean compile from the tarball, not the working dir** (tectonic is installed at
   `~/.local/bin/tectonic`):

   ```bash
   SCRATCH=$(mktemp -d)
   tar -xzf research/papers/qwen3-0.6b-study/arxiv_package.tar.gz -C "$SCRATCH"
   (cd "$SCRATCH" && tectonic qwen3-0.6b-study.tex)   # must exit 0
   ```

   or via the skill's wrapper (fails on any undefined citation/reference):

   ```bash
   bash .claude/skills/manuscript/scripts/build_pdf.sh \
        research/papers/qwen3-0.6b-study/qwen3-0.6b-study.tex \
        research/papers/qwen3-0.6b-study
   ```

2. **No missing glyphs / undefined references**: scan the compile output --

   ```bash
   (cd "$SCRATCH" && tectonic qwen3-0.6b-study.tex 2>&1) \
     | grep -Ei 'missing character|undefined (citation|reference)' && echo BAD || echo CLEAN
   ```

3. **Evidence-manifest cross-check**: open `evidence_manifest.json` and spot-check the
   headline numbers against their listed source files -- at minimum the anneal delta
   (+0.2716, CI [+0.2641, +0.2792]), the -17.9% in-loop headline it decomposes, and
   the 0.68 -> +0.009 collapse. Every entry names the on-disk log/result it came from.

4. **Tarball freshness**: `tar -tzf arxiv_package.tar.gz` must list exactly the 35
   files above, and the `.bbl` must be newer than `refs.bib`
   (`ls -la qwen3-0.6b-study.bbl refs.bib`).

5. **LLM-artifact scrub** (arXiv issues bans for leftover model meta-comments):

   ```bash
   python3 .claude/skills/manuscript/scripts/check_llm_artifacts.py \
           research/papers/qwen3-0.6b-study
   ```

6. **Read the compiled 34-page PDF end to end.** No exceptions.

## [HUMAN] The verification attestation (required -- the skill cannot self-certify)

arXiv issues a one-year ban for unchecked LLM output (hallucinated references,
leftover model meta-comments). Before you upload, confirm and sign:

> I have personally verified that every reference in `refs.bib` resolves to a real
> paper, that every reported number matches the files named in
> `evidence_manifest.json`, and that there are no hallucinated references and no
> leftover LLM meta-comments in the source.
>
> Signed: ______________________  Date: __________

(Spot-check suggestion: open `refs.bib.provenance.json` and click 2-3 of the source
URLs; re-read the abstract numbers +0.2716 / [-]17.9% / 0.68 -> +0.009 against the
files listed in `evidence_manifest.json`.)

## [HUMAN] arXiv submission -- step by step

1. **Account + ORCID.** Ensure you have an arXiv account; create/link an ORCID if you
   want it on the paper.
2. **Endorsement (2026 policy).** A new/unaffiliated author needs a *personal
   endorsement* from an established arXiv author already eligible in the target
   endorsement domain (institutional email alone no longer suffices). Arrange this
   for **cs.LG** before submitting. See
   `.claude/skills/manuscript/references/publishing.md`.
3. **Start a new submission** and choose license **CC BY 4.0**.
4. **Upload `arxiv_package.tar.gz`.** It contains the TeX source, `refs.bib`, the
   generated `qwen3-0.6b-study.bbl` (required -- AutoTeX does not run BibTeX), and
   the 7 figure PDFs. Watch the AutoTeX processing log; the compiled preview must
   show resolved citations (no `[?]`) and all 7 figures.
5. **Metadata.** Paste the title, author, abstract, and comments line from the
   metadata block above. Primary `cs.LG`, cross-list `cs.CL`.
6. **Disclose generative-AI assistance** per arXiv policy.
7. **Preview, then submit.** Respond to any moderation hold or reclassification.
   Submission is a human action -- this package never submits itself.

## [HUMAN] After the arXiv ID is live

- **Hugging Face Papers:** self-index at `hf.co/papers/<arxiv-id>` (or
  `hf.co/papers/submit`), then claim authorship (HF admin validates).
- Add the arXiv/HF paper URL to the README of
  https://github.com/yashb98/BuildFromScratch so the Hub and the repo cross-link.

## [AUTOMATIC]

- **Google Scholar** crawls the arXiv abstract page and picks up correct metadata for
  free (Highwire `citation_*` tags incl. `citation_pdf_url`). No meta-tag editing
  needed.
