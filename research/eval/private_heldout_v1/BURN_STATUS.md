# `bfs-private-v1` — burn status

**Measured 2026-08-05.** This file records that the corpus's own
`burn_conditions[0]` has **already partially fired**, independently of any
decision to publish it.

## The condition

`MANIFEST.json` (`burn_conditions[0]`) states:

> Publishing this repository (the KaggleLab / GitHub-Pages arm) makes
> `bfs-private-v1` PUBLIC and voids the unseen-ness argument for every future
> model.

`github.com/yashb98/BuildFromScratch` is public (`"private": false`,
`"visibility": "public"`, created 2026-06-02).

## What is actually exposed

The slice's own files are **not** exposed — `MANIFEST.json`,
`private_prose_v1.txt`, `private_code_v1.txt` and `build_private_heldout.py`
are gitignored, and the first three return 404 on `main`.

But the slice is built from files **inside this repository**
(`build_private_heldout.py:149`, `REPO.rglob`), and most of those sources are
public. Intersecting the 175 manifest paths against the full recursive tree of
public `main` (921 blobs):

| shard | source files public | share of files | bytes public | share of bytes |
|---|---:|---:|---:|---:|
| `private_prose` | 38 / 84 | 45.2 % | 358,227 / 998,849 | 35.9 % |
| `private_code` | 73 / 91 | 80.2 % | 728,330 / 999,506 | **72.9 %** |
| **total** | **111 / 175** | **63.4 %** | **1,086,557 / 1,998,355** | **54.4 %** |

`private_code`, at 72.9 % public by bytes, should be treated as **effectively
burned**.

Publishing `MANIFEST.json` would complete the burn even though the manifest
contains no corpus text: its `files[]` array is in concatenation order, so
manifest + the public tree reconstructs a shard byte-for-byte. This was
verified — `private_code` rebuilt to a sha256 matching the pinned shard digest
exactly. There is no safe "metadata-only" middle path. Publish the builder
(the method); withhold the manifest and the shards.

## What this does and does not invalidate

**Unaffected:** the *temporal* unseen-ness argument for models already scored.
Repository text postdates any crawl in the training corpus, and
`train_qwen3.py:151` streams FineWeb-Edu with no pinned `revision`, whose newest
dump predates this repo's first commit (`84a96c0`, 2026-05-20).

**Affected:** every *future* use. Once sources are on the crawlable web, models
trained thereafter may have ingested them, and a BPB number on this slice stops
meaning "unseen".

## Required caveat

Any `base-eval` claim resting on `bfs-private-v1` must carry:

> Private held-out set; 54.4 % of its source bytes were already on a public
> GitHub repository at scoring time (measured 2026-08-05).

Anything stronger is not supportable.

## v2

Rotate from the reserved complement — 26 unused eligible prose files
(110 − 84) and 150 unused code files (241 − 91). Per
`build_private_heldout.py:56`, bump `SALT` and `CORPUS_ID` rather than re-running
v1.

**Add a `not_public_on_remote` filter to the builder's exclusion rules.** The
current rule set excludes generated content, model output and web-harvested
material, but has **no notion of "already pushed"** — which is precisely why
this happened.
