# Pooling size factors through a local Python backend

`ov.pp.scran_pooling(adata, layer=None, key_added='size_factors', copy=False, **kwargs)` forwards to `scran_pooling.compute_pooled_factors`. It returns AnnData, writes/overwrites the selected obs column, and does not change expression values. `copy=True` returns a new object. No new clustering, filtering, normalization or solver is implemented in OV.

Install the local `scran-pooling-py==0.1.0.dev0` wheel plus the supplied `pylimma==0.2.0+limpa.3`, `pyedger==0.2.1+limpa.3`, and `ashr-py==0.1.0.dev0+limpa.3` wheels. These are local review artifacts, not advertised PyPI releases. Follow the backend's `INSTALL_QUALIFIED.md` and exact conda/Python locks. This independent branch does not declare a `scran-pooling` extra. These unpublished backend artifacts are NOT included in this PR or publicly fetchable from this repository; installation is blocked until the reviewed artifacts are supplied separately. The backend needs Python >=3.10; the exact qualified profile uses Python 3.12.15, NumPy 2.5.4 and SciPy 1.18.1 with pinned numerical libraries. Installing equal version numbers from a different binary source does not establish the same qualification.

The backend is scoped Python reconstruction of scuttle 1.16.0 pooling and cleaner functions, with no new compiled extension or runtime R. NumPy/SciPy retain normal native scientific libraries. The reused local QR dependency chain is material (31,669 Python lines), not just NumPy/SciPy. GPL and derivative notices remain in the separate backend/dependency wheels; see their SOURCE_AND_LICENSES.md.

AnnData input is cells x genes. `layer=None` selects X even if a counts layer exists; use `layer='counts'` explicitly. Pass raw count data intentionally. `clusters` is a vector in current cell order, not an obs-column name. Numeric `subset_row` and `ref_clust` use zero-based positions; named gene subsets require unique var_names. Options and errors/warnings are passed to the backend without a fallback. An unsuccessful LSQR stop state raises before factors return. If desired, normalize/log-transform separately after reviewing factors; this function does neither.

The standalone qualified Linux profile passes 116 frozen R cases at strict absolute error <1e-8, rtol0. Those 105 development and 11 independently seeded pre-frozen cases were reused during development. The same source with ordinary PyPI NumPy/SciPy has 3 internal NLS failures (maximum coefficient difference 2.5647e-7); this limitation remains. The historical pre-2026-10-31 report (not freshly rerun) states that full original PBMC3k (2700 cells x32738 genes) exercises all17 pool sizes and agrees with R to6.9651e-12. A matching OV/native comparison tests the adapter, not biological validity or all possible numerical inputs.

The independent tutorial repository contains `docs/Tutorials-single/t_scran_pooling.ipynb`; it reads already-local PBMC3k files and compares the original backend to OV without downloading data. `scran-pooling-normalization` in the independent skills repository gives a small synthetic executable example. Ordinary Jupyter socket transport, other platforms, delayed/backed input, parallel operation and other scran/scuttle functionality are not qualified here. Stock OV's existing import-time analytics behavior predates this adapter; examples do not permanently monkeypatch network libraries.

Source, direct-wheel, independently source-built-wheel and independent final installed runs each passed27 tests in the inherited qualified OV profile. The6-cell full PBMC3k tutorial and3 skill blocks passed against both source and final installed OV. Genuine backend-absence probing passed3 tests with1 scientific module skip. Direct and source-built OV wheels contain identical runtime files. The separate standalone backend also passed a fresh self-contained qualified installation; that distinct result does not make this inherited OV environment self-contained. See the accompanying validation report for exact artifacts, timings and remaining ordinary-PyPI failures.

## Fresh cloud audit, 2026-10-31

The current integration restores the previously reviewed thin adapter rather
than copying its scientific algorithm into OmicVerse. The restored backend wheel
was checked against its source Python files. A fresh exact-lock Linux prefix
passed 21 backend pytest tests, including all 116 frozen R-reference cases at
atol 1e-8 / rtol 0. These fixtures are reused reference evidence, **not a new R
run or blind validation**. The ordinary current PyPI environment reproduced
113/116 cases; three NLS cases still differ by up to 2.56e-7. This is not a
portable parity pass and the tolerance was not changed.

The OmicVerse adapter passed all 27 tests in the current combined checkout.
A fresh full PBMC3k rerun was attempted but the official data download returned
HTTP 403, so the earlier real-data report is historical, not rerun here.
The exact backend numerical environment and four local development wheels
remain required; do not replace them silently with unrelated PyPI packages.

Offline import requires the separately proposed analytics opt-out PR, then setting
`OMICVERSE_DISABLE_ANALYTICS=1`. This independent branch does not itself implement
that switch; do not assume the environment variable stops stock upstream telemetry.
