# Roadmap #989 standalone adapters

Algorithms live in independent packages. OmicVerse imports each backend only when called. Four distributions are released; two maintenance variants remain candidates. Base OmicVerse does not require these backends.

## Pinned installation and candidate status

The first four distributions are released on PyPI. The last two are independent
maintenance variants on release branches, **not yet published**. This integration
branch is preparatory; do not promote its six-package extra before both releases
and actual PyPI installation validation are complete.

| Import module | Distribution/version | Status |
|---|---|---|
| pyruvseq | py-ruvseq-omicverse==0.1.0 | Published |
| pysparkx | py-sparkx-omicverse==0.1.0 | Published |
| pymusic_deconv | py-music-omicverse==0.1.0 | Published |
| pymuscat | py-muscat-omicverse==0.1.0 | Published |
| ov_scranpool | py-scranpool-omicverse==0.1.0 | Candidate |
| ov_nebulosa | py-nebulosa-omicverse==0.1.0 | Candidate |

Install the released packages with normal pip commands. For local acceptance of
the two candidates, build each release branch with `python -m build` and install
the resulting wheel path. Do not replace them with PyPI `scranpy` (libscran's
different implementation) or `pynebulosa==0.1.1` (missing the required patches).
The `roadmap-ports` extra records all six exact candidate/release pins; it becomes
usable from PyPI only when the last two are published.

| Backend | Public API |
|---|---|
| py-ruvseq-omicverse | ov.bulk.ruvg / ruvr / ruvs |
| py-sparkx-omicverse | ov.space.sparkx |
| py-music-omicverse | ov.bulk.music_basis / music_prop |
| py-muscat-omicverse | ov.single.muscat_aggregate / muscat_pb_ds |
| py-scranpool-omicverse | ov.pp.scran_pooling |
| py-nebulosa-omicverse | ov.pl.nebulosa_density |

The latter two keep distinct Python namespaces so upstream installations can
coexist. They are maintained in `hurry060215-tech/scranPY` and
`hurry060215-tech/pyNebulosa` on separate release branches; the original small
upstream bug-fix PRs remain unchanged. Original authorship and licenses are retained.

## Scope

- RUVSeq: genes x samples. DataFrame controls are labels; array controls are zero-based positions. RUVr requires aligned residuals; RUVs requires zero-based replicate indices padded with -1. Use W in the DE design and original counts for count-model testing.
- SPARK-X: AnnData cells x genes, count layer and spatial coordinates. Sparse counts are transposed without densification. Returns gene-labelled tables and filtering masks without changing AnnData.
- MuSiC: genes x cells reference and genes x samples bulk DataFrames, with aligned cell type/donor Series. Only the default independent cell-type variance path is covered; check convergence diagnostics.
- muscat: sum pseudobulks by cluster and biological sample, retaining absent combinations. Explicit sample-indexed designs and coefficient-indexed contrasts. edgeR and limma-voom return local and per-contrast global BH; untestable clusters are reported.
- scran pooling: patched scranPY QR only, dense counts, >=101 cells per cluster. No implicit densification, CVXPY constraint, cluster splitting or small-cluster fallback. Returns factors without modifying input. Fix: sfortma2/scranPY#5.
- Nebulosa: patched wkde grid with required explicit bandwidth, e.g. R ks::hpi output. Default Silverman bandwidth and scipy's ks replacement are not selected. Returns per-cell density. Fix: mdmanurung/pyNebulosa#10.

## Examples

    W = ov.bulk.ruvg(counts_df, control_genes, k=2)["W"]
    svg = ov.space.sparkx(adata, layer="counts", spatial_key="spatial")
    pb = ov.single.muscat_aggregate(adata, cluster_key="cell_type", sample_key="sample", group_key="condition")
    ds = ov.single.muscat_pb_ds(pb, design, contrasts, method="limma-voom")

## Validation

Standalone repositories include frozen R oracles, export coverage manifests and four executed notebooks each. Numerical gates were fixed before implementation. muscat additionally uses the public Kang subset; synthetic fixtures do not establish biological accuracy.

    python -m pytest tests/test_roadmap_standalone_ports.py tests/bulk/test_ruvg.py -q
    python scripts/validate_roadmap_ports.py --ports-root PATH

The first command checks adapter contracts and optional RUVg integration. Missing real backends cause skips, not successful backend validation. The second exercises all six real backends against the standalone/fork fixture layout (py-RUVSeq, py-SPARKX, py-MuSiC, py-muscat, references/scranPY and references/pyNebulosa).
