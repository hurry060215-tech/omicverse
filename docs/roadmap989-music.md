# MuSiC default weighted and two-level core

`ov.bulk.music_deconvolution` consumes a genes-by-bulk-samples DataFrame and
an AnnData scRNA reference, with explicit cell-type and biological-donor columns.
It calculates donor-specific relative abundances and mean cell library sizes,
cross-donor variances, and the native iterative weighted nonnegative least squares
sequence. Counts are not silently normalized, log-transformed, imputed or pooled
across donors. `layer=None` uses X; specify `layer="counts"` when appropriate.

```python
result = ov.bulk.music_deconvolution(
    bulk_counts, reference, cell_type_key="cell_type", sample_key="donor",
    layer="counts",
)
result["proportions"].to_csv("music-proportions.csv")
result["diagnostics"].to_csv("music-diagnostics.csv")
```

`music_deconvolution_recursive` follows the native two-level `music_prop.cluster`
procedure. Explicit broad-group membership and per-group discriminative markers
are required. Marker genes are withheld from the broad-group fit; subtype
estimation uses the original residual adjustment, not a generic rescaling.
This is not an arbitrary tree traversal. The original subgroup defaults are
retained even when top-level iteration controls differ.

```python
result = ov.bulk.music_deconvolution_recursive(
    bulk_counts, reference, group_key="broad_group", cell_type_key="cell_type",
    sample_key="donor", layer="counts",
    group_cell_types={"immune": ["T", "B"], "stromal": ["fibroblast"]},
    group_markers={"immune": immune_markers, "stromal": stromal_markers},
)
```

## Evidence and limits

The numerical reference is official **MuSiC 1.0.0**, commit
`f21fe67f5670d5e9fca0ad7550abaae3423eb59c` (xuranw/MuSiC). Exact unmodified
`analysis.R`, `construct.R`, and `utils.R` were sourced in R with native nnls and
SingleCellExperiment. This is source-function validation, **not installation of
MuSiC's whole namespace**. The frozen JSON fixtures under tests/fixtures/music
were generated from those functions, not from the Python candidate.

Nine synthetic default-core cases cover missing donor/type combinations, markers,
custom cell sizes, zeros, bulk-gene reordering, sparse references and iteration
limits. The gate fixed before translation is absolute tolerance 1e-8 plus relative
1e-7; proportions agree to approximately 1e-15. Three additional dense/sparse/zero-bulk tests of the two-level native recursion
agree within 3.34e-16. A fresh official-R comparison on the full bundled pancreas
reference (1,097 cells, 25,453 genes) and all 89 GSE50244 bulk samples also passes:
18,325 common genes, six vignette cell types, maximum proportion difference
1.86e-14. All 89 samples converged. Maximum gene-weight absolute difference
4.21e-8 passes the predeclared combined absolute/relative gate. See the adjacent
machine-readable report. This is a real-data computational comparison, not proof
of ground-truth biological accuracy. No arbitrary cell/gene subsampling was used.

The native two-level recursion also passes on all 10 mouse-kidney bulk samples
against the entire **upstream-supplied 10,000-cell subset** (16,273 genes; 13 types,
4 groups): maximum proportion difference 5.59e-15. This is not the original
43,745-cell experiment. The current marker RData contains `Epith.marker` and
`Immune.marker`, not the old vignette's `IEmarkers` object; these were explicitly
mapped to epithelial C3 and immune C4, with empty single-cell-type C1/C2 markers.
Both R and Python used the same documented mapping.

Other operating systems, arbitrary solver degeneracy and full native package
installation are not yet qualified. Convergence
flags must be checked. Near-singular Gram systems fail rather than emitting
negative variances. Public reference datasets are not copied into this patch;
source IDs and checksums are retained for private reproduction.

MuSiC2, cross-cell-type covariance (`ct.cov=TRUE`), normalization/centering modes,
marker discovery, simulation and plotting utilities are **not implemented**.
This API never executes third-party `music-py` and does not claim whole-package
coverage. Singular or underdetermined designs fail explicitly; no ridge or
pseudoinverse substitutes a different estimator. Invalid cell types are reported;
recursive inference rejects unsupported types instead of silently dropping them.

## Provenance and licensing

Algorithm translation from [MuSiC](https://github.com/xuranw/MuSiC), Xuran Wang
and Jiaxin Fan, under GPL (>=3), compatible with OmicVerse's GPL-3 license.
Cite Wang et al., Nature Communications (2019), DOI
[10.1038/s41467-018-08023-x](https://doi.org/10.1038/s41467-018-08023-x).
