# Sample-aware pseudobulk differential state

This workflow is **not a full muscat port**. It reuses OmicVerse's existing
`single.pseudobulk` aggregation and vendored `pylimma`/`pyedger` engines. It
adds the missing sample-level orchestration, explicit contrast, covariates,
paired design, exclusion receipts and multiple-testing tables.

```python
import omicverse as ov
r = ov.bulk.pseudobulk_ds(
    adata, sample_key="sample_id", group_key="cell_type",
    condition_key="condition", control="control", treatment="treated",
    layer="counts", covariates=["age", "batch"],
    min_cells=10, min_samples=3, method="limma_voom",
)
r.results.to_csv("ds_results.csv", index=False)
r.qc.to_json("ds_qc.json", orient="records", indent=2)
r.pseudobulk.write_h5ad("pseudobulk_counts.h5ad")
for cell_type, design in r.designs.items():
    print(cell_type, design)
```

Use **biological samples** as the replicates. An arbitrary cell ID column cannot
establish independent sampling; callers must supply correct experimental
metadata. Technical replicates must be combined before this workflow. Every
sample must have exactly one condition and one value for each covariate.
Raw integer counts are required. The default is the explicit `counts` layer;
`layer=None` intentionally uses X and rejects non-integer/log data. Input
objects are not modified.

For paired data, supply `paired_by="subject_id"`, with a different sample ID
for each subject/condition specimen. A fixed subject block is fitted. Complete
pairs are selected independently within each cell type after QC; excluded
unpaired profiles remain listed in `qc`. More than one sample per subject and
condition is rejected. Covariates redundant with subject blocks or confounded
with condition cause a rank-deficiency error rather than a misleading test.

## Outputs and direction

`log2FC > 0` always means treatment minus control. Results include per-cell-type
BH `qvalue` and across-all-tested-genes/types BH `qvalue_global`. These are
separate multiplicity conventions, not substitutes. No adjustment is performed
over untested genes; missing backend p-values remain missing.

`qc` reports each cell type as tested or skipped, retained samples by condition,
excluded low-cell/zero-count profiles, incomplete pairs, and number of tested
genes. Missing sample/type combinations are **not invented as zero libraries**.
`designs` preserves the actual sample-order-aligned design matrix per tested
cell type. Fewer than two samples per arm is always insufficient; even two is
only a computational minimum, not a design recommendation.

## Engines and limitations

- `limma_voom` (default): existing Python TMM normalization, voom, weighted
  linear fit, explicit condition contrast and empirical-Bayes moderation.
- `edger_ql`: existing Python legacy QL, with design-based dispersion using
  `trend_method="none"`. This intentionally does **not** claim native edgeR v4
  defaults, bias-adjusted QL, or muscat parity. Robust QL is not implemented.
- Numeric covariates are centred; other covariates are dummy encoded. Cast
  numeric batch labels to strings/categories if a categorical effect is
  intended. Formula parsing, interactions, arbitrary multi-column contrasts,
  random effects, and full muscat result/API compatibility are not provided.
- Count aggregation remains in memory. No disk-backed large-cohort promise.
- Synthetic planted-signal tests exercise actual engines but are not biological
  validation or an independent numerical-equivalence certification.

For any downstream claim, inspect sample sizes, excluded groups, actual model
matrices, and which multiplicity convention was used.

## Cloud verification snapshot (2026-10-31)

The final workflow and opt-out implementation passed 26 DS tests plus six
privacy tests in an independent rerun. This includes actual limma-voom and
legacy QL computation, pairing, shuffled cells, reversed contrasts, missing
sample/type combinations, rank failures, raw-count validation, reserved-QC
column protection, generated design-column collisions, numeric categorical
metadata and conservative exact-integer aggregation bounds.

The complete packaged muscat example recovered from the prior source-review
artifact (1,556 cells, 1,267 genes) ran without cell/gene sampling. It produced
20 sample/type profiles, tested 4,842 gene/type rows across five cell types,
and passed H5AD count/metadata export-reload. Aggregated counts agreed exactly
with the frozen original-R aggregation projection. This is **not** a new R
execution, native muscat DS numerical parity, or a biological-result claim.

The workflow rejects reserved grouping/model names, preserves categorical
covariates (including numerically labelled categories), and rejects collisions
in encoded model-column names. To guarantee exact float64 aggregation, a
conservative `maximum_count * source_cell_count <= 2**53` bound is checked
before aggregation. Extreme artificial counts exceeding this bound must be
handled with an exact-integer aggregation implementation instead.
