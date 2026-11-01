# moscot result and workflow extensions



Existing solve entry points are retained. Three result helpers close identity and
storage gaps:

```python
problem = ov.single.moscot_temporal(adata, time_key='day', joint_attr='X_pca',
                                   solve_kwargs={'epsilon': 0.5})
key = next(iter(problem.solutions))
coupling = ov.single.moscot_coupling(problem, key)
coupling.write_h5ad('coupling.h5ad')
```

Rows are subproblem source cells; columns are target cells. For mapping this means
spatial observations × reference single cells. Values are transport mass; marginal
sums are saved on both axes. Dense materialization is bounded by `max_entries`.
The saved AnnData can be reopened without loading pickle or a native model.

```python
mapped = ov.space.moscot_mapping(reference, spatial, batch_key='slice',
    prepare_kwargs={'joint_attr': {'attr': 'X'}},
    solve_kwargs={'alpha': 0, 'epsilon': 0.5})
predicted = ov.space.moscot_impute(mapped, var_names=['gene1', 'gene2'])
predicted.write_h5ad('predicted.h5ad')
```

Use this identity-aligned helper rather than native moscot 0.5.2 `impute()` when
spatial batch rows are interleaved. Native subproblem pulls are assigned by cell
ID, not concatenated and relabeled. Missing/overlapping identities and nonfinite
outputs raise; no NaN-to-zero repair hides failures. These retain native target-marginal transport scaling, not
measured counts or source-row conditional means; constant features need not remain
constant for unequal source/target populations.

```python
aligned_problem = ov.space.moscot_alignment(slices, batch_key='slice',
    joint_attr='X_pca', solve_kwargs={'alpha': 0.5, 'epsilon': 0.5})
aligned = ov.space.moscot_align_coordinates(aligned_problem, reference='sliceA')
# aligned.obsm['spatial'] unchanged; new coordinates under 'spatial_aligned'.
aligned.write_h5ad('aligned.h5ad')
```

`examples/roadmap989/moscot_workflows.py` provides executable temporal, mapping and
alignment modes. It reads real input files, does not silently subsample, and
verifies save/reload identities. The temporal mode was executed on the official
moscot 250-cell, 1,479-gene WOT-derived test fixture. This preprocessed fixture is
not raw-data preprocessing validation. Its checksum is recorded; original input
bytes are not distributed because underlying dataset redistribution terms have
not been separately established. A later real alignment acceptance uses both author-provided PASTE breast slices
from pinned GitHub commit `5f0d58c67c7ad2b51ccdf67bad3c31df761fd9bc`: 505
input spots, 504 retained by explicit author QC, and 7,224 shared genes. Native
couplings and both warp/affine coordinates match exactly and survive H5AD.
Coordinates rely on the author-paired CSV row ordering (no barcode field in the
coordinate CSV). The reproducible script verifies all four input SHA256 hashes
and refuses to overwrite existing results. Mapping real-data acceptance is
recorded in `roadmap989-moscot-mop-mapping.md`: full retained-population linear
OT passes native parity, while both attempted quadratic rank64 configurations
failed native convergence and remain explicitly unvalidated.
No biological lineage or registration ground truth is claimed.


This separate draft includes the unchanged adapter prerequisite from PR #1008. No existing PR branch was modified. Historical native tests and real-data receipts are dated 2026-10-31; publication-time syntax checks do not imply a fresh rerun of every dataset.
