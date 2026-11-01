# Real MOp mapping acceptance

This example uses the official [Tangram tutorial inputs](https://tangram-sc.readthedocs.io/en/latest/tutorial_link.html), but **runs moscot, not Tangram**. It compares direct native `MappingProblem` execution with the OV adapter and result helper. This is adapter/computational acceptance, not independent algorithm verification or proof of biological accuracy.

## Population and split

- Original reference: 26,431 nuclei × 27,742 genes; original Slide-seq: 9,852 spots × 24,518 genes.
- All 26,431 nuclei and 7,750 spots retained. Exactly 2,102 spots have zero total among training markers and are excluded before fitting; their IDs are in the receipt. No random cell or spot subsampling.
- 252 shared curated markers, alphabetically sorted; every fifth marker is held out (51), leaving 201 training markers.
- Selected marker inputs are checked finite, nonnegative and integer. Each observation is normalized by **training-marker total only**, then `log1p(10000 * count / total)`. Heldout genes are absent from the normalization denominator, fitting features and reference PCA.

## Reproduce

Download the three official URLs recorded in the receipt, decompress the two H5AD files, then run in an environment with the documented OV checkout and moscot dependencies:

```bash
export OMICVERSE_DISABLE_ANALYTICS=1
python examples/roadmap989/moscot_mapping_mop.py DATA_DIR NEW_OUTPUT_DIR
# Reproduce the separate low-rank profile (did not converge in this environment):
python examples/roadmap989/moscot_mapping_mop.py DATA_DIR NEW_QUADRATIC_OUTPUT_DIR --alpha 0.5
```

Output directories must be absent or empty. No data are downloaded implicitly. Actual decompressed inputs and downloaded archives are hashed. The example does not redistribute raw input data.

## Verified linear result

`alpha=0`, epsilon 0.05, threshold 1e-5, at most 1,000 iterations, cost batches of 512; full-rank linear OT. Both native and OV converge. Native and OV heldout predictions agree exactly (maximum absolute difference 0) on the recorded Linux environment, passing `atol=1e-6, rtol=1e-5`. Prediction IDs, gene order, coordinates and H5AD value roundtrip are checked. The full 7,750 × 26,431 coupling is deliberately not materialized; the dense-export memory guard is checked instead.

**Heldout median Pearson correlation is only about 0.0861**, so this is weak predictive agreement and is not biological validation. Linear alpha=0 ignores spatial coordinates in the objective. Imputed values retain native target-marginal transport scaling, not measured counts or conditional expected expression.

The optional alpha>0 path uses training-only 30-component PCA, epsilon0 (the backend recommendation) and rank64 approximation; its results must be reported separately and cannot be called full-rank quadratic parity. See the accompanying run receipt for completion/convergence status rather than assuming this optional configuration passed.

A preceding rank64/alpha0.5/epsilon0.05 trial did **not** converge in 1,000 iterations and was stopped before OV comparison. Its failure log is retained; no prediction from that trial was accepted.

## Quadratic result: not accepted

Both rank64/alpha0.5 profiles (epsilon0.05 and the backend-recommended epsilon0) returned native `converged=False` with threshold1e-5 and a 1,000-iteration budget. Both stopped **before** OV comparison and prediction export. These failures are preserved in `roadmap989-moscot-mop-quadratic-failures.json` and the run logs. Thus **real-data quadratic mapping remains unvalidated**; the real-data success above is linear mapping only. This is not an assertion that quadratic mapping always fails, nor an identified OV wrapper defect. Tolerance was not relaxed to obtain a pass.
