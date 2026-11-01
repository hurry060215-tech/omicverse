# SPARK-X recovered backend and adapter

The existing, previously delivered `sparkx-py` 0.1.0 backend was recovered from
its private review archive and all 239 manifest hashes verified. The adapter is
available via `ov.space.svg(adata, mode="sparkx", layer="counts")`. Install the
separately supplied local backend wheel (NOT included in this PR and no public installation URL established); no unrelated similarly named PyPI package is
selected automatically. The optional backend is loaded only when requested.

The default uses native BY-adjusted q < .05 selection, capped by n_svgs; explicit
`selection="top_n"` is a ranking, not significance. `layer=None` uses X, coordinates
come from `obsm[spatial_key]`, and covariates require exact observation alignment.
Filtered genes retain their positions with NaN values and false selection flags.
Multi-library pooling is rejected. Native numerical outputs and provenance are
stored in var/varm/uns and survive H5AD serialization.

## Current cloud revalidation

- Recovered standalone wheel: **270 passed, 3 expected skips**, using the frozen
  official R outputs. Skips concern unavailable real-fixture covariates, not errors.
- Genuine upstream breast layer: **14,789 genes × 251 spots**, single and mixture
  (11 kernels) rerun. BY < .05 counts **708 / 509**, matching R. Worst BY absolute
  error **4.38e-12**, top-50 sets identical. This is frozen-R oracle comparison,
  not a new R oracle regeneration.
- Current adapter plus existing SVG regressions: **120 passed, 11 skipped**
  for unrelated optional backends; the 58 SPARK-X adapter tests all pass.
- Optional native backend is pure Python and slower than compiled R. This is not
  an acceleration claim or a port of SPARK's other models.
- Linux long-double behavior is covered; Windows/macOS remain unverified.
- The recovered package includes the executed notebook/tutorial/skill artifacts.
  Their original logs are historical, not fresh execution claims.

Real-data redistribution licensing was not independently established; those inputs
stay in the private validation archive, not in the repository patch. No remote
push, package publication or merge has been performed.
