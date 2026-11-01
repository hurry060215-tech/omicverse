"""Sample-aware differential state analysis, not a reimplementation of muscat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.stats.multitest import multipletests


@dataclass
class PseudobulkDSResult:
    """Auditable DS outputs; absent sample/cell-type pairs are never invented."""
    pseudobulk: Any
    results: pd.DataFrame
    qc: pd.DataFrame
    designs: dict
    parameters: dict


def _design(metadata, condition_key, treatment, covariates, paired_by):
    design = pd.DataFrame({"Intercept": 1.0,
                           "treatment": (metadata[condition_key] == treatment).astype(float)},
                          index=metadata.index)
    for key in list(covariates) + ([paired_by] if paired_by else []):
        values = metadata[key]
        if key != paired_by and pd.api.types.is_numeric_dtype(values):
            values = values.astype(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite covariate: {key}")
            column = f"cov:{key}"
            if column in design:
                raise ValueError("Encoded covariate column labels collide; rename metadata columns/levels.")
            design[column] = values - values.mean()
        else:
            # Explicitly fix categories across sample orderings.
            encoded = values.astype(str)
            categories = sorted(encoded.unique())
            if len(categories) != values.nunique():
                raise ValueError(f"Categorical levels collide after string encoding: {key}")
            if len(categories) < 2:
                raise ValueError(f"Categorical covariate has only one observed level: {key}")
            for level in categories[1:]:
                column = f"cov:{key}={level}"
                if column in design:
                    raise ValueError("Encoded covariate column labels collide; rename metadata columns/levels.")
                design[column] = (encoded == level).astype(float)
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Design is rank deficient: condition is confounded with covariates or covariates are redundant.")
    if len(design) <= design.shape[1]:
        raise ValueError("Design has no residual degrees of freedom.")
    return design


def _fit(counts, design, method):
    from ..external import pyedger
    y = pyedger.calcNormFactors(pyedger.DGEList(counts, group=np.where(design["treatment"].to_numpy() == 1, "treatment", "control")))
    if method == "limma_voom":
        from ..external import pylimma
        effective = pyedger.effectiveLibSizes(y)
        v = pylimma.voom(counts, design.to_numpy(), lib_size=effective)
        fit = pylimma.lmFit(v.E, design.to_numpy(), weights=v.weights)
        contrast = np.zeros(design.shape[1]); contrast[1] = 1.0
        fit = pylimma.eBayes(pylimma.contrasts_fit(fit, contrast))
        table = pd.DataFrame({"log2FC": np.asarray(fit.coefficients)[:, 0],
                              "statistic": np.asarray(fit.t)[:, 0],
                              "pvalue": np.asarray(fit.p_value)[:, 0]}, index=counts.index)
    else:
        # Explicitly use the implemented legacy QL path, not edgeR v4's
        # approximate bias-adjusted QL port. Design-based dispersion supports
        # trend_method='none'; this differs from muscat/edgeR defaults.
        pyedger.estimateDisp(y, design=design, trend_method="none")
        fit = pyedger.glmQLFit(y, design=design, legacy=True)
        ql = pyedger.glmQLFTest(fit, coef="treatment")
        table = ql.table.rename(columns={"logFC": "log2FC", "PValue": "pvalue", "F": "statistic"})
    table.index.name = "gene"
    return table


def pseudobulk_ds(adata, *, sample_key, group_key, condition_key, control, treatment,
                  layer="counts", covariates=(), paired_by=None,
                  method="limma_voom", min_cells=10, min_samples=2,
                  min_count=10, min_total_count=15, alpha=0.05):
    """Aggregate raw counts per biological sample/cell type and test DS.

    Parameters
    ----------
    sample_key : str
        Biological sample (not cell ID or technical replicate) in ``obs``.
        Independent sampling cannot be inferred from labels: the caller is
        responsible for the design. Technical replicates must be combined.
    group_key : str
        Cell type or cluster in ``obs``. Each is analysed separately.
    condition_key : str
        Must have one constant value per biological sample.
    control, treatment : str
        Explicit contrast; positive log2FC means treatment minus control.
        Other conditions are excluded before aggregation and testing.
    layer : str or None
        Raw integer counts. None explicitly chooses X; no count inference.
    covariates : sequence of str
        Sample-constant covariates; numeric variables are centred and other
        columns dummy encoded. Confounded and saturated designs fail clearly.
    paired_by : str or None
        Subject ID for paired samples, fitted as a fixed blocking factor.
        Only complete pairs within each retained cell type are tested; more
        than one sample per subject/condition is rejected rather than treated
        as independent replication. Unpaired samples are not silently paired.
    method : {'limma_voom', 'edger_ql'}
        Uses existing vendored Python engines, not R/muscat. ``edger_ql`` uses
        legacy QL and no abundance trend for dispersion, explicitly differing
        from native edgeR/muscat defaults. No new numerical-equivalence claim.

    Returns
    -------
    PseudobulkDSResult
        Pseudobulk counts, gene-level tables, per-group exclusion/QC receipts,
        actual model matrices, and parameters. ``qvalue`` is BH within each
        group and ``qvalue_global`` is BH over every finite tested p-value.
        Missing/failed p-values stay missing. The input is unchanged.
    """
    if method not in {"limma_voom", "edger_ql"}:
        raise ValueError("method must be 'limma_voom' or 'edger_ql'.")
    if not isinstance(control, str) or not isinstance(treatment, str) or control == treatment:
        raise ValueError("control and treatment must be distinct string labels.")
    if isinstance(covariates, str):
        raise TypeError("covariates must be a sequence of column names, not one string.")
    covariates = list(covariates)
    if len(set(covariates)) != len(covariates) or condition_key in covariates or sample_key in covariates or group_key in covariates or paired_by in covariates:
        raise ValueError("Covariates must be unique and distinct from grouping and pairing columns.")
    if len({sample_key, group_key, condition_key}) != 3 or paired_by in {sample_key, group_key, condition_key}:
        raise ValueError("sample, group, condition and optional pairing columns must be distinct.")
    for name, value, minimum in [("min_cells", min_cells, 1), ("min_samples", min_samples, 2)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if not 0 < alpha < 1 or not np.isfinite([min_count, min_total_count]).all() or min_count < 0 or min_total_count < 0:
        raise ValueError("Invalid alpha or expression-filter thresholds.")
    required = [sample_key, group_key, condition_key] + covariates + ([paired_by] if paired_by else [])
    reserved = {"psbulk_n_cells", "psbulk_counts", "psbulk_props"}
    if reserved.intersection(required):
        raise ValueError("Grouping/model columns cannot use reserved pseudobulk QC names.")
    missing = [key for key in required if key not in adata.obs]
    if missing:
        raise KeyError(f"Missing obs columns: {missing}")
    if adata.obs[required].isna().any().any():
        raise ValueError("Missing labels/covariates are not permitted.")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell and gene identifiers must be unique.")
    for key in [sample_key, group_key, condition_key] + ([paired_by] if paired_by else []):
        if not all(isinstance(v, str) and v and '\x00' not in v for v in adata.obs[key]):
            raise ValueError(f"{key} must contain nonempty string labels without NUL characters.")
    sample_columns = [condition_key] + covariates + ([paired_by] if paired_by else [])
    for key in sample_columns:
        if (adata.obs.groupby(sample_key, observed=True)[key].nunique(dropna=False) > 1).any():
            raise ValueError(f"{key} is not constant within each biological sample.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Counts layer {layer!r} is absent.")
    X = adata.X if layer is None else adata.layers[layer]
    values = X.data if sparse.issparse(X) else np.asarray(X)
    if (not np.isfinite(values).all() or (values < 0).any() or
            not np.equal(values, np.floor(values)).all() or (values > 2**53).any()):
        raise ValueError("DS requires finite nonnegative integer raw counts (not log-normalized data).")
    # A conservative bound guarantees every profile/gene sum is exactly
    # representable before float64 matrix multiplication can round it.
    if values.size and int(np.max(values)) * int(adata.n_obs) > 2**53:
        raise ValueError("Raw counts exceed the conservative exact float64 aggregation bound.")
    selected = adata.obs[condition_key].isin([control, treatment]).to_numpy()
    if set(adata.obs.loc[selected, condition_key]) != {control, treatment}:
        raise ValueError("Both specified conditions must be present.")
    work = adata[selected].copy()
    # Do not allow unrelated input QC columns to overwrite the aggregator
    # receipt (the existing aggregator forwards constant obs metadata).
    work.obs = work.obs.loc[:, required].copy()
    # Existing aggregator carries constant columns as strings. Restore actual
    # sample-level dtypes for covariate modelling immediately afterwards.
    from ..single._pseudobulk import pseudobulk
    pb = pseudobulk(work, sample_col=sample_key, groups_col=group_key,
                    layer=layer, mode="sum", dtype=np.float64)
    metadata = work.obs.drop_duplicates(sample_key).set_index(sample_key)[sample_columns]
    for key in sample_columns:
        pb.obs[key] = metadata.loc[pb.obs[sample_key], key].array
    # Existing display labels join names with ' | ', which can collide.
    # Stable machine IDs avoid conflating different sample/type pairs.
    pb.obs_names = [f"pb_{i:06d}" for i in range(pb.n_obs)]
    if not np.isfinite(pb.X).all() or (pb.X > 2**53).any():
        raise ValueError("Aggregated counts exceed the exact float64 integer range.")
    rows, receipts, designs = [], [], {}
    from ..external import pyedger
    for group in sorted(pb.obs[group_key].unique()):
        choose = (pb.obs[group_key] == group).to_numpy()
        subset = pb[choose].copy()
        keep = (subset.obs.psbulk_n_cells >= min_cells).to_numpy() & (subset.X.sum(axis=1) > 0)
        excluded = subset.obs_names[~keep].tolist()
        subset = subset[keep].copy()
        unpaired = []
        if paired_by and subset.n_obs:
            pair_counts = subset.obs.groupby([paired_by, condition_key], observed=True).size()
            if (pair_counts > 1).any():
                raise ValueError(f"{group}: multiple samples per subject/condition; combine technical replicates first.")
            complete = subset.obs.groupby(paired_by, observed=True)[condition_key].nunique()
            complete = complete.index[complete == 2]
            pair_keep = subset.obs[paired_by].isin(complete).to_numpy()
            unpaired = subset.obs_names[~pair_keep].tolist()
            subset = subset[pair_keep].copy()
        ns = subset.obs[condition_key].value_counts()
        receipt = dict(group=group, n_control=int(ns.get(control, 0)), n_treatment=int(ns.get(treatment, 0)),
                       excluded_profiles=excluded, unpaired_profiles=unpaired, n_genes_tested=0)
        if min(receipt["n_control"], receipt["n_treatment"]) < min_samples:
            receipt.update(status="skipped", reason="insufficient biological samples")
            receipts.append(receipt); continue
        design = _design(subset.obs, condition_key, treatment, covariates, paired_by)
        designs[group] = design
        counts = pd.DataFrame(subset.X.T, index=subset.var_names, columns=subset.obs_names)
        gene_keep = pyedger.filterByExpr(counts, design=design, min_count=min_count, min_total_count=min_total_count)
        counts = counts.loc[gene_keep]
        if len(counts) < 2 or (counts.sum(axis=0) <= 0).any():
            receipt.update(status="skipped", reason="insufficient expressed genes or zero filtered library")
            receipts.append(receipt); continue
        table = _fit(counts, design, method).copy()
        p = table["pvalue"].to_numpy()
        finite = np.isfinite(p)
        if ((p[finite] < 0) | (p[finite] > 1)).any():
            raise RuntimeError("Statistical backend returned invalid p-values.")
        table["qvalue"] = np.nan
        table.loc[finite, "qvalue"] = multipletests(p[finite], method="fdr_bh")[1] if finite.any() else []
        table["group"] = group
        table["contrast"] = f"{treatment} - {control}"
        table["n_control"] = receipt["n_control"]
        table["n_treatment"] = receipt["n_treatment"]
        table["significant"] = table["qvalue"] < alpha
        rows.append(table.reset_index())
        receipt.update(status="tested", reason="", n_genes_tested=len(counts))
        receipts.append(receipt)
    results = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["gene", "group", "log2FC", "pvalue", "qvalue", "contrast", "n_control", "n_treatment", "significant"])
    results["qvalue_global"] = np.nan
    if len(results):
        finite = np.isfinite(results.pvalue.to_numpy())
        if finite.any():
            results.loc[finite, "qvalue_global"] = multipletests(results.loc[finite, "pvalue"], method="fdr_bh")[1]
    params = dict(sample_key=sample_key, group_key=group_key, condition_key=condition_key,
                  control=control, treatment=treatment, layer=layer, covariates=covariates,
                  paired_by=paired_by, method=method, min_cells=min_cells, min_samples=min_samples,
                  min_count=min_count, min_total_count=min_total_count, alpha=alpha,
                  backend="vendored Python pylimma/pyedger", excluded_other_condition_cells=int((~selected).sum()))
    return PseudobulkDSResult(pb, results, pd.DataFrame(receipts), designs, params)
