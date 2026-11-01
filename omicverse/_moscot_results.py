"""Identity-preserving, serializable results from native moscot problems."""
from __future__ import annotations

import importlib.metadata
import numpy as np
from scipy import sparse
from anndata import AnnData

from ._backend_validation import validate_adata, validate_matrix, output_key
from ._registry import register_function


def _limit(shape, max_entries):
    if isinstance(max_entries, bool) or not isinstance(max_entries, (int, np.integer)) or max_entries < 1:
        raise ValueError("max_entries must be a positive integer.")
    if int(np.prod(shape)) > max_entries:
        raise ValueError(f"Dense result {shape} exceeds max_entries={max_entries}; subset or explicitly raise the limit.")


def _version():
    return importlib.metadata.version("moscot")


def _subproblem(problem, key):
    if key not in problem.solutions or key not in problem.problems:
        raise KeyError(f"No solved subproblem for {key!r}.")
    sub = problem.problems[key]
    src, tgt = sub.adata_src, sub.adata_tgt
    if not src.obs_names.is_unique or not tgt.obs_names.is_unique:
        raise ValueError("Subproblem cell identifiers must be unique.")
    return src, tgt, problem.solutions[key]


@register_function(aliases=["moscot coupling export"], category="single",
                   description="Export one native moscot coupling with explicit row and column cell identifiers")
def moscot_coupling(problem, key, *, require_convergence=True, max_entries=10_000_000):
    """Return a saveable AnnData with source rows and target columns.

    MappingProblem uses spatial rows and single-cell columns; temporal and
    alignment problems use the source and target indicated by ``key``. X is
    transport mass, NOT a row-stochastic probability, p-value or lineage truth.
    This explicitly materializes a dense matrix and rejects oversized outputs.
    """
    src, tgt, solution = _subproblem(problem, key)
    _limit((src.n_obs, tgt.n_obs), max_entries)
    if require_convergence and not bool(solution.converged):
        raise RuntimeError("The requested moscot solution has not converged.")
    matrix = np.asarray(solution.transport_matrix)
    if matrix.shape != (src.n_obs, tgt.n_obs):
        raise RuntimeError("Coupling shape does not match named source/target cells.")
    validate_matrix(matrix, "Coupling")
    if np.any(matrix < 0):
        raise RuntimeError("Transport mass must be nonnegative.")
    result = AnnData(matrix.copy(), obs=src.obs.copy(), var=tgt.obs.copy())
    result.obs["transport_mass"] = matrix.sum(axis=1)
    result.var["transport_mass"] = matrix.sum(axis=0)
    result.uns["moscot"] = {"backend_version": _version(), "source": str(key[0]),
                            "target": str(key[1]), "converged": bool(solution.converged),
                            "orientation": "source_rows_target_columns", "value": "transport_mass"}
    return result


@register_function(aliases=["moscot aligned coordinates"], category="space",
                   description="Create aligned coordinates in a new AnnData without replacing physical coordinates")
def moscot_align_coordinates(problem, *, reference, mode="warp", key_added="spatial_aligned", require_convergence=True):
    """Run native AlignmentProblem.align and return a new saveable AnnData.

    Physical coordinates are retained. ``reference`` is an actual batch label,
    not a row number. Affine metadata is not copied into uns because native keys
    may be non-string; mode, reference and backend version are recorded instead.
    """
    validate_adata(problem.adata)
    if not isinstance(reference, str):
        raise ValueError("reference must be a string batch label; moscot 0.5.2 alignment does not reliably support numeric labels.")
    policy_reference = getattr(getattr(problem, "_policy", None), "reference", None)
    if policy_reference is not None and policy_reference != reference:
        raise ValueError("reference conflicts with the fixed reference of the native star policy.")
    if mode not in {"warp", "affine"}:
        raise ValueError("mode must be 'warp' or 'affine'.")
    if key_added == problem.spatial_key:
        raise ValueError("Aligned coordinates must use a different key from physical coordinates.")
    if not isinstance(key_added, str) or "/" in key_added:
        raise ValueError("key_added must be a storage-safe string without slashes.")
    output_key(problem.adata, key_added)
    if not problem.solutions:
        raise RuntimeError("No solved alignment subproblems.")
    convergence = {"source": [str(k[0]) for k in problem.solutions],
                   "target": [str(k[1]) for k in problem.solutions],
                   "converged": [bool(v.converged) for v in problem.solutions.values()]}
    if require_convergence and not all(convergence["converged"]):
        raise RuntimeError("An alignment solution has not converged.")
    original = np.asarray(problem.adata.obsm[problem.spatial_key]).copy()
    aligned, _ = problem.align(reference=reference, mode=mode, key_added=None)
    if not np.array_equal(original, problem.adata.obsm[problem.spatial_key]):
        raise RuntimeError("Native alignment modified physical coordinates.")
    aligned = np.asarray(aligned)
    if aligned.shape != original.shape:
        raise RuntimeError("Aligned coordinate shape does not match input observations.")
    validate_matrix(aligned, "Aligned coordinates")
    result = problem.adata.copy()
    result.obsm[key_added] = aligned.copy()
    result.uns[f"{key_added}_provenance"] = {"mode": mode, "reference": str(reference),
                                           "physical_key": problem.spatial_key, "backend_version": _version(), "convergence": convergence}
    return result


@register_function(aliases=["moscot safe imputation"], category="space",
                   description="Impute mapped expression while preserving interleaved spatial batch identities")
def moscot_impute(problem, *, var_names=None, batch_size=128, max_entries=10_000_000,
                   require_convergence=True):
    """Return spatial-by-gene predictions, explicitly aligning every subproblem.

    Avoids native moscot 0.5.2's batch-stacking identity ambiguity for interleaved
    spatial observations. Uses the native solution.pull(scale_by_marginals=True)
    for each feature block, then writes by source cell ID. No NaN-to-zero repair
    is performed. Values retain the native target-marginal scaling; they are transported expression,
    not measured counts or a source-row conditional mean. A constant feature need
    not remain constant when source/target population sizes differ.
    """
    ref, sp = problem.adata_sc, problem.adata_sp
    validate_adata(ref)
    validate_adata(sp)
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    genes = ref.var_names if var_names is None else np.asarray(var_names)
    import pandas as pd
    genes = pd.Index(genes)
    if not len(genes) or not genes.is_unique or not genes.isin(ref.var_names).all():
        raise ValueError("var_names must be unique, nonempty reference gene identifiers.")
    _limit((sp.n_obs, len(genes)), max_entries)
    predictions = np.empty((sp.n_obs, len(genes)), dtype=float)
    covered = np.zeros(sp.n_obs, dtype=bool)
    for key in problem.solutions:
        src, tgt, solution = _subproblem(problem, key)
        if require_convergence and not bool(solution.converged):
            raise RuntimeError(f"moscot solution {key!r} has not converged.")
        rows = sp.obs_names.get_indexer(src.obs_names)
        if np.any(rows < 0) or covered[rows].any():
            raise RuntimeError("Spatial solution cells are absent or occur in overlapping subproblems.")
        ref_rows = ref.obs_names.get_indexer(tgt.obs_names)
        if np.any(ref_rows < 0):
            raise RuntimeError("Mapping target cells are absent from the reference.")
        for start in range(0, len(genes), batch_size):
            stop = min(start + batch_size, len(genes))
            _limit((len(ref_rows), stop-start), max_entries)
            values = ref[ref_rows, genes[start:stop]].X
            validate_matrix(values, "Reference expression")
            values = values.toarray() if sparse.issparse(values) else np.asarray(values)
            block = np.asarray(solution.pull(values, scale_by_marginals=True))
            if block.shape != (len(rows), stop-start):
                raise RuntimeError("Native imputed output shape does not match mapped identities.")
            validate_matrix(block, "Imputed expression")
            predictions[rows, start:stop] = block
        covered[rows] = True
    if not covered.all():
        raise RuntimeError("Mapping solutions do not cover every spatial observation.")
    result = AnnData(predictions, obs=sp.obs.copy(), var=ref[:, genes].var.copy(),
                     obsm={k: v.copy() for k,v in sp.obsm.items()})
    result.uns["moscot_impute"] = {"backend_version": _version(), "identity_alignment": "subproblem_cell_ids",
                                    "value": "native_target_marginal_scaled_transport", "normalization": "solution.pull(scale_by_marginals=True)", "batch_size": int(batch_size)}
    return result
