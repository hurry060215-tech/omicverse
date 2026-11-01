"""Donor-aware MuSiC weighted deconvolution, limited core translation.

Numerical sequence follows xuranw/MuSiC 1.0.0 (GPL-3.0-or-later), commit
f21fe67f5670d5e9fca0ad7550abaae3423eb59c, construct.R and analysis.R.
Default weighted core and two-level music_prop.cluster are implemented.
This is not MuSiC2 or cross-type covariance.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls


def _labels(values, name):
    if pd.isna(values).any():
        raise ValueError(f"{name} must not contain missing labels")
    result = np.asarray(values).astype(str)
    if len(pd.unique(result)) != len(pd.unique(values)):
        raise ValueError(f"{name} labels collide after string conversion")
    return result


def _music_fit(y, design, sizes, sigma, *, max_iter, nu, eps):
    # MuSiC's default music.iter rescales relative bulk abundance by 100.
    y = y * 100
    if design.shape[0] <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("MuSiC requires a full-rank design with more genes than cell types")
    q0, _ = nnls(design, y)
    if q0.sum() <= 0:
        raise ValueError("NNLS returned zero total abundance")
    residual = y - design @ q0
    def fit(q, r):
        weights = 1 / (nu + r ** 2 + sigma @ ((q * sizes) ** 2))
        root = np.sqrt(weights)
        weighted_design = design * root[:, None]
        q_new, _ = nnls(weighted_design, y * root)
        if q_new.sum() <= 0:
            raise ValueError("Weighted NNLS returned zero total abundance")
        return q_new, y * root - weighted_design @ q_new, weights, weighted_design
    q, residual, weights, weighted_design = fit(q0, residual)
    p = q / q.sum()
    converged = False
    for iteration in range(1, max_iter + 1):
        q_new, residual, weights, weighted_design = fit(q, residual)
        p_new = q_new / q_new.sum()
        converged = np.abs(p_new - p).sum() < eps
        p, q = p_new, q_new
        if converged:
            break
    unweighted_residual = y - design @ q
    r_squared = 1 - np.var(unweighted_residual, ddof=1) / np.var(y, ddof=1)
    gram = weighted_design.T @ weighted_design
    condition = np.linalg.cond(gram, p=1)
    if not np.isfinite(condition) or condition >= 1 / np.finfo(float).eps:
        raise ValueError("MuSiC weighted design is numerically singular; variance cannot be estimated")
    variance = np.diag(np.linalg.inv(gram)) * np.mean(residual ** 2) / q.sum() ** 2
    if not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError("MuSiC variance calculation is nonfinite or negative; design is unstable")
    return {"weighted": p, "nnls": q0 / q0.sum(), "weights": weights,
            "variance": variance, "r_squared": r_squared,
            "converged": converged, "iterations": iteration}


def _music_setup(bulk, reference, *, cell_type_key, sample_key,
                        layer=None, cell_types=None, markers=None, cell_sizes=None,
                        max_iter=1000, nu=1e-4, eps=0.01):
    """Estimate bulk cell proportions using donor-aware MuSiC's default core.

    Parameters
    ----------
    bulk : pandas.DataFrame
        Nonnegative, untransformed genes-by-samples expression. Unique IDs required.
    reference : AnnData
        Cells-by-genes untransformed counts. Donors must be biological samples.
    cell_type_key, sample_key : str
        Reference ``obs`` columns for type and donor. At least two donors must
        support each retained type; invalid types are excluded and reported.
    layer : str, optional
        Explicit count layer; ``None`` uses X without normalization.
    cell_types : sequence, optional
        Requested ordered cell types, all of which must be present.
    markers : sequence, optional
        Genes to use. Cell-size totals always use all retained reference genes.
    cell_sizes : pandas.Series, optional
        Positive type-indexed cell sizes overriding mean library size.
    max_iter, nu, eps : int, float, float
        Native iteration limit, weight stabilizer and L1 proportion convergence.

    Returns
    -------
    dict
        Labeled weighted/NNLS proportions, gene weights, variances, R-squared,
        convergence and audited basis. Inputs are not mutated. This method is
        explicitly limited to independent cross-donor variance, not recursive
        cluster deconvolution, MuSiC2 or cross-cell-type covariance.
    """
    if not isinstance(bulk, pd.DataFrame) or min(bulk.shape) < 1:
        raise ValueError("bulk must be a nonempty genes-by-samples DataFrame")
    if not bulk.index.is_unique or not bulk.columns.is_unique or bulk.index.hasnans or bulk.columns.hasnans:
        raise ValueError("bulk gene and sample IDs must be unique and nonmissing")
    if not reference.var_names.is_unique or reference.var_names.hasnans or not reference.obs_names.is_unique or reference.obs_names.hasnans:
        raise ValueError("reference cell and gene IDs must be unique and nonmissing")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite([nu, eps]).all() or nu <= 0 or eps <= 0:
        raise ValueError("nu and eps must be finite positive numbers")
    for key in (cell_type_key, sample_key):
        if key not in reference.obs:
            raise KeyError(f"Reference obs column not found: {key}")
    types = _labels(reference.obs[cell_type_key], cell_type_key)
    donors = _labels(reference.obs[sample_key], sample_key)
    if isinstance(cell_types, (str, bytes)) or isinstance(markers, (str, bytes)):
        raise ValueError("cell_types and markers must be sequences, not strings")
    requested = list(pd.unique(types)) if cell_types is None else list(map(str, cell_types))
    if len(requested) < 2 or len(set(requested)) != len(requested) or not set(requested).issubset(types):
        raise ValueError("cell_types must contain at least two distinct observed cell types")
    selected = np.isin(types, requested)
    types, donors = types[selected], donors[selected]
    donor_order = list(pd.unique(donors))
    if len(donor_order) < 2:
        raise ValueError("MuSiC requires at least two biological reference donors")
    if layer is not None and layer not in reference.layers:
        raise KeyError(f"Reference count layer not found: {layer}")
    counts = reference.X if layer is None else reference.layers[layer]
    if not isinstance(counts, np.ndarray) and not sparse.issparse(counts):
        raise TypeError("reference counts must be an in-memory NumPy or SciPy matrix")
    values = counts.data if sparse.issparse(counts) else counts
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("reference counts must be finite and nonnegative")
    counts = counts[selected]
    keep = np.asarray(counts.sum(axis=0)).ravel() > 0
    counts = counts[:, keep]
    genes = np.asarray(reference.var_names)[keep]
    bulk_values = bulk.to_numpy(dtype=float)
    if not np.isfinite(bulk_values).all() or np.any(bulk_values < 0) or np.any(bulk_values.sum(axis=0) <= 0):
        raise ValueError("bulk must be finite, nonnegative and nonzero per sample")
    bulk_keep = bulk_values.mean(axis=1) != 0
    bulk_genes = bulk.index[bulk_keep]
    wanted = list(bulk_genes) if markers is None else [g for g in dict.fromkeys(markers) if g in bulk_genes]
    gene_set = set(genes)
    common = [g for g in wanted if g in gene_set]
    denominator = min(len(bulk_genes), reference.n_vars) if markers is None else len(markers)
    if len(common) < max(2, .2 * denominator):
        raise ValueError("Too few common genes for MuSiC")
    means = np.full((len(genes), len(requested)), np.nan)
    sigma = np.full_like(means, np.nan)
    library_sizes = np.full((len(donor_order), len(requested)), np.nan)
    # Keep one donor-by-gene block, not a donor-by-gene-by-type cube.
    if len(donor_order) * len(genes) * 8 > 512 * 1024 ** 2:
        raise MemoryError("MuSiC reference donor-by-gene block exceeds 512 MiB; split inputs explicitly")
    for k, ct in enumerate(requested):
        theta = np.full((len(donor_order), len(genes)), np.nan)
        for j, donor in enumerate(donor_order):
            rows = (types == ct) & (donors == donor)
            if not rows.any():
                continue
            total = np.asarray(counts[rows].sum(axis=0)).ravel()
            if total.sum() > 0:
                theta[j] = total / total.sum()
                library_sizes[j, k] = total.sum() / rows.sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means[:, k] = np.nanmean(theta, axis=0)
            sigma[:, k] = np.nanvar(theta, axis=0, ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sizes = np.nanmean(library_sizes, axis=0)
    if cell_sizes is not None:
        if not isinstance(cell_sizes, pd.Series) or not cell_sizes.index.is_unique or not set(requested).issubset(cell_sizes.index):
            raise ValueError("cell_sizes must be a unique type-indexed pandas Series")
        sizes = cell_sizes.loc[requested].to_numpy(dtype=float)
        if not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("cell_sizes must be finite and positive")
    positions = pd.Index(genes).get_indexer(common)
    design = means[positions] * sizes
    sigma = sigma[positions]
    valid = np.isfinite(design).all(axis=0) & np.isfinite(sigma).all(axis=0) & np.isfinite(sizes)
    if valid.sum() < 2:
        raise ValueError("Not enough valid cell types with cross-donor variance")
    omitted = [ct for ct, ok in zip(requested, valid) if not ok]
    retained = [ct for ct, ok in zip(requested, valid) if ok]
    design, sigma, sizes = design[:, valid], sigma[:, valid], sizes[valid]
    y = bulk.loc[common].to_numpy(dtype=float)
    if np.any(y.sum(axis=0) <= 0):
        raise ValueError("Each bulk sample needs positive expression in common genes")
    y = y / y.sum(axis=0)
    return y, design, sigma, sizes, common, retained, omitted, donor_order


def music_deconvolution(bulk, reference, *, cell_type_key, sample_key,
                        layer=None, cell_types=None, markers=None, cell_sizes=None,
                        max_iter=1000, nu=1e-4, eps=0.01):
    """Donor-aware default MuSiC weighted core; see module documentation.

    bulk is a genes-by-samples DataFrame; reference is cells-by-genes AnnData.
    X (or explicit layer) must be untransformed nonnegative counts. Donors are
    identified by sample_key. Returns labeled proportions, uncertainty, weights,
    convergence diagnostics and basis. Does not mutate inputs.
    """
    y, design, sigma, sizes, common, retained, omitted, donor_order = _music_setup(
        bulk, reference, cell_type_key=cell_type_key, sample_key=sample_key,
        layer=layer, cell_types=cell_types, markers=markers, cell_sizes=cell_sizes,
        max_iter=max_iter, nu=nu, eps=eps)
    fits, weights = [], np.full(y.shape, np.nan)
    for i in range(y.shape[1]):
        nonzero = y[:, i] != 0
        fit = _music_fit(y[nonzero, i], design[nonzero], sizes, sigma[nonzero],
                         max_iter=max_iter, nu=nu, eps=eps)
        fits.append(fit)
        weights[nonzero, i] = fit["weights"]
    return {
        "proportions": pd.DataFrame([f["weighted"] for f in fits], index=bulk.columns, columns=retained),
        "nnls_proportions": pd.DataFrame([f["nnls"] for f in fits], index=bulk.columns, columns=retained),
        "gene_weights": pd.DataFrame(weights, index=common, columns=bulk.columns),
        "variance": pd.DataFrame([f["variance"] for f in fits], index=bulk.columns, columns=retained),
        "diagnostics": pd.DataFrame([{k:f[k] for k in ("r_squared", "converged", "iterations")} for f in fits], index=bulk.columns),
        "basis": {"design":pd.DataFrame(design,index=common,columns=retained),
                  "sigma":pd.DataFrame(sigma,index=common,columns=retained),
                  "cell_sizes":pd.Series(sizes,index=retained)},
        "provenance": {"method":"MuSiC", "scope":"default weighted core; no recursion/MuSiC2/ct.cov", "reference_commit":"f21fe67f5670d5e9fca0ad7550abaae3423eb59c",
                       "layer":"X" if layer is None else layer, "sample_key":sample_key,
                       "cell_type_key":cell_type_key, "reference_donors":donor_order,
                       "excluded_cell_types":omitted,"max_iter":int(max_iter),"nu":float(nu),"eps":float(eps)}}


def music_deconvolution_recursive(bulk, reference, *, group_key, cell_type_key,
                                  sample_key, group_cell_types, group_markers,
                                  layer=None, max_iter=1000, nu=1e-4, eps=0.01):
    """MuSiC ``music_prop.cluster`` two-level group/subtype deconvolution.

    group_cell_types and group_markers are mappings with exactly the same
    ordered group keys. Every selected fine cell type must belong to its stated
    group. Marker genes are excluded from the top-level estimate and used in
    the native residual-adjusted subtype estimate. This is a fixed two-level
    procedure, not arbitrary tree recursion or MuSiC2. Native subgroup calls use
    max_iter=1000, nu=1e-4, eps=.01 regardless of top-level settings.
    """
    from collections.abc import Mapping
    if not isinstance(group_cell_types, Mapping) or not isinstance(group_markers, Mapping):
        raise ValueError("group_cell_types and group_markers must be mappings")
    groups = list(group_cell_types)
    if len(groups) < 2 or set(groups) != set(group_markers):
        raise ValueError("At least two groups with matching marker keys are required")
    if any(not isinstance(g, str) for g in groups):
        raise ValueError("Group keys must be strings matching reference group labels")
    if any(isinstance(group_cell_types[g], (str, bytes)) or isinstance(group_markers[g], (str, bytes)) for g in groups):
        raise ValueError("Group cell types and markers must be sequences, not strings")
    fine = [str(ct) for g in groups for ct in group_cell_types[g]]
    if any(len(group_cell_types[g]) == 0 for g in groups) or len(set(fine)) != len(fine):
        raise ValueError("Each group needs distinct, non-overlapping cell types")
    if group_key not in reference.obs:
        raise KeyError(group_key)
    group_labels = _labels(reference.obs[group_key], group_key)
    type_labels = _labels(reference.obs[cell_type_key], cell_type_key)
    for g in groups:
        for ct in group_cell_types[g]:
            rows = type_labels == str(ct)
            if not rows.any() or np.any(group_labels[rows] != g):
                raise ValueError("group_cell_types does not match reference group membership")
    y, design, sigma, sizes, common, kept, omitted, donors = _music_setup(
        bulk, reference, cell_type_key=cell_type_key, sample_key=sample_key,
        layer=layer, cell_types=fine, max_iter=max_iter,nu=nu,eps=eps)
    if omitted:
        raise ValueError("Recursive MuSiC requires cross-donor support for every selected cell type")
    _, group_design, group_sigma, group_sizes, group_genes, group_kept, group_omitted, _ = _music_setup(
        bulk, reference, cell_type_key=group_key, sample_key=sample_key,
        layer=layer, cell_types=groups,max_iter=max_iter,nu=nu,eps=eps)
    if group_omitted or group_kept != groups:
        raise ValueError("Recursive MuSiC requires cross-donor support for every group")
    align = pd.Index(group_genes).get_indexer(common)
    if np.any(align < 0):
        raise ValueError("Group and fine-cell bases have inconsistent gene coverage")
    group_design, group_sigma = group_design[align], group_sigma[align]
    common_set = set(common)
    marker_map = {g: [x for x in dict.fromkeys(group_markers[g]) if x in common_set] for g in groups}
    all_markers = {x for values in marker_map.values() for x in values}
    top = np.asarray([g not in all_markers for g in common])
    results=[]; group_results=[]
    for i in range(y.shape[1]):
        valid=(y[:,i] != 0) & top
        top_fit=_music_fit(y[valid,i],group_design[valid],group_sizes,group_sigma[valid],
                           max_iter=max_iter,nu=nu,eps=eps)
        proportions=top_fit['weighted']; group_results.append(proportions)
        out=[]
        for j,g in enumerate(groups):
            ct=list(map(str,group_cell_types[g]))
            if len(ct)==1:
                out.append(proportions[j]);continue
            if proportions[j]==0:
                out.extend([0.]*len(ct));continue
            ids=np.asarray([k for k,gene in enumerate(common) if gene in marker_map[g] and y[k,i] != 0],dtype=int)
            sub_y=(group_design[ids,j]*proportions[j]+
                   (y[ids,i]-group_design[ids]@proportions)*proportions[j])
            pos=sub_y>0;ids=ids[pos];sub_y=sub_y[pos]
            columns=pd.Index(kept).get_indexer(ct)
            fit=_music_fit(sub_y,design[np.ix_(ids,columns)],sizes[columns],sigma[np.ix_(ids,columns)],
                           max_iter=1000,nu=1e-4,eps=.01)
            out.extend(proportions[j]*fit['weighted'])
        results.append(out)
    return {'proportions':pd.DataFrame(results,index=bulk.columns,columns=fine),
            'group_proportions':pd.DataFrame(group_results,index=bulk.columns,columns=groups),
            'provenance':{'method':'MuSiC music_prop.cluster','scope':'two-level default variance recursion',
                          'reference_commit':'f21fe67f5670d5e9fca0ad7550abaae3423eb59c',
                          'group_cell_types':dict(group_cell_types),'group_markers':marker_map,
                          'reference_donors':donors,'max_iter':max_iter,'nu':nu,'eps':eps}}
