"""AnnData adaptation for the optional, R-free SPARK-X backend."""

from collections.abc import Mapping
from copy import deepcopy
from importlib import import_module

import numpy as np
import pandas as pd
from scipy import sparse


def _covariate_matrix(adata, covariates):
    """Resolve explicit covariates without alignment, scaling or an intercept."""
    if covariates is None:
        return None, {'source': 'none'}
    if isinstance(covariates, str):
        covariates = [covariates]
    if isinstance(covariates, (list, tuple)) and all(
        isinstance(key, str) for key in covariates
    ):
        if not covariates:
            raise ValueError('covariates must contain at least one column.')
        missing = [key for key in covariates if key not in adata.obs]
        if missing:
            raise KeyError(f'Covariate columns not found in adata.obs: {missing}.')
        metadata = {'source': 'obs', 'columns': list(covariates)}
        values = adata.obs.loc[:, list(covariates)]
    elif isinstance(covariates, pd.DataFrame):
        if not covariates.index.equals(adata.obs_names):
            raise ValueError('covariates DataFrame index must exactly match adata.obs_names in order.')
        metadata = {'source': 'dataframe', 'columns': list(map(str, covariates.columns))}
        values = covariates
    else:
        metadata = {'source': 'array', 'alignment': 'positional_obs_order'}
        values = covariates
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('covariates must be numeric; encode categorical columns explicitly.') from exc
    if values.ndim != 2 or values.shape[0] != adata.n_obs or values.shape[1] == 0:
        raise ValueError('covariates must be a 2-D matrix with one row per observation and at least one column.')
    if not np.isfinite(values).all():
        raise ValueError('covariates must contain only finite values.')
    metadata['n_columns'] = values.shape[1]
    return values, metadata


def _retained_indices(result, key, size):
    indices = np.asarray(result[key])
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= size)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError(f'SPARK-X returned invalid {key}; cannot align results to AnnData.')
    return indices.astype(np.int64, copy=False)


def _sparkx_svg(adata, *, n_svgs, selection, library_key=None, layer=None,
                spatial_key='spatial', covariates=None, option='mixture',
                n_jobs=1, verbose=True, qval_threshold=0.05):
    """Run SPARK-X once on one slice and write results by original gene position."""
    if library_key is not None:
        if library_key not in adata.obs or adata.obs[library_key].isna().any():
            raise ValueError('library_key must exist in adata.obs without missing labels.')
        if adata.obs[library_key].nunique() > 1:
            raise ValueError('SPARK-X supports one spatial library per call; subset each library before inference.')
    spatial_metadata = adata.uns.get('spatial', {})
    if isinstance(spatial_metadata, Mapping) and len(spatial_metadata) > 1:
        raise ValueError("SPARK-X supports one spatial library per call; subset each library and its uns['spatial'] metadata before inference.")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError('SPARK-X requires a nonempty expression matrix.')
    if layer is not None and layer not in adata.layers:
        raise KeyError(f'layer {layer!r} was not found in adata.layers.')
    matrix = adata.X if layer is None else adata.layers[layer]
    if not (isinstance(matrix, np.ndarray) or sparse.issparse(matrix)):
        raise TypeError('SPARK-X requires an in-memory NumPy or SciPy sparse expression matrix.')
    if spatial_key not in adata.obsm:
        raise KeyError(f'spatial_key {spatial_key!r} was not found in adata.obsm.')
    try:
        coordinates = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('SPARK-X spatial coordinates must be numeric.') from exc
    if (coordinates.ndim != 2 or coordinates.shape[0] != adata.n_obs
            or coordinates.shape[1] == 0 or not np.isfinite(coordinates).all()):
        raise ValueError('SPARK-X spatial coordinates must be a finite 2-D matrix with one row per observation.')
    design, covariate_metadata = _covariate_matrix(adata, covariates)

    try:
        backend = import_module('sparkx')
    except ImportError as exc:
        raise ImportError(
            "SPARK-X requires the optional 'sparkx-py' backend. Install the "
            "local py-SPARK-X package with `pip install /path/to/py-SPARK-X`."
        ) from exc
    if not callable(getattr(backend, 'sparkx', None)):
        raise ImportError("The installed 'sparkx' module does not expose sparkx(); install the compatible 'sparkx-py' backend.")

    # Transposition preserves sparse storage. Filtering and inference belong to
    # the backend; the adapter never normalizes or densifies expression.
    result = backend.sparkx(matrix.T, coordinates, X_in=design,
                            numCores=n_jobs, option=option, verbose=verbose)
    genes = _retained_indices(result, 'gene_indices', adata.n_vars)
    spots = _retained_indices(result, 'spot_indices', adata.n_obs)
    stats = result['stats']
    pvalues = result['res_stest']
    combined = result['res_mtest']
    if (not all(isinstance(frame, pd.DataFrame) for frame in (stats, pvalues, combined))
            or len(combined) != len(genes) or len(stats) != len(genes)
            or stats.shape != pvalues.shape or not stats.columns.equals(pvalues.columns)
            or not stats.index.equals(pvalues.index) or not stats.index.equals(combined.index)
            or not {'combinedPval', 'adjustedPval'}.issubset(combined.columns)):
        raise ValueError('SPARK-X returned inconsistent result dimensions or columns.')
    full_p = np.full(adata.n_vars, np.nan)
    full_q = np.full(adata.n_vars, np.nan)
    full_p[genes] = combined['combinedPval'].to_numpy(dtype=float)
    full_q[genes] = combined['adjustedPval'].to_numpy(dtype=float)
    full_stats = np.full((adata.n_vars, stats.shape[1]), np.nan)
    full_kernel_p = np.full_like(full_stats, np.nan)
    full_stats[genes] = stats.to_numpy(dtype=float)
    full_kernel_p[genes] = pvalues.to_numpy(dtype=float)

    candidates = np.flatnonzero(np.isfinite(full_p) & np.isfinite(full_q))
    if selection == 'significant':
        candidates = candidates[full_q[candidates] < qval_threshold]
    # q, then p, then original position, including duplicate gene names.
    order = np.lexsort((candidates, full_p[candidates], full_q[candidates]))
    selected = candidates[order][:n_svgs]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[selected] = True
    kernels = list(map(str, stats.columns))
    run = {
        'method': 'sparkx', 'backend': 'sparkx-py',
        'version': str(getattr(backend, '__version__', 'unknown')),
        'spatial_evidence': True, 'correction_method': 'BY',
        'params': {'option': option, 'n_jobs': n_jobs, 'verbose': verbose,
                   'selection': selection, 'qval_threshold': float(qval_threshold),
                   'n_svgs': 'all' if n_svgs is None else int(n_svgs)},
        'input': {'expression': 'X' if layer is None else f'layers/{layer}',
                  'spatial_key': spatial_key, 'n_obs': adata.n_obs,
                  'n_vars': adata.n_vars, 'covariates': covariate_metadata},
        'filter': {'gene_indices': genes.copy(), 'spot_indices': spots.copy(),
                   'filtered_gene_indices': np.setdiff1d(np.arange(adata.n_vars), genes),
                   'filtered_spot_indices': np.setdiff1d(np.arange(adata.n_obs), spots)},
        'kernels': kernels, 'n_selected': int(mask.sum()),
        'backend_metadata': deepcopy(result.get('metadata', {})),
    }
    if library_key is not None:
        run['input']['library_key'] = library_key

    # Do not partially overwrite a prior run when input/backend validation fails.
    adata.var['sparkx_pval'] = full_p
    adata.var['sparkx_qval'] = full_q
    adata.varm['sparkx_stats'] = pd.DataFrame(full_stats, index=adata.var_names, columns=kernels)
    adata.varm['sparkx_pvals'] = pd.DataFrame(full_kernel_p, index=adata.var_names, columns=kernels)
    adata.var['space_variable_features'] = mask
    adata.var['highly_variable'] = mask
    adata.uns.setdefault('space_svg_runs', {})['sparkx'] = run
    return adata
