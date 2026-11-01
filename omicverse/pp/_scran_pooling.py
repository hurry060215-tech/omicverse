"""Pooling size factors through the separately installed Python backend."""
from importlib.metadata import PackageNotFoundError, version

from .._registry import register_function


@register_function(
    aliases=['scran_pooling'], category='pp',
    description='Compute pooling size factors with explicit AnnData input',
    examples=["ov.pp.scran_pooling(adata, layer='counts')"],
)
def scran_pooling(adata, *, layer=None, key_added='size_factors', copy=False, **kwargs):
    """Return AnnData with native pooled size factors in ``obs[key_added]``.

    Thin forwarding to ``scran_pooling.compute_pooled_factors`` from the local
    ``scran-pooling-py==0.1.0.dev0`` review package. Install its three fixed local
    dependency wheels and follow its qualified numerical-profile lock; the
    required pylimma primitive is not in the ordinary published pylimma release.
    The qualified Linux profile passes the frozen 116-case R gate. The same
    backend on ordinary PyPI NumPy/SciPy retains three strict NLS failures.

    AnnData is cells x genes. ``layer=None`` uses X even if counts exists;
    select ``layer='counts'`` explicitly when that is the intended matrix.
    Raw counts are the intended input. No counts inference, normalization,
    clustering, HVG selection or reordering is performed. Expression values
    remain unchanged. The requested obs column is written or overwritten;
    ``copy=False`` returns the original object and ``copy=True`` returns a copy.

    All ``kwargs`` go unchanged to the backend: sizes, clusters, ref_clust,
    max_cluster_size, positive, scaling, min_mean and subset_row. Supply cluster
    labels in current cell order (not the name of an obs column); numeric gene
    and reference-cluster indices are zero-based. Named subset_row requires
    unique AnnData gene names. Native errors, warnings and unsuccessful solver
    exits propagate. This entrypoint adds no alternate solver or R runtime.
    Other scran/scuttle methods, parallel/backed data and cross-platform strict
    parity are outside this scoped integration.
    """
    try:
        installed = version('scran-pooling-py')
    except PackageNotFoundError as exc:
        raise ImportError(
            'Install the local scran-pooling-py==0.1.0.dev0 review wheel and its '
            'fixed dependencies following INSTALL_QUALIFIED.md; it is not published on PyPI.'
        ) from exc
    if installed != '0.1.0.dev0':
        raise ImportError(f'This integration supports scran-pooling-py==0.1.0.dev0; found {installed}.')
    from scran_pooling import compute_pooled_factors

    return compute_pooled_factors(
        adata, layer=layer, key_added=key_added, copy=copy, **kwargs)
