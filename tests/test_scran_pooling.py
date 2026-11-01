"""Thin-adapter contracts against the fixed separately installed backend."""
import importlib
from importlib.metadata import PackageNotFoundError, version
import warnings
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
import omicverse as ov
try:
    installed = version('scran-pooling-py')
except PackageNotFoundError:
    pytest.skip('optional local scran backend is not installed', allow_module_level=True)
if installed != '0.1.0.dev0':
    pytest.skip('comparisons require the fixed local backend', allow_module_level=True)
import scran_pooling as native


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def example(kind='dense'):
    rng = np.random.default_rng(989)
    counts = rng.poisson(rng.uniform(1, 8, (160, 1)) * rng.uniform(.5, 2, (1, 30))).astype(float)
    convert = {'dense':np.asarray, 'csr':sparse.csr_matrix, 'csc':sparse.csc_matrix}[kind]
    a = AnnData(convert(counts), obs=pd.DataFrame({'kept':np.arange(160)}, index=[f'c{i}' for i in range(160)]), var=pd.DataFrame({'kept':np.arange(30)}, index=[f'g{i}' for i in range(30)]))
    a.layers['counts'] = convert(counts * np.linspace(1, 3, 160)[:, None])
    a.obsm['existing'] = np.arange(320).reshape(160, 2)
    a.uns['existing'] = {'value':'keep'}
    return a


def call_with_warnings(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result = function(*args, **kwargs)
    return result, [(w.category.__name__, str(w.message)) for w in captured]


@pytest.mark.parametrize('kind', ['dense','csr','csc'])
@pytest.mark.parametrize('layer', [None,'counts'])
@pytest.mark.parametrize('copy', [False,True])
def test_native_result_and_mutation(kind, layer, copy):
    a = example(kind); before = a.copy(); b = a.copy()
    options = dict(layer=layer, key_added='pooled', copy=copy, sizes=[5,9], min_mean=0)
    expected, nw = call_with_warnings(native.compute_pooled_factors, b, **options)
    actual, ow = call_with_warnings(ov.pp.scran_pooling, a, **options)
    assert nw == ow and (actual is a) is (not copy)
    np.testing.assert_array_equal(actual.obs['pooled'], expected.obs['pooled'])
    for obj in [a, actual]:
        np.testing.assert_array_equal(dense(obj.X), dense(before.X))
        np.testing.assert_array_equal(dense(obj.layers['counts']), dense(before.layers['counts']))
        np.testing.assert_array_equal(obj.obsm['existing'], before.obsm['existing'])
        pd.testing.assert_frame_equal(obj.var, before.var)
        pd.testing.assert_series_equal(obj.obs['kept'], before.obs['kept'])
        assert obj.uns == before.uns
    assert ('pooled' in a.obs) is (not copy)


def test_no_counts_layer_guess():
    a=example(); b=a.copy(); c=a.copy()
    ov.pp.scran_pooling(a, sizes=[5,9], min_mean=0)
    native.compute_pooled_factors(b, layer=None, sizes=[5,9], min_mean=0)
    native.compute_pooled_factors(c, layer='counts', sizes=[5,9], min_mean=0)
    np.testing.assert_array_equal(a.obs.size_factors, b.obs.size_factors)
    assert not np.allclose(a.obs.size_factors, c.obs.size_factors)


def test_order_clusters_scaling_named_subset():
    a=example('csr')[np.random.default_rng(2).permutation(160), :].copy()
    a=a[:, np.arange(a.n_vars)[::-1]].copy(); b=a.copy()
    labels=pd.Categorical(np.tile(['b','a'],80), categories=['b','a'])
    options=dict(clusters=labels, ref_clust=0, scaling=np.linspace(2,3,160), subset_row=['g29','g19','g9'], sizes=[5,9], max_cluster_size=50, min_mean=0, positive=False)
    names=a.obs_names.copy(); genes=a.var_names.copy()
    native.compute_pooled_factors(b, **options); ov.pp.scran_pooling(a, **options)
    np.testing.assert_array_equal(a.obs.size_factors,b.obs.size_factors)
    assert a.obs_names.equals(names) and a.var_names.equals(genes)


@pytest.mark.parametrize('options', [dict(layer='absent'),dict(subset_row=['missing']),dict(sizes=[5,5]),dict(clusters=['a']),dict(unknown_parameter=True)])
def test_native_errors_propagate(options):
    a=example(); b=a.copy()
    with pytest.raises(Exception) as expected:native.compute_pooled_factors(b, **options)
    with pytest.raises(type(expected.value)) as actual:ov.pp.scran_pooling(a, **options)
    assert str(actual.value) == str(expected.value)
    assert 'size_factors' not in a.obs


def test_duplicate_genes_named_subset_rejected():
    a=example();a.var_names=['duplicate']*a.n_vars
    with pytest.raises(ValueError,match='unique AnnData var_names'):
        ov.pp.scran_pooling(a, subset_row=['duplicate'])


def test_forward_objects_without_copy(monkeypatch):
    data=object(); clusters=object(); scaling=object(); result=object(); seen={}
    def backend(adata, **kwargs):seen.update(adata=adata, **kwargs);return result
    monkeypatch.setattr(native,'compute_pooled_factors',backend)
    actual=ov.pp.scran_pooling(data, layer='counts', key_added='p', copy=True, clusters=clusters, scaling=scaling)
    assert actual is result and seen['adata'] is data
    assert seen['clusters'] is clusters and seen['scaling'] is scaling
    assert seen['layer']=='counts' and seen['key_added']=='p' and seen['copy'] is True


def test_solver_failure_not_swallowed(monkeypatch):
    def fail(*args, **kwargs):raise RuntimeError('LSQR did not converge: istop=7')
    monkeypatch.setattr(native,'compute_pooled_factors',fail)
    with pytest.raises(RuntimeError,match='istop=7'):ov.pp.scran_pooling(example())


def test_native_warning_is_preserved():
    a=example();a.X *= 75000 / np.median(a.X.sum(axis=1));b=a.copy()
    expected, nw=call_with_warnings(native.compute_pooled_factors,b,sizes=[5,9],positive=False)
    actual, ow=call_with_warnings(ov.pp.scran_pooling,a,sizes=[5,9],positive=False)
    assert nw and nw==ow and any('assuming UMI data' in text for _,text in nw)
    np.testing.assert_array_equal(actual.obs.size_factors,expected.obs.size_factors)


def test_requested_obs_column_is_overwritten():
    a=example();a.obs['chosen']=-1.;b=a.copy()
    expected=native.compute_pooled_factors(b,key_added='chosen',sizes=[5,9],min_mean=0)
    actual=ov.pp.scran_pooling(a,key_added='chosen',sizes=[5,9],min_mean=0)
    assert actual is a
    np.testing.assert_array_equal(actual.obs.chosen,expected.obs.chosen)
