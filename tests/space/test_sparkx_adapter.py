"""SPARK-X adapter regressions against the actual optional Python backend.

Install the local sparkx-py package to run numerical/integration tests. Only
dependency failures and dispatch are mocked; numerical outputs are never faked.
"""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from scipy import sparse
from statsmodels.stats.multitest import multipletests

from omicverse.space import svg
from omicverse.space import _sparkx as adapter


@pytest.fixture
def backend():
    return pytest.importorskip('sparkx', reason='Install the optional sparkx-py backend')


def spatial_data(kind='dense'):
    rng = np.random.default_rng(733)
    coordinates = rng.uniform(-2, 2, (48, 2))
    expression = rng.poisson(.7, (48, 8)).astype(float)
    expression[:, 0] = rng.poisson(np.exp(coordinates[:, 0] * 2) + 1)
    expression[:, 1] = expression[:, 0]  # Exact p/q ties, with different labels.
    expression[:, 3] = 0  # Filtered gene is intentionally not last.
    expression[:, 4] = rng.poisson(np.exp(coordinates[:, 1]) + 1)
    expression[13] = 0  # Filtered spot is intentionally not last.
    if kind != 'dense':
        expression = getattr(sparse, f'{kind}_matrix')(expression)
    data = AnnData(expression)
    data.obs_names = [f'spot-{i}' for i in range(data.n_obs)]
    data.var_names = [f'gene-{i}' for i in range(data.n_vars)]
    data.obs['technical'] = rng.normal(size=data.n_obs)
    data.obs['intercept'] = 1.
    data.obsm['spatial'] = coordinates
    return data


def assert_matrix_unchanged(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    if sparse.issparse(actual):
        for attr in ('data', 'indices', 'indptr'):
            np.testing.assert_array_equal(getattr(actual, attr), getattr(expected, attr))
    else:
        np.testing.assert_array_equal(actual, expected)


def expected_selection(p, q, n_svgs, threshold=None):
    idx = np.flatnonzero(np.isfinite(p) & np.isfinite(q))
    if threshold is not None:
        idx = idx[q[idx] < threshold]
    order = np.lexsort((idx, p[idx], q[idx]))
    result = np.zeros(len(p), dtype=bool)
    result[idx[order][:n_svgs]] = True
    return result


@pytest.mark.parametrize('option', ['single', 'mixture'])
@pytest.mark.parametrize('kind', ['dense', 'csr', 'csc'])
@pytest.mark.parametrize('with_covariates', [False, True])
def test_actual_backend_matches_all_outputs_and_preserves_inputs(backend, option, kind, with_covariates):
    data = spatial_data(kind)
    matrix_object = data.X
    expression_before = data.X.copy()
    obs_before = data.obs.copy(deep=True)
    var_before = data.var_names.copy()
    coordinates_before = data.obsm['spatial'].copy()
    covariates = ['intercept', 'technical'] if with_covariates else None
    design = data.obs[covariates].to_numpy() if with_covariates else None
    direct = backend.sparkx(data.X.T, data.obsm['spatial'], X_in=design,
                            option=option, numCores=1, verbose=False)

    result = svg(data, mode='sparkx', option=option, covariates=covariates,
                 n_svgs=3, verbose=False)

    assert result is data
    assert data.X is matrix_object
    assert_matrix_unchanged(data.X, expression_before)
    pd.testing.assert_frame_equal(data.obs, obs_before)
    pd.testing.assert_index_equal(data.var_names, var_before)
    np.testing.assert_array_equal(data.obsm['spatial'], coordinates_before)
    genes = direct['gene_indices']
    p = np.full(data.n_vars, np.nan)
    q = np.full(data.n_vars, np.nan)
    p[genes] = direct['res_mtest']['combinedPval']
    q[genes] = direct['res_mtest']['adjustedPval']
    np.testing.assert_allclose(data.var['sparkx_pval'], p, equal_nan=True)
    np.testing.assert_allclose(data.var['sparkx_qval'], q, equal_nan=True)
    finite = np.isfinite(p)
    np.testing.assert_allclose(q[finite], multipletests(p[finite], method='fdr_by')[1])
    for key, native in [('sparkx_stats', 'stats'), ('sparkx_pvals', 'res_stest')]:
        stored = data.varm[key]
        pd.testing.assert_index_equal(stored.index, data.var_names)
        assert list(stored.columns) == list(direct[native].columns)
        np.testing.assert_allclose(stored.iloc[genes], direct[native], equal_nan=True)
        assert stored.iloc[3].isna().all()
    selected = expected_selection(p, q, 3, .05)
    np.testing.assert_array_equal(data.var['space_variable_features'], selected)
    np.testing.assert_array_equal(data.var['highly_variable'], selected)
    run = data.uns['space_svg_runs']['sparkx']
    assert run['version'] == backend.__version__
    assert run['correction_method'] == 'BY'
    assert run['input']['expression'] == 'X'
    assert run['input']['covariates']['source'] == ('obs' if with_covariates else 'none')
    np.testing.assert_array_equal(run['filter']['gene_indices'], genes)
    np.testing.assert_array_equal(run['filter']['spot_indices'], direct['spot_indices'])
    np.testing.assert_array_equal(run['filter']['filtered_gene_indices'], [3])
    np.testing.assert_array_equal(run['filter']['filtered_spot_indices'], [13])
    assert run['backend_metadata']['intercept_added'] is False


def test_default_significance_explicit_top_n_and_stable_ties(backend):
    data = spatial_data()
    svg(data, mode='sparkx', option='single', verbose=False, n_svgs=None)
    assert data.var['space_variable_features'].sum() == 3
    assert data.var.loc['gene-0', 'sparkx_pval'] == data.var.loc['gene-1', 'sparkx_pval']
    assert data.var.loc['gene-0', 'sparkx_qval'] == data.var.loc['gene-1', 'sparkx_qval']
    svg(data, mode='sparkx', option='single', verbose=False, n_svgs=1, selection='top_n')
    assert data.var_names[data.var['space_variable_features']].tolist() == ['gene-0']
    svg(data, mode='sparkx', option='single', verbose=False, n_svgs=7, selection='top_n')
    assert data.var['space_variable_features'].sum() == 7
    assert (data.var.loc[data.var['space_variable_features'], 'sparkx_qval'] >= .05).any()
    svg(data, mode='sparkx', option='single', verbose=False, n_svgs=None, qval_threshold=0)
    assert not data.var['space_variable_features'].any()
    svg(data, mode='sparkx', option='single', verbose=False, n_svgs=0, selection='top_n')
    assert not data.var['highly_variable'].any()


def test_custom_threshold_is_strict_and_caps_significant_genes(backend):
    data = spatial_data()
    svg(data, mode='sparkx', option='single', n_svgs=None, verbose=False)
    cutoff = float(data.var.loc['gene-0', 'sparkx_qval'])
    svg(data, mode='sparkx', option='single', n_svgs=2, qval_threshold=cutoff, verbose=False)
    assert not data.var.loc[['gene-0', 'gene-1'], 'space_variable_features'].any()
    svg(data, mode='sparkx', option='single', n_svgs=1, qval_threshold=.05, verbose=False)
    assert data.var['space_variable_features'].sum() == 1


def test_layers_are_explicit_and_source_matrices_are_unchanged(backend):
    data = spatial_data('csr')
    counts = data.X.copy()
    data.layers['counts'] = counts
    data.X = data.X.copy()
    data.X.data = np.log1p(data.X.data)
    transformed_before = data.X.copy()
    reference_x = backend.sparkx(data.X.T, data.obsm['spatial'], option='single', verbose=False)
    reference_counts = backend.sparkx(counts.T, data.obsm['spatial'], option='single', verbose=False)
    idx = reference_x['gene_indices']
    svg(data, mode='sparkx', option='single', verbose=False)
    np.testing.assert_allclose(data.var['sparkx_pval'].iloc[idx], reference_x['res_mtest']['combinedPval'])
    assert data.uns['space_svg_runs']['sparkx']['input']['expression'] == 'X'
    svg(data, mode='sparkx', layer='counts', option='single', verbose=False)
    np.testing.assert_allclose(data.var['sparkx_pval'].iloc[idx], reference_counts['res_mtest']['combinedPval'])
    assert data.uns['space_svg_runs']['sparkx']['input']['expression'] == 'layers/counts'
    assert not np.allclose(reference_x['res_mtest'], reference_counts['res_mtest'])
    assert_matrix_unchanged(data.X, transformed_before)
    assert_matrix_unchanged(data.layers['counts'], counts)


@pytest.mark.parametrize('covariate_source', ['obs', 'frame', 'array', 'one_obs'])
def test_explicit_covariate_forms_preserve_scale_and_intercept(backend, covariate_source):
    data = spatial_data()
    frame = data.obs[['technical']].copy()
    values = frame.to_numpy().copy()
    argument = {'obs': ['technical'], 'frame': frame, 'array': values, 'one_obs': 'technical'}[covariate_source]
    direct = backend.sparkx(data.X.T, data.obsm['spatial'], X_in=values,
                            option='single', verbose=False)
    svg(data, mode='sparkx', covariates=argument, option='single', verbose=False)
    np.testing.assert_allclose(data.var['sparkx_pval'].iloc[direct['gene_indices']],
                               direct['res_mtest']['combinedPval'], equal_nan=True)
    np.testing.assert_array_equal(values, frame.to_numpy())
    assert data.uns['space_svg_runs']['sparkx']['backend_metadata']['covariates'] == 1


def test_custom_coordinate_key_reordering_and_duplicate_gene_names(backend):
    data = spatial_data('csc')
    data.obsm['xy'] = data.obsm.pop('spatial')
    data = data[np.arange(data.n_obs)[::-1], [7, 3, 1, 0, 5, 4, 2, 6]].copy()
    data.var_names = ['duplicate', 'zero', 'duplicate', 'signal', 'g5', 'g4', 'g2', 'g6']
    direct = backend.sparkx(data.X.T, data.obsm['xy'], option='single', verbose=False)
    svg(data, mode='sparkx', spatial_key='xy', option='single', n_svgs=1,
        selection='top_n', verbose=False)
    np.testing.assert_allclose(data.var['sparkx_pval'].iloc[direct['gene_indices']],
                               direct['res_mtest']['combinedPval'])
    assert np.isnan(data.var['sparkx_pval'].iloc[1])
    assert data.var['space_variable_features'].tolist() == [False, False, True, False, False, False, False, False]


def test_sparse_dispatch_preserves_storage_and_native_arguments(monkeypatch):
    data = spatial_data('csr')
    class Dispatched(Exception):
        pass
    def capture(count_in, locus_in, **kwargs):
        assert sparse.issparse(count_in)
        assert count_in.shape == (data.n_vars, data.n_obs)
        np.testing.assert_array_equal(count_in.data, data.X.data)
        np.testing.assert_array_equal(locus_in, data.obsm['spatial'])
        assert kwargs == {'X_in': None, 'numCores': 2, 'option': 'single', 'verbose': False}
        raise Dispatched
    monkeypatch.setattr(adapter, 'import_module', lambda name: SimpleNamespace(sparkx=capture))
    with pytest.raises(Dispatched):
        svg(data, mode='sparkx', option='single', n_jobs=2, verbose=False)


def test_rerun_overwrites_kernel_results_and_serializes(backend, tmp_path):
    data = spatial_data('csr')
    svg(data, mode='sparkx', option='mixture', verbose=False)
    assert data.varm['sparkx_stats'].shape[1] == 11
    svg(data, mode='sparkx', option='single', verbose=False, n_svgs=None)
    assert list(data.varm['sparkx_stats'].columns) == ['projection']
    assert list(data.varm['sparkx_pvals'].columns) == ['projection']
    assert data.uns['space_svg_runs']['sparkx']['params']['option'] == 'single'
    path = tmp_path / 'sparkx.h5ad'
    data.write_h5ad(path)
    restored = read_h5ad(path)
    pd.testing.assert_frame_equal(restored.var, data.var)
    pd.testing.assert_frame_equal(restored.varm['sparkx_stats'], data.varm['sparkx_stats'])
    np.testing.assert_array_equal(restored.uns['space_svg_runs']['sparkx']['filter']['gene_indices'],
                                  data.uns['space_svg_runs']['sparkx']['filter']['gene_indices'])


@pytest.mark.parametrize('source', ['obs', 'uns', 'both', 'stale_uns'])
def test_multiple_slices_are_rejected_before_backend_import(monkeypatch, source):
    data = spatial_data()
    kwargs = {}
    if source in ('obs', 'both', 'stale_uns'):
        data.obs['slice'] = ['a'] * (data.n_obs // 2) + ['b'] * (data.n_obs // 2)
        kwargs['library_key'] = 'slice'
    if source in ('uns', 'both', 'stale_uns'):
        data.uns['spatial'] = {'a': {}, 'b': {}}
    if source == 'stale_uns':
        data.obs['slice'] = 'a'
    def prohibit_import(*args):
        raise AssertionError('Multi-slice input should fail before backend import')
    monkeypatch.setattr(adapter, 'import_module', prohibit_import)
    with pytest.raises(ValueError, match='one spatial library'):
        svg(data, mode='sparkx', **kwargs)
    assert 'sparkx_pval' not in data.var


def test_single_explicit_library_is_supported(backend):
    data = spatial_data()
    data.obs['slice'] = 'a'
    data.uns['spatial'] = {'a': {}}
    svg(data, mode='sparkx', library_key='slice', option='single', verbose=False)
    assert data.uns['space_svg_runs']['sparkx']['input']['library_key'] == 'slice'


@pytest.mark.parametrize('shape', [(0, 8), (48, 0)])
def test_empty_expression_axes_have_a_clear_error(shape):
    data = AnnData(np.empty(shape))
    data.obsm['spatial'] = np.ones((shape[0], 2))
    with pytest.raises(ValueError, match='nonempty expression'):
        svg(data, mode='sparkx')


@pytest.mark.parametrize('case,match', [
    ('missing_layer', 'layer'), ('missing_spatial', 'spatial_key'),
    ('nonfinite_spatial', 'finite'), ('missing_covariate', 'Covariate columns'),
    ('wrong_covariate_order', 'exactly match'), ('wrong_covariate_rows', 'one row'),
    ('one_dimensional_covariates', '2-D'), ('nonfinite_covariates', 'finite'),
    ('categorical_covariates', 'numeric'), ('empty_covariates', 'at least one'),
    ('missing_library', 'library_key'), ('missing_library_label', 'missing labels'),
])
def test_invalid_adapter_inputs_fail_before_backend_import(monkeypatch, case, match):
    data = spatial_data()
    kwargs = {}
    if case == 'missing_layer':
        kwargs['layer'] = 'missing'
    elif case == 'missing_spatial':
        kwargs['spatial_key'] = 'missing'
    elif case == 'nonfinite_spatial':
        data.obsm['spatial'][0, 0] = np.nan
    elif case == 'missing_covariate':
        kwargs['covariates'] = 'missing'
    elif case == 'wrong_covariate_order':
        kwargs['covariates'] = data.obs[['technical']].iloc[::-1]
    elif case == 'wrong_covariate_rows':
        kwargs['covariates'] = np.ones((data.n_obs - 1, 1))
    elif case == 'one_dimensional_covariates':
        kwargs['covariates'] = np.ones(data.n_obs)
    elif case == 'nonfinite_covariates':
        kwargs['covariates'] = np.full((data.n_obs, 1), np.nan)
    elif case == 'categorical_covariates':
        data.obs['batch'] = 'a'
        kwargs['covariates'] = ['batch']
    elif case == 'empty_covariates':
        kwargs['covariates'] = []
    elif case == 'missing_library':
        kwargs['library_key'] = 'missing'
    elif case == 'missing_library_label':
        data.obs['slice'] = 'a'
        data.obs.iloc[0, data.obs.columns.get_loc('slice')] = None
        kwargs['library_key'] = 'slice'
    def prohibit_import(*args):
        raise AssertionError('Invalid input should fail before backend import')
    monkeypatch.setattr(adapter, 'import_module', prohibit_import)
    with pytest.raises((ValueError, KeyError), match=match):
        svg(data, mode='sparkx', **kwargs)
    assert 'sparkx_pval' not in data.var


@pytest.mark.parametrize('case,match', [
    ('all_zero', 'at least two retained'), ('one_gene', 'at least two retained'),
    ('singular_design', 'singular'), ('degenerate_coordinates', 'singular'),
    ('nonfinite_expression', 'finite'), ('bad_option', 'option'), ('bad_jobs', 'positive integer'),
])
def test_backend_errors_preserve_existing_results(backend, case, match):
    data = spatial_data()
    svg(data, mode='sparkx', option='single', verbose=False)
    kwargs = {'option': 'single', 'verbose': False}
    if case == 'all_zero':
        data.X[:] = 0
    elif case == 'one_gene':
        data.X[:, 1:] = 0
    elif case == 'singular_design':
        kwargs['covariates'] = np.ones((data.n_obs, 2))
    elif case == 'degenerate_coordinates':
        data.obsm['spatial'][:] = 0
    elif case == 'nonfinite_expression':
        data.X[0, 0] = np.nan
    elif case == 'bad_option':
        kwargs['option'] = 'invalid'
    elif case == 'bad_jobs':
        kwargs['n_jobs'] = 0
    before = data.copy()
    with pytest.raises(ValueError, match=match):
        svg(data, mode='sparkx', **kwargs)
    pd.testing.assert_frame_equal(data.var, before.var)
    pd.testing.assert_frame_equal(data.varm['sparkx_stats'], before.varm['sparkx_stats'])
    assert_matrix_unchanged(data.X, before.X)
    assert data.uns['space_svg_runs']['sparkx']['params'] == before.uns['space_svg_runs']['sparkx']['params']


@pytest.mark.parametrize('n_svgs', [-1, 2.5, True])
def test_existing_svg_cap_validation_applies(n_svgs):
    with pytest.raises(ValueError, match='n_svgs'):
        svg(spatial_data(), mode='sparkx', n_svgs=n_svgs)


@pytest.mark.parametrize('threshold', [-.1, 1.1, np.nan, np.inf])
def test_existing_svg_threshold_validation_applies(threshold):
    with pytest.raises(ValueError, match='qval_threshold'):
        svg(spatial_data(), mode='sparkx', qval_threshold=threshold)


def test_missing_backend_has_local_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, 'sparkx', None)
    with pytest.raises(ImportError, match='sparkx-py.*pip install /path/to/py-SPARK-X'):
        svg(spatial_data(), mode='sparkx')


def test_backend_is_not_imported_when_adapter_is_loaded(monkeypatch):
    monkeypatch.setitem(sys.modules, 'sparkx', None)
    importlib.reload(adapter)
    assert sys.modules['sparkx'] is None


def test_unrecognized_method_option_is_not_silently_ignored():
    with pytest.raises(TypeError, match='unexpected keyword'):
        svg(spatial_data(), mode='sparkx', normalize=True)
