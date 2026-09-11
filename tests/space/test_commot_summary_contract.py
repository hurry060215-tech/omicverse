import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from omicverse.space._commot import (
    create_communication_anndata,
    process_all_commot,
    update_classification_from_database,
)


def fixture():
    a = ad.AnnData(np.ones((5, 2)))
    a.obs['type'] = ['A', 'A', 'B', 'B', 'B']
    db = pd.DataFrame({'ligand': ['L-X'], 'receptor': ['R_Y'], 'pathway': ['P-Q']})
    matrix = np.zeros((5, 5))
    matrix[:2, 2:] = 1
    for name in ['db-one', 'db-two']:
        a.uns[f'commot-{name}-info'] = {'df_ligrec': db.copy()}
        for suffix in ['L-X-R_Y', 'P-Q', 'total-total']:
            a.obsp[f'commot-{name}-{suffix}'] = sparse.csr_matrix(matrix)
    return a


def test_ambiguous_database_rejected():
    with pytest.raises(ValueError, match='database_name'):
        create_communication_anndata(fixture(), 'type', 9, level='lr')


@pytest.mark.parametrize('level,suffix', [('lr', 'L-X-R_Y'), ('pathway', 'P-Q'), ('total', 'total-total')])
def test_exact_metadata_levels_and_h5ad(level, suffix, tmp_path):
    a = fixture()
    out = create_communication_anndata(a, 'type', 9, database_name='db-one', level=level)
    assert list(out.var_names) == ['commot-db-one-' + suffix]
    assert out.X[out.obs_names.get_loc('A|B'), 0] == 6
    assert out.obs.loc['A|B', 'n_sender'] == 2
    assert out.obs.loc['A|B', 'n_receiver'] == 3
    assert out.var.iloc[0]['secreted'] == 'Unknown'
    if level == 'lr':
        assert out.var.iloc[0]['gene_a'] == 'L-X'
        assert out.var.iloc[0]['gene_b'] == 'R_Y'
        assert out.var.iloc[0]['classification'] == 'P-Q'
    out.write_h5ad(tmp_path / 'comm.h5ad')
    restored = ad.read_h5ad(tmp_path / 'comm.h5ad')
    assert restored.uns['commot_summary']['statistic'] == 'sum'


def test_mean_dict_and_rng_restoration():
    a = fixture()
    state = np.random.get_state()
    out = process_all_commot(a, 'type', 9, 'dict', database_name='db-one', level='lr', statistic='mean')
    assert out['commot-db-one-L-X-R_Y']['communication'].loc['A', 'B'] == 1
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])


def test_single_database_legacy_all_and_annotation_update():
    a = fixture()
    del a.uns['commot-db-two-info']
    with pytest.warns(UserWarning, match='overlap'):
        out = create_communication_anndata(a, 'type', 9)
    assert out.n_vars == 3
    a.uns['commot-db-one-info']['df_ligrec']['pathway'] = 'new'
    update_classification_from_database(out, a)
    assert out.var.loc['commot-db-one-L-X-R_Y', 'classification'] == 'new'


def test_missing_metadata_never_guesses_hyphenated_names():
    a = fixture()
    a.uns.clear()
    with pytest.raises(ValueError, match='metadata|df_ligrec'):
        create_communication_anndata(a, 'type', 9, database_name='db-one', level='lr')


def test_summary_works_with_existing_cellchatviz():
    from omicverse.pl import CellChatViz

    summary = create_communication_anndata(fixture(), 'type', 9, database_name='db-one', level='lr')
    viz = CellChatViz(summary)
    assert set(viz.cell_types) == {'A', 'B'}
    aggregated = viz.compute_aggregated_network(pvalue_threshold=1.0)
    assert aggregated is not None


@pytest.mark.parametrize('aliases,function', [
    (['commot汇总', '细胞通信汇总', 'cellchat格式转换'], create_communication_anndata),
    (['commot_pathway_update', '通信数据库注释', '更新pathway分类'], update_classification_from_database),
])
def test_existing_registry_aliases_and_prerequisites(aliases, function):
    from inspect import unwrap
    from omicverse._registry import get_registry

    registry = get_registry()
    for alias in aliases:
        assert unwrap(registry.get_function(alias)) is unwrap(function)
    check = registry.check_prerequisites(function.__name__, ad.AnnData(np.ones((2, 2))))
    assert not check['satisfied']
    assert check['missing_structures']


def test_annotation_update_after_h5ad_roundtrip(tmp_path):
    a = fixture()
    db = a.uns['commot-db-one-info']['df_ligrec']
    db.loc[1] = ['L2', 'R_Y', 'P-Q']
    a.obsp['commot-db-one-L2-R_Y'] = a.obsp['commot-db-one-L-X-R_Y'].copy()
    out = create_communication_anndata(a, 'type', 2, database_name='db-one', level='lr')
    out.var['user_note'] = ['keep-a', 'keep-b']
    out.write_h5ad(tmp_path / 'summary.h5ad')
    restored = ad.read_h5ad(tmp_path / 'summary.h5ad')
    assert isinstance(restored.var['classification'].dtype, pd.CategoricalDtype)
    db['pathway'] = 'new-pathway'
    db['secreted'] = 'True'
    update_classification_from_database(restored, a)
    assert restored.var['classification'].tolist() == ['new-pathway'] * 2
    assert restored.var['secreted'].tolist() == ['True'] * 2
    assert restored.var['user_note'].tolist() == ['keep-a', 'keep-b']
    restored.write_h5ad(tmp_path / 'updated.h5ad')
    assert ad.read_h5ad(tmp_path / 'updated.h5ad').var['classification'].tolist() == ['new-pathway'] * 2


def test_quick_demo_limits_work_before_summarizing(monkeypatch):
    from omicverse.space import _commot

    a = fixture()
    del a.uns['commot-db-two-info']
    db = a.uns['commot-db-one-info']['df_ligrec']
    db.loc[1] = ['L2', 'R_Y', 'P-Q']
    a.obsp['commot-db-one-L2-R_Y'] = a.obsp['commot-db-one-L-X-R_Y'].copy()
    original_keys = list(a.obsp)
    summarize = _commot._get_summarize_cluster_gpu()
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return summarize(*args, **kwargs)

    monkeypatch.setattr(_commot, '_get_summarize_cluster_gpu', lambda: counted)
    out = _commot.quick_demo(a, 'type', max_pathways=1)
    assert len(calls) == 1
    assert list(out.var_names) == ['commot-db-one-L-X-R_Y']
    assert out.X[out.obs_names.get_loc('A|B'), 0] == 6
    assert list(a.obsp) == original_keys
    assert len(db) == 2
