"""DS workflow tests: real existing Python engines plus input/design contracts."""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from omicverse.bulk._pseudobulk_ds import pseudobulk_ds


def example(sparse_matrix=False):
    rng = np.random.default_rng(72)
    rows, obs = [], []
    for group in ['T', 'B']:
        for i in range(8):
            condition = 'case' if i >= 4 else 'control'
            for cell in range(12):
                mean = np.ones(70) * 3
                if condition == 'case': mean[:8] *= 4
                rows.append(rng.negative_binomial(10, 10 / (10 + mean)))
                obs.append((f's{i}', group, condition, f'd{i%4}', float(i%4), i%2))
    X = np.asarray(rows)
    obj = AnnData(X=np.log1p(X), obs=pd.DataFrame(obs, columns=['sample','type','condition','subject','age','batch'], index=[f'c{i}' for i in range(len(obs))]), var=pd.DataFrame(index=[f'g{i}' for i in range(70)]))
    obj.layers['counts'] = sparse.csr_matrix(X) if sparse_matrix else X
    return obj


def call(a, **kwargs):
    return pseudobulk_ds(a, sample_key='sample', group_key='type', condition_key='condition', control='control', treatment='case', **kwargs)


@pytest.mark.parametrize('sparse_matrix', [False, True])
def test_real_limma_recovery_and_nonmutation(sparse_matrix):
    a=example(sparse_matrix); before=a.copy()
    result=call(a)
    assert result.qc.status.tolist() == ['tested', 'tested']
    assert result.results.shape[0] == 140
    assert (result.results.query("gene in ['g0','g1','g2','g3']").log2FC > 1).all()
    assert result.results.qvalue.between(0,1).all()
    assert result.results.qvalue_global.between(0,1).all()
    np.testing.assert_array_equal(a.X, before.X)
    pd.testing.assert_frame_equal(a.obs, before.obs)
    for _, row in result.pseudobulk.obs.iterrows():
        ix=(a.obs['sample']==row['sample']) & (a.obs.type==row['type'])
        counts=a.layers['counts'][ix.to_numpy()]
        np.testing.assert_array_equal(result.pseudobulk[row.name].X[0], np.asarray(counts.sum(axis=0)).ravel())


def test_paired_and_covariate_models():
    paired=call(example(), paired_by='subject')
    assert paired.designs['T'].shape == (8,5)
    adjusted=call(example(), covariates=['age','batch'])
    assert adjusted.designs['T'].shape == (8,4)


def test_real_legacy_ql():
    r=call(example(), method='edger_ql')
    assert len(r.results) == 140
    assert np.isfinite(r.results.pvalue).all()
    assert (r.results.query("gene == 'g0'").log2FC > 1).all()


def test_missing_group_samples_not_fabricated():
    a=example(); a=a[~((a.obs.type=='B') & a.obs['sample'].isin(['s4','s5','s6']))].copy()
    r=call(a)
    assert len(r.pseudobulk.obs) == 13
    assert r.qc.set_index('group').loc['B','status'] == 'skipped'
    assert set(r.results.group) == {'T'}


def test_unpaired_samples_receipt():
    a=example(); a=a[~((a.obs.type=='B') & (a.obs['sample']=='s4'))].copy()
    r=call(a, paired_by='subject')
    row=r.qc.set_index('group').loc['B']
    assert row.n_control==row.n_treatment==3
    assert len(row.unpaired_profiles)==1


@pytest.mark.parametrize('bad', ['fractional','negative','nan','inf'])
def test_bad_counts_rejected(bad):
    a=example(); a.layers['counts']=a.layers['counts'].astype(float)
    a.layers['counts'][0,0]={'fractional':.5,'negative':-1,'nan':np.nan,'inf':np.inf}[bad]
    with pytest.raises(ValueError, match='raw counts'): call(a)


def test_log_x_requires_explicit_counts():
    with pytest.raises(ValueError, match='raw counts'): call(example(), layer=None)


def test_missing_and_inconsistent_metadata():
    a=example(); a.obs.loc['c0','condition']='other'
    with pytest.raises(ValueError, match='constant'): call(a)
    a=example(); a.obs.loc['c0','sample']=None
    with pytest.raises(ValueError, match='Missing labels'): call(a)


def test_rank_deficient():
    a=example(); a.obs['confound']=a.obs.condition
    with pytest.raises(ValueError, match='rank deficient'): call(a, covariates=['confound'])


def test_duplicate_subject_condition_rejected():
    a=example(); a.obs['subject']='one'
    with pytest.raises(ValueError, match='multiple samples'): call(a, paired_by='subject')


def test_duplicate_cell_gene_rejected():
    a=example(); a.obs_names=['same']*a.n_obs
    with pytest.raises(ValueError, match='unique'): call(a)
    a=example(); a.var_names=['same']*a.n_vars
    with pytest.raises(ValueError, match='unique'): call(a)


def test_direction_and_shuffle():
    a=example(); x=call(a).results.sort_values(['group','gene']).reset_index(drop=True)
    shuffled=a[np.random.default_rng(1).permutation(a.n_obs)].copy()
    y=call(shuffled).results.sort_values(['group','gene']).reset_index(drop=True)
    np.testing.assert_allclose(x.log2FC,y.log2FC,atol=1e-10)
    rev=pseudobulk_ds(a, sample_key='sample',group_key='type',condition_key='condition',control='case',treatment='control')
    z=rev.results.sort_values(['group','gene']).reset_index(drop=True)
    np.testing.assert_allclose(x.log2FC,-z.log2FC,atol=1e-10)


def test_other_condition_excluded_and_min_cells():
    a=example(); a.obs.loc[a.obs['sample']=='s7','condition']='other'
    r=call(a, min_cells=13)
    assert r.results.empty
    assert (r.qc.status=='skipped').all()
    assert r.parameters['excluded_other_condition_cells']==24


def test_design_dtypes_preserved():
    a=example(); a.obs['age']=a.obs.age.astype(float)
    r=call(a,covariates=['age'])
    assert 'cov:age' in r.designs['T']
    assert len(r.designs['T'].columns)==3


def test_backend_error_not_hidden(monkeypatch):
    import omicverse.bulk._pseudobulk_ds as module
    def fail(*args): raise RuntimeError('solver failed')
    monkeypatch.setattr(module,'_fit',fail)
    with pytest.raises(RuntimeError,match='solver failed'): call(example())


def test_existing_qc_metadata_cannot_override_actual_counts():
    a=example(); a.obs['psbulk_n_cells']=0; a.obs['psbulk_counts']=0
    r=call(a)
    assert (r.pseudobulk.obs.psbulk_n_cells==12).all()
    assert (r.pseudobulk.obs.psbulk_counts>0).all()
    assert (r.qc.status=='tested').all()


def test_reserved_model_column_rejected():
    a=example(); a.obs['psbulk_n_cells']=a.obs['sample']
    with pytest.raises(ValueError,match='reserved'):
        pseudobulk_ds(a,sample_key='psbulk_n_cells',group_key='type',condition_key='condition',control='control',treatment='case')


def test_preaggregation_exact_integer_bound():
    a=example(); a.layers['counts']=a.layers['counts'].astype(np.float64)
    a.layers['counts'][0,0]=2**53; a.layers['counts'][1,0]=1
    with pytest.raises(ValueError,match='aggregation bound'): call(a)


def test_encoded_covariate_name_collision_rejected():
    a=example()
    a.obs['x']=a.obs['sample'].map({f's{i}':('a' if i%2==0 else 'b') for i in range(8)})
    a.obs['x=b']=a.obs['sample'].map(dict(zip([f's{i}' for i in range(8)],[1,2,4,3,3,1,4,2])))
    with pytest.raises(ValueError,match='collide'): call(a,covariates=['x','x=b'])


def test_numeric_categorical_covariate_keeps_dummy_encoding():
    a=example()
    values=a.obs['sample'].map(dict(zip([f's{i}' for i in range(8)],[1,2,3,1,2,3,1,2])))
    a.obs['numeric_category']=pd.Categorical(values,categories=[1,2,3])
    r=call(a,covariates=['numeric_category'])
    assert isinstance(r.pseudobulk.obs['numeric_category'].dtype,pd.CategoricalDtype)
    assert r.designs['T'].shape[1]==4
    assert {'cov:numeric_category=2','cov:numeric_category=3'}.issubset(r.designs['T'].columns)


def test_mixed_category_level_collision():
    a=example(); a.obs['ambiguous']=pd.Categorical([1 if i%2 else '1' for i in a.obs['sample'].str[1:].astype(int)],categories=[1,'1'])
    with pytest.raises(ValueError,match='levels collide'): call(a,covariates=['ambiguous'])


def test_constant_categorical_covariate():
    a=example(); a.obs['constant']='batch1'
    with pytest.raises(ValueError,match='one observed level'): call(a,covariates=['constant'])
