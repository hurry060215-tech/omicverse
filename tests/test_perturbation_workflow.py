import numpy as np
import pandas as pd
import anndata as ad
import pytest
from omicverse.single._perturbation_workflow import (prepare_perturbation_data,
    perturbation_sample_summary,perturbation_baseline)


def test_real_guide_rna_identity_missing_and_multiple(tmp_path):
    pytest.importorskip('pertpy')
    rna=ad.AnnData(np.ones((5,3)),obs=pd.DataFrame(index=['a','b','c','d','missing']))
    guides=ad.AnnData(np.array([[1,9],[9,0],[8,8],[0,0]]),
                      obs=pd.DataFrame(index=['b','a','c','d']),var=pd.DataFrame(index=['NT','G1']))
    original=rna.X.copy()
    out=prepare_perturbation_data(rna,guides,guide_to_target={'NT':'control','G1':'GENE'},
                                   control_guides=['NT'],assignment_threshold=5)
    assert out.obs_names.equals(rna.obs_names)
    assert out.obs.loc['a','perturbation_target']=='control'
    assert out.obs.loc['b','perturbation_target']=='GENE'
    assert out.obs.loc['c','perturbation_status']=='multiple_guides'
    assert out.obs.loc['d','perturbation_status']=='unassigned'
    assert out.obs.loc['missing','perturbation_status']=='missing_guide_modality'
    assert pd.isna(out.obs.loc['c','perturbation_target'])
    np.testing.assert_array_equal(rna.X,original)
    assert not len(rna.obs.columns)
    out.write_h5ad(tmp_path/'joined.h5ad')
    loaded=ad.read_h5ad(tmp_path/'joined.h5ad')
    assert loaded.obs.equals(out.obs)


def baseline_fixture():
    x=ad.AnnData(np.array([[0,0],[2,4],[2,2],[4,6],[1,1],[3,5]],float))
    x.obs['sample']=['s1','s1','s2','s2','s3','s3']
    x.obs['condition']=['ctrl','stim']*3
    x.obs['class']=['NP','KO']*3
    return x


def test_donor_baselines_and_no_leakage():
    a=baseline_fixture()
    result=perturbation_baseline(a,sample_key='sample',perturbation_key='condition',control='ctrl',
                                train_samples=['s1','s2'],test_samples=['s3'],strategy='condition_mean')
    np.testing.assert_allclose(result.rmse,0)
    result=perturbation_baseline(a,sample_key='sample',perturbation_key='condition',control='ctrl',
                                train_samples=['s1','s2'],test_samples=['s3'],strategy='control_mean')
    assert result.loc[result.perturbation=='stim','rmse'].iloc[0]==pytest.approx(np.sqrt(10))
    with pytest.raises(ValueError,match='disjoint'):
        perturbation_baseline(a,sample_key='sample',perturbation_key='condition',control='ctrl',
                               train_samples=['s1'],test_samples=['s1','s2'])
    with pytest.raises(ValueError,match='Every.*control'):
        perturbation_baseline(a[1:].copy(),sample_key='sample',perturbation_key='condition',control='ctrl',
                               train_samples=['s1','s2'],test_samples=['s3'])


def test_sample_summary_missing_retained():
    a=baseline_fixture()
    a.obs.loc[a.obs_names[1],'class']=np.nan
    result=perturbation_sample_summary(a,sample_key='sample',perturbation_key='condition',classification_key='class')
    assert result.n_cells.sum()==a.n_obs
    assert '<missing>' in set(result['class'])
    np.testing.assert_allclose(result.fraction,1)


def test_split_rejects_shared_donor_across_libraries():
    a=baseline_fixture()
    a.obs['donor']=['D1','D1','D2','D2','D1','D1']
    with pytest.raises(ValueError,match='disjoint biological donors'):
        perturbation_baseline(a,sample_key='sample',donor_key='donor',perturbation_key='condition',
            control='ctrl',train_samples=['s1','s2'],test_samples=['s3'])


def test_summary_missing_label_collision_rejected():
    a=baseline_fixture()
    a.obs.loc[a.obs_names[0],'class']='<missing>'
    a.obs.loc[a.obs_names[1],'class']=np.nan
    with pytest.raises(ValueError,match='collides'):
        perturbation_sample_summary(a,sample_key='sample',perturbation_key='condition',classification_key='class')


def test_native_guide_sparse_storage_independent():
    pytest.importorskip('pertpy')
    from scipy import sparse
    rna=ad.AnnData(np.ones((2,2)))
    matrix=sparse.csr_matrix((np.array([8.,8.,4.,4.]),np.array([1,0,0,0]),np.array([0,2,4])),shape=(2,2))
    g=ad.AnnData(matrix,obs=rna.obs.copy(),var=pd.DataFrame(index=['NT','G']))
    kwargs=dict(guide_to_target={'NT':'control','G':'GENE'},control_guides=['NT'],
                assignment_threshold=5,multiple_guides='max')
    got=prepare_perturbation_data(rna,g,**kwargs)
    dense=g.copy();dense.X=matrix.toarray()
    expected=prepare_perturbation_data(rna,dense,**kwargs)
    assert got.obs.equals(expected.obs)
    assert got.obs['perturbation_guide'].tolist()==['NT','NT']
    np.testing.assert_array_equal(g.X.indices,[1,0,0,0])
    assert got.obs['perturbation_n_guides'].tolist()==[2,1]


def test_guide_mapping_slash_ids_survive_h5ad(tmp_path):
    pytest.importorskip('pertpy')
    rna=ad.AnnData(np.ones((2,2)))
    guides=ad.AnnData(np.array([[8,0],[0,9]]),var=pd.DataFrame(index=['NT/01','g/A']))
    out=prepare_perturbation_data(rna,guides,guide_to_target={'NT/01':'control','g/A':'A'},
        control_guides=['NT/01'],assignment_threshold=5)
    out.write_h5ad(tmp_path/'slash.h5ad')
    restored=ad.read_h5ad(tmp_path/'slash.h5ad')
    assert restored.uns['perturbation']['guide_to_target']['guide_ids'].tolist()==['NT/01','g/A']
