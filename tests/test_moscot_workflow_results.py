"""Native moscot result identity, persistence and failure-path checks."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from omicverse._moscot import moscot_temporal, moscot_mapping, moscot_alignment
from omicverse._moscot_results import moscot_coupling, moscot_impute, moscot_align_coordinates


def mapping_fixture():
    ref = ad.AnnData(np.array([[1,8,2],[2,7,4],[8,1,12],[7,2,15]],dtype=float),
                      obs=pd.DataFrame(index=['r0','r1','r2','r3']),var=pd.DataFrame(index=['a','b','heldout']))
    sp = ad.AnnData(np.array([[1,8],[8,1],[2,7],[7,2]],dtype=float),
                    obs=pd.DataFrame({'slice':pd.Categorical(['A','B','A','B'])},index=['A0','B0','A1','B1']),
                    var=pd.DataFrame(index=['a','b']))
    sp.obsm['spatial'] = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    return ref,sp


def test_native_batched_mapping_preserves_interleaved_identity_and_roundtrip(tmp_path):
    pytest.importorskip('moscot')
    ref,sp=mapping_fixture()
    p=moscot_mapping(ref,sp,batch_key='slice',prepare_kwargs={'joint_attr':{'attr':'X'}},
                     solve_kwargs={'alpha':0,'epsilon':.2})
    result=moscot_impute(p,var_names=['heldout','a'],batch_size=1)
    assert result.obs_names.equals(sp.obs_names)
    for key,sol in p.solutions.items():
        sub=p.problems[key]
        expected=np.asarray(sol.pull(ref[sub.adata_tgt.obs_names,['heldout','a']].X,scale_by_marginals=True))
        np.testing.assert_allclose(result[sub.adata_src.obs_names].X,expected)
    result.write_h5ad(tmp_path/'imputed.h5ad')
    reloaded=ad.read_h5ad(tmp_path/'imputed.h5ad')
    np.testing.assert_array_equal(reloaded.X,result.X)
    assert reloaded.obs_names.equals(sp.obs_names)
    np.testing.assert_array_equal(sp.X,[[1,8],[8,1],[2,7],[7,2]])
    with pytest.raises(ValueError,match='max_entries'):
        moscot_impute(p,max_entries=1)
    with pytest.raises(ValueError,match='var_names'):
        moscot_impute(p,var_names=['missing'])


def test_native_temporal_coupling_roundtrip_names_and_marginals(tmp_path):
    pytest.importorskip('moscot')
    a=ad.AnnData(np.ones((10,5)))
    a.obs_names=[f'cell{i}' for i in range(10)]
    a.obs['day']=np.repeat([0,1],5)
    a.obsm['X_known']=np.vstack([np.eye(5),np.eye(5)])
    p=moscot_temporal(a,time_key='day',joint_attr='X_known',solve_kwargs={'epsilon':.05})
    result=moscot_coupling(p,(0,1))
    assert list(result.obs_names)==list(a.obs_names[:5])
    assert list(result.var_names)==list(a.obs_names[5:])
    np.testing.assert_allclose(result.X.sum(0),np.full(5,.2),atol=1e-5)
    result.write_h5ad(tmp_path/'coupling.h5ad')
    np.testing.assert_array_equal(ad.read_h5ad(tmp_path/'coupling.h5ad').X,result.X)
    with pytest.raises(ValueError,match='max_entries'):
        moscot_coupling(p,(0,1),max_entries=2)


def test_native_alignment_preserves_coordinates_and_native_result(tmp_path):
    pytest.importorskip('moscot')
    rng=np.random.default_rng(7)
    a=ad.AnnData(rng.uniform(.1,3,(16,8)))
    a.obs['slice']=pd.Categorical(np.tile(['A','B'],8))
    a.obsm['spatial']=rng.normal(size=(16,2))
    a.obsm['X_pca']=rng.normal(size=(16,3))
    original=a.obsm['spatial'].copy()
    p=moscot_alignment(a,batch_key='slice',joint_attr='X_pca',solve_kwargs={'epsilon':.5,'alpha':.5})
    expected,_=p.align(reference='A',mode='warp')
    result=moscot_align_coordinates(p,reference='A')
    np.testing.assert_allclose(result.obsm['spatial_aligned'],expected)
    np.testing.assert_array_equal(a.obsm['spatial'],original)
    assert 'spatial_aligned' not in a.obsm
    result.write_h5ad(tmp_path/'aligned.h5ad')
    np.testing.assert_array_equal(ad.read_h5ad(tmp_path/'aligned.h5ad').obsm['spatial_aligned'],expected)
    with pytest.raises(ValueError,match='different key'):
        moscot_align_coordinates(p,reference='A',key_added='spatial')


def test_alignment_rejects_misleading_native_reference_policy():
    from types import SimpleNamespace
    a=ad.AnnData(np.ones((3,2)))
    p=SimpleNamespace(adata=a,_policy=SimpleNamespace(reference='actual'))
    with pytest.raises(ValueError,match='fixed reference'):
        moscot_align_coordinates(p,reference='different')
    with pytest.raises(ValueError,match='string batch label'):
        moscot_align_coordinates(p,reference=1)


def test_native_impute_scaling_is_not_conditional_expression():
    pytest.importorskip('moscot')
    ref,sp=mapping_fixture()
    ref.X[:,-1]=1.
    p=moscot_mapping(ref,sp,batch_key='slice',prepare_kwargs={'joint_attr':{'attr':'X'}},
                     solve_kwargs={'alpha':0,'epsilon':.2,'threshold':1e-7,'max_iterations':1000})
    predicted=moscot_impute(p,var_names=['heldout'])
    # Each subproblem has four reference columns and two spatial rows.
    np.testing.assert_allclose(predicted.X,2.,atol=2e-4)
    assert predicted.uns['moscot_impute']['normalization']=='solution.pull(scale_by_marginals=True)'
