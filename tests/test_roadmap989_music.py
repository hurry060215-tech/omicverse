"""Frozen reference tests from official MuSiC R source, not music-py."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_music_test',ROOT/'omicverse/bulk/_music.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
run=module.music_deconvolution
FIX=Path(__file__).parent/'fixtures/music'


def load(case):
    z=json.loads((FIX/f'{case}.json').read_text())
    ad=AnnData(np.asarray(z['counts'],float)); ad.obs_names=z['cells'];ad.var_names=z['genes']
    ad.obs['ct']=z['types'];ad.obs['donor']=z['donors']
    if case=='sparse_reference':ad.X=sparse.csr_matrix(ad.X)
    b=pd.DataFrame(z['bulk'],index=z['bulk_genes'],columns=z['bulk_samples'])
    kw={'max_iter':z['max_iter']}
    if 'markers'in z:kw['markers']=z['markers']
    if 'cell_sizes'in z:kw['cell_sizes']=pd.Series(z['cell_sizes'],index=z['size_types'])
    return ad,b,kw


@pytest.mark.parametrize('case',['balanced','missing_donor_type','marker_subset','custom_cell_sizes','zero_bulk_gene','permuted_bulk','zero_in_one_bulk','sparse_reference','maxiter'])
def test_official_r_source_parity(case):
    ad,b,kw=load(case);before=ad.copy();bb=b.copy()
    out=run(b,ad,cell_type_key='ct',sample_key='donor',**kw)
    ref=json.loads((FIX/f'{case}-R.json').read_text())
    for rk,pk in [('Est.prop.weighted','proportions'),('Est.prop.allgene','nnls_proportions'),('Weight.gene','gene_weights'),('Var.prop','variance')]:
        np.testing.assert_allclose(out[pk],np.asarray(ref[rk],float),atol=1e-8,rtol=1e-7,equal_nan=True)
    np.testing.assert_allclose(out['diagnostics'].r_squared,ref['r.squared.full'],atol=1e-8,rtol=1e-7)
    assert out['proportions'].index.equals(b.columns)
    assert list(out['proportions'].columns)==['A','B','C']
    np.testing.assert_allclose(out['proportions'].sum(axis=1),1)
    pd.testing.assert_frame_equal(b,bb);pd.testing.assert_frame_equal(ad.obs,before.obs)
    if sparse.issparse(ad.X):assert (ad.X!=before.X).nnz==0
    else:np.testing.assert_array_equal(ad.X,before.X)


@pytest.mark.parametrize('kind',['negative','nan','duplicate_gene','single_donor','missing_label','bad_size','bad_iters','zero_bulk','no_common'])
def test_invalid_inputs_fail(kind):
    ad,b,kw=load('balanced')
    if kind=='negative':ad.X[0,0]=-1
    if kind=='nan':ad.X[0,0]=np.nan
    if kind=='duplicate_gene':ad.var_names=['x']*ad.n_vars
    if kind=='single_donor':ad.obs['donor']='d1'
    if kind=='missing_label':ad.obs.iloc[0,0]=None
    if kind=='bad_size':kw['cell_sizes']=pd.Series([-1,2,3],index=['A','B','C'])
    if kind=='bad_iters':kw['max_iter']=0
    if kind=='zero_bulk':b.iloc[:,0]=0
    if kind=='no_common':b.index=['missing'+str(i) for i in range(len(b))]
    with pytest.raises(ValueError):run(b,ad,cell_type_key='ct',sample_key='donor',**kw)


@pytest.mark.parametrize('case',['recursive','recursive_zero','recursive_sparse'])
def test_two_level_native_recursion(case):
    z=json.loads((FIX/f'{case}.json').read_text())
    ad=AnnData(np.asarray(z['counts'],float));ad.obs_names=z['cells'];ad.var_names=z['genes']
    ad.obs['ct']=z['types'];ad.obs['donor']=z['donors'];ad.obs['group']=z['groups']
    if case=='recursive_sparse':ad.X=sparse.csr_matrix(ad.X)
    bulk=pd.DataFrame(z['bulk'],index=z['bulk_genes'],columns=z['bulk_samples'])
    out=module.music_deconvolution_recursive(bulk,ad,group_key='group',cell_type_key='ct',sample_key='donor',group_cell_types=z['group_cell_types'],group_markers=z['group_markers'])
    r=json.loads((FIX/f'{case}-R.json').read_text())
    np.testing.assert_allclose(out['proportions'],r['Est.prop.weighted.cluster'],atol=1e-8,rtol=1e-7)
    np.testing.assert_allclose(out['proportions'].sum(axis=1),1)
    np.testing.assert_allclose(out['proportions'][['A','B']].sum(axis=1),out['group_proportions']['G1'])


@pytest.mark.parametrize('keyword',['cell_types','markers'])
def test_rejects_ambiguous_string_sequences(keyword):
    ad,b,kw=load('balanced');kw[keyword]='ABC'
    with pytest.raises(ValueError,match='sequences'):
        run(b,ad,cell_type_key='ct',sample_key='donor',**kw)


def test_near_singular_variance_is_rejected_not_clipped():
    rng=np.random.default_rng(12);x=rng.uniform(1,2,20);z=rng.normal(size=20)
    y=rng.uniform(.01,.1,20);y/=y.sum();D=np.stack([x,x+1e-9*z],axis=1)
    with pytest.raises(ValueError,match='numerically singular|variance'):
        module._music_fit(y,D,np.ones(2),np.zeros_like(D),max_iter=20,nu=1e-4,eps=.01)


def test_under_supported_type_is_reported_not_fabricated():
    ad,b,kw=load('balanced')
    ad=ad[~((ad.obs.ct=='C')&(ad.obs.donor!='d1'))].copy()
    out=run(b,ad,cell_type_key='ct',sample_key='donor',**kw)
    assert list(out['proportions'].columns)==['A','B']
    assert out['provenance']['excluded_cell_types']==['C']
    ad.obs['group']=np.where(ad.obs.ct=='B','G2','G1')
    with pytest.raises(ValueError,match='cross-donor support'):
        module.music_deconvolution_recursive(b,ad,group_key='group',cell_type_key='ct',sample_key='donor',group_cell_types={'G1':['A','C'],'G2':['B']},group_markers={'G1':['g0','g1','g2'],'G2':['g3']})
