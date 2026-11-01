"""Papalexi real-data core workflow; explicit target subset, no random cell thinning.

Input: official pertpy papalexi_2021.h5mu, not bundled.
This is computational acceptance on THP-1 experimental replicates, not donor
clinical generalization. Control guides are NT; gene target parsing is dataset
specific and recorded. Learning feature selection uses training replicates only.
"""
from pathlib import Path
import argparse, re, json, hashlib, importlib.metadata
import numpy as np
import pandas as pd
import scanpy as sc
import mudata as md
import anndata as ad
from scipy import sparse
from omicverse.single import (prepare_perturbation_data, mixscape,
    perturbation_sample_summary,perturbation_baseline,perturbation_distance)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--target',default='CUL3');p.add_argument('--threshold',type=float,default=10)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): p.error('Output must be new/empty.')
    a.output.mkdir(parents=True,exist_ok=True)
    m=md.read_h5mu(a.input)
    rna=m.mod['rna'].copy();guides=m.mod['gdo']
    assert rna.obs_names.is_unique and guides.obs_names.is_unique
    rna.obs['replicate']=m.obs['replicate'].reindex(rna.obs_names)
    mapping={g:re.sub(r'g\d+$','',g) for g in guides.var_names}
    controls=[g for g,t in mapping.items() if t=='NT']
    mapping.update({g:'control' for g in controls})
    joined=prepare_perturbation_data(rna,guides,guide_to_target=mapping,control_guides=controls,
                                    assignment_threshold=a.threshold,multiple_guides='exclude')
    joined.obs.to_csv(a.output/'all_cell_assignments.csv')
    qc=joined.uns['perturbation']
    (a.output/'assignment_qc.json').write_text(json.dumps(qc,indent=2)+'\n')
    selected=joined[joined.obs.perturbation_target.isin(['control',a.target])].copy()
    if not selected.n_obs: raise ValueError('No eligible control/target cells.')
    selected.layers['counts']=selected.X.copy()
    sc.pp.normalize_total(selected,target_sum=1e4);sc.pp.log1p(selected)
    train=selected[selected.obs.replicate.isin(['rep1','rep2'])].copy()
    sc.pp.highly_variable_genes(train,n_top_genes=min(2000,train.n_vars),flavor='seurat')
    selected=selected[:,train.var.highly_variable.to_numpy()].copy()
    counts_before=selected.layers['counts'].copy()
    np.random.seed(0)
    fitted=mixscape(selected,pert_key='perturbation_target',control='control',split_by='replicate',
                    signature_kwargs={'ref_selection_mode':'split_by'},
                    mixscape_kwargs={'min_de_genes':5,'iter_num':3,'test_method':'t-test'})
    if sparse.issparse(counts_before): assert (fitted.layers['counts']!=counts_before).nnz==0
    else: np.testing.assert_array_equal(fitted.layers['counts'],counts_before)
    summary=perturbation_sample_summary(fitted,sample_key='replicate',perturbation_key='perturbation_target',
                                        classification_key='mixscape_class_global')
    summary.to_csv(a.output/'sample_summary.csv',index=False)
    for strategy in ['control_mean','condition_mean']:
        result=perturbation_baseline(selected,sample_key='replicate',perturbation_key='perturbation_target',
             control='control',train_samples=['rep1','rep2'],test_samples=['rep3'],strategy=strategy)
        result.to_csv(a.output/f'{strategy}.csv',index=False)
    # Distances per experimental replicate, not a pooled-cell inferential test.
    distances=[]
    for rep in fitted.obs.replicate.unique():
        sub=fitted[fitted.obs.replicate==rep].copy()
        if sub.obs.perturbation_target.nunique()<2: continue
        sc.pp.pca(sub,n_comps=min(20,sub.n_obs-1,sub.n_vars-1),random_state=0)
        d=perturbation_distance(sub,groupby='perturbation_target',pairwise_kwargs={'show_progressbar':False,'n_jobs':1})
        distances.append({'replicate':str(rep),'edistance':float(d.loc['control',a.target])})
    pd.DataFrame(distances).to_csv(a.output/'descriptive_distances.csv',index=False)
    fitted.write_h5ad(a.output/'result.h5ad',compression='gzip')
    restored=ad.read_h5ad(a.output/'result.h5ad')
    assert restored.obs_names.equals(fitted.obs_names)
    assert restored.var_names.equals(fitted.var_names)
    if sparse.issparse(fitted.layers['X_pert']):
        assert (restored.layers['X_pert']!=fitted.layers['X_pert']).nnz==0
    else:
        np.testing.assert_array_equal(restored.layers['X_pert'],fitted.layers['X_pert'])
    receipt={'source':'https://exampledata.scverse.org/pertpy/papalexi_2021.h5mu',
      'sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'all_rna_cells':rna.n_obs,
      'all_guide_features':guides.n_vars,'selected_target':a.target,'selected_shape':list(selected.shape),
      'replicate_counts':{str(k):int(v) for k,v in selected.obs.replicate.value_counts().items()},
      'threshold':a.threshold,'training_replicates':['rep1','rep2'],'heldout_replicate':'rep3',
      'counts_preserved':True,'save_reload_verified':True,
      'versions':{name:importlib.metadata.version(name) for name in ['pertpy','anndata','scanpy','numpy']},
      'scope':'core computational workflow only; no Systema, calibrated inference or donor generalization'}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
