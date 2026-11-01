"""Explicit real-data moscot workflows; inputs are not bundled or silently subsampled.

Temporal example used in verification:
  python examples/roadmap989/moscot_workflows.py temporal data/moscot/wot_temporal.h5ad out/ --group day
Spatial examples require authentic coordinate-bearing files supplied by the user:
  python examples/roadmap989/moscot_workflows.py mapping sc.h5ad out/ --spatial sp.h5ad
  python examples/roadmap989/moscot_workflows.py alignment slices.h5ad out/ --group slice --reference sliceA

No trusted-pickle loading. Dense exports have an explicit size bound.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import anndata as ad
import numpy as np
import scanpy as sc
from omicverse._moscot import moscot_temporal, moscot_mapping, moscot_alignment
from omicverse._moscot_results import moscot_coupling, moscot_impute, moscot_align_coordinates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['temporal','mapping','alignment'])
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--spatial',type=Path)
    parser.add_argument('--group',default='day')
    parser.add_argument('--reference')
    parser.add_argument('--epsilon',type=float,default=.5)
    parser.add_argument('--max-entries',type=int,default=10_000_000)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be absent or empty; existing results are not overwritten.')
    args.output.mkdir(parents=True,exist_ok=True)
    a=ad.read_h5ad(args.input)
    # Preprocessed input is required. No normalization is guessed from X.
    source_hash=hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.mode=='temporal':
        # Ignore upstream fixtures' precomputed results and artificial batch labels.
        a.uns.clear()
        sc.pp.pca(a,n_comps=min(30,a.n_obs-1,a.n_vars-1),random_state=0)
        p=moscot_temporal(a,time_key=args.group,joint_attr='X_pca',
                          solve_kwargs={'epsilon':args.epsilon,'threshold':1e-5,'max_iterations':1000})
        a.write_h5ad(args.output/'input_preprocessed.h5ad')
    elif args.mode=='mapping':
        if args.spatial is None:
            parser.error('--spatial is required for mapping.')
        sp=ad.read_h5ad(args.spatial)
        p=moscot_mapping(a,sp,prepare_kwargs={'joint_attr':{'attr':'X'}},
                         solve_kwargs={'alpha':0,'epsilon':args.epsilon,'threshold':1e-5})
        pred=moscot_impute(p,max_entries=args.max_entries)
        pred.write_h5ad(args.output/'imputed.h5ad')
        reread=ad.read_h5ad(args.output/'imputed.h5ad')
        np.testing.assert_array_equal(pred.X,reread.X)
        assert pred.obs_names.equals(reread.obs_names)
    else:
        if args.reference is None:
            parser.error('--reference is required for alignment.')
        sc.pp.pca(a,n_comps=min(30,a.n_obs-1,a.n_vars-1),random_state=0)
        p=moscot_alignment(a,batch_key=args.group,joint_attr='X_pca',
                           solve_kwargs={'alpha':.5,'epsilon':args.epsilon})
        aligned=moscot_align_coordinates(p,reference=args.reference)
        aligned.write_h5ad(args.output/'aligned.h5ad')
        np.testing.assert_array_equal(ad.read_h5ad(args.output/'aligned.h5ad').obsm['spatial_aligned'],
                                      aligned.obsm['spatial_aligned'])
    results=[]
    for i,key in enumerate(p.solutions):
        coupling=moscot_coupling(p,key,max_entries=args.max_entries)
        path=args.output/f'coupling_{i}.h5ad'
        coupling.write_h5ad(path)
        reread=ad.read_h5ad(path)
        np.testing.assert_array_equal(coupling.X,reread.X)
        assert coupling.obs_names.equals(reread.obs_names) and coupling.var_names.equals(reread.var_names)
        results.append({'key':[str(k) for k in key],'shape':list(coupling.shape),
                        'mass':float(coupling.X.sum()),'converged':bool(p.solutions[key].converged)})
    receipt={'mode':args.mode,'input_sha256':source_hash,'input_shape':list(a.shape),'solutions':results,
             'versions':{n:importlib.metadata.version(n) for n in ['moscot','anndata','scanpy','numpy','jax','ott-jax']},
             'scope':'computational execution and identity preservation, not lineage/registration biological truth'}
    if args.spatial:
        receipt['spatial_input_sha256']=hashlib.sha256(args.spatial.read_bytes()).hexdigest()
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
