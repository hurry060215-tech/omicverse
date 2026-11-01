"""Native parity on full author-provided PASTE breast slices, no cell subsampling.

Download the four CSVs from the pinned source recorded in source.json separately.
This tests software/native agreement, not anatomical registration ground truth.
"""
import argparse,json,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from moscot.problems.space import AlignmentProblem
from omicverse.space import moscot_alignment,moscot_align_coordinates,moscot_coupling

EXPECTED_SHA256 = {'slice1.csv': 'f7e8e1b37be579b66b63f1541055c1df5380e87e993d8cbc4e6fda457dc16002', 'slice1_coor.csv': '77647774caeb6b03bc481a76d97e1b6117d722a8ddfcf192625cc30a8144ed88', 'slice2.csv': 'c6c44953348db76aba7120d189834a6c55cb5fa4351c83b6a6b88d6b4b4ccadc', 'slice2_coor.csv': '5eb5f4ca0ccd274a1b30caec877ae02b10cee2f1a21ed0c218d254a42dfa53d4'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.output.exists() and any(x.name != 'source.json' for x in a.output.iterdir()):
  p.error('Output must be new or contain only source.json; existing results are not overwritten.')
 a.output.mkdir(parents=True,exist_ok=True)
 for name,expected_hash in EXPECTED_SHA256.items():
  if hashlib.sha256((a.input/name).read_bytes()).hexdigest()!=expected_hash:
   raise ValueError(f'Input checksum does not match pinned author dataset: {name}')
 pieces={};raw_shapes={}
 for name in ['slice1','slice2']:
  d=sc.read_csv(a.input/f'{name}.csv');d.obsm['spatial']=np.genfromtxt(a.input/f'{name}_coor.csv',delimiter=',')
  raw_shapes[name]=list(d.shape)
  # Same explicit QC as author PASTE quick-start; no random selection.
  sc.pp.filter_genes(d,min_counts=15);sc.pp.filter_cells(d,min_counts=100)
  pieces[name]=d
 d=ad.concat(pieces,join='inner',label='slice',index_unique=':');d.layers['counts']=d.X.copy()
 sc.pp.normalize_total(d,target_sum=1e4);sc.pp.log1p(d);sc.pp.pca(d,n_comps=30,random_state=0)
 before=d.copy();kwargs=dict(alpha=.5,epsilon=.05,threshold=1e-5,max_iterations=500)
 print('Input:',raw_shapes,'after QC/shared genes:',d.shape,flush=True)
 native=AlignmentProblem(d.copy()).prepare(batch_key='slice',spatial_key='spatial',joint_attr='X_pca').solve(**kwargs)
 wrapped=moscot_alignment(d,batch_key='slice',joint_attr='X_pca',solve_kwargs=kwargs)
 receipt={'raw_shapes':raw_shapes,'processed_shape':list(d.shape),'qc':{'min_gene_counts':15,'min_cell_counts':100,'join':'shared_genes','normalization_target':10000,'pca_components':30,'random_seed':0},'solve_kwargs':kwargs,'couplings':[],'alignments':[]}
 for i,key in enumerate(wrapped.solutions):
  sol=wrapped.solutions[key];nsol=native.solutions[key]
  assert sol.converged and nsol.converged
  c=moscot_coupling(wrapped,key);expected=np.asarray(nsol.transport_matrix)
  np.testing.assert_allclose(c.X,expected,atol=1e-6,rtol=1e-6)
  assert np.isfinite(c.X).all() and (c.X>=0).all()
  c.write_h5ad(a.output/f'coupling{i}.h5ad',compression='gzip');r=ad.read_h5ad(a.output/f'coupling{i}.h5ad')
  np.testing.assert_array_equal(r.X,c.X);assert r.obs_names.equals(c.obs_names) and r.var_names.equals(c.var_names)
  receipt['couplings'].append({'key':[str(x) for x in key],'shape':list(c.shape),'converged':True,'mass':float(c.X.sum()),'native_max_abs':float(np.max(np.abs(c.X-expected)))})
 for mode in ['warp','affine']:
  expected,_=native.align(reference='slice1',mode=mode)
  result=moscot_align_coordinates(wrapped,reference='slice1',mode=mode,key_added=f'spatial_{mode}')
  np.testing.assert_allclose(result.obsm[f'spatial_{mode}'],expected,atol=1e-5,rtol=1e-5)
  np.testing.assert_array_equal(result.obsm['spatial'],before.obsm['spatial'])
  result.write_h5ad(a.output/f'aligned_{mode}.h5ad',compression='gzip');r=ad.read_h5ad(a.output/f'aligned_{mode}.h5ad')
  np.testing.assert_array_equal(r.obsm[f'spatial_{mode}'],result.obsm[f'spatial_{mode}']);assert r.obs_names.equals(result.obs_names)
  receipt['alignments'].append({'mode':mode,'native_max_abs':float(np.max(np.abs(result.obsm[f'spatial_{mode}']-expected))),'save_reload':True})
 np.testing.assert_array_equal(d.X,before.X);np.testing.assert_array_equal(d.obsm['spatial'],before.obsm['spatial']);assert 'spatial_warp' not in d.obsm
 receipt.update(expression_and_physical_coordinates_unchanged=True,coordinate_identity='author-provided paired CSV row order; coordinates have no independent barcode column',input_sha256=EXPECTED_SHA256,scope='Full two author-provided real slices; native numerical/software acceptance, not anatomical ground truth',versions={n:importlib.metadata.version(n) for n in ['moscot','anndata','scanpy','numpy','jax','ott-jax']})
 (a.output/'parity.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
