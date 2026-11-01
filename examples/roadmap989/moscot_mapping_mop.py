"""Official Tangram MOp data with explicit zero-training-total exclusions; native moscot adapter acceptance, not Tangram parity."""
import os
os.environ['OMICVERSE_DISABLE_ANALYTICS']='1'
from pathlib import Path
import json,hashlib,time,importlib.metadata,gc,argparse
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse,stats
from moscot.problems.space import MappingProblem
from omicverse._moscot import moscot_mapping
from omicverse._moscot_results import moscot_impute,moscot_coupling
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('data',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--alpha',type=float,default=0);args=parser.parse_args()
P=args.data;O=args.output
if O.exists() and any(O.iterdir()):parser.error('Output must be absent or empty; existing evidence will not be overwritten.')
O.mkdir(parents=True,exist_ok=True);t=time.time()
ref0=ad.read_h5ad(P/'mop_sn_tutorial.h5ad',backed='r');sp0=ad.read_h5ad(P/'slideseq_MOp_1217.h5ad',backed='r')
markers=sorted(set(pd.read_csv(P/'MOp_markers.csv',index_col=0).values.ravel()) & set(ref0.var_names) & set(sp0.var_names))
original_shapes=[list(ref0.shape),list(sp0.shape)]
held=markers[::5];train=[g for g in markers if g not in held]
exclusions={}
def extract(a,label):
 x=a[:,markers].X
 x=x.toarray() if sparse.issparse(x) else np.asarray(x)
 assert np.isfinite(x).all() and (x>=0).all() and np.equal(x,np.floor(x)).all(), 'Selected source marker values must be finite nonnegative integer counts'
 out=ad.AnnData(x.astype(np.float32),obs=a.obs.copy(),var=pd.DataFrame(index=markers))
 denom=out[:,train].X.sum(axis=1)
 keep=denom>0
 exclusions[label]=out.obs_names[~keep].tolist()
 out=out[keep].copy();denom=denom[keep]
 out.X=np.log1p(out.X/denom[:,None]*10000).astype(np.float32)
 return out
ref=extract(ref0,'reference');sp=extract(sp0,'spatial'); ref0.file.close();sp0.file.close();del ref0,sp0;gc.collect()
sp.obsm['spatial']=sp.obs[['x','y']].to_numpy()
prepare={'joint_attr':{'attr':'X'},'var_names':train}
solve={'alpha':0,'epsilon':.05,'threshold':1e-5,'max_iterations':1000,'batch_size':512}
sc_attr=None
if args.alpha:
 from sklearn.decomposition import PCA
 ref.obsm['X_train_pca']=PCA(n_components=30,svd_solver='full').fit_transform(ref[:,train].X)
 sc_attr='X_train_pca';solve.update(alpha=args.alpha,rank=64,epsilon=0)
print('DATA',ref.shape,sp.shape,len(train),len(held),flush=True)
native=MappingProblem(adata_sc=ref,adata_sp=sp).prepare(sc_attr=sc_attr,spatial_key='spatial',batch_key=None,**prepare).solve(**solve)
if not all(bool(s.converged) for s in native.solutions.values()):
 failure={'status':'failed_native_nonconvergence','native_converged':False,'ov_solve_started':False,'solve':solve,'elapsed_seconds':time.time()-t,'reference_shape':list(ref.shape),'spatial_shape':list(sp.shape),'scope':'No prediction or OV parity result accepted.'}
 (O/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
 raise RuntimeError('Native mapping did not converge; failure receipt written, OV comparison not started.')
print('NATIVE SOLVED',time.time()-t,flush=True)
expected=native.impute(var_names=held,batch_size=len(held));expected.write_h5ad(O/'native-heldout.h5ad')
native_marginals={str(k):{'a_sum':float(np.asarray(v.a).sum()),'b_sum':float(np.asarray(v.b).sum())} for k,v in native.solutions.items()}
del native;gc.collect()
p=moscot_mapping(ref,sp,sc_attr=sc_attr,prepare_kwargs=prepare,solve_kwargs=solve)
print('OV SOLVED',time.time()-t,flush=True)
pred=moscot_impute(p,var_names=held,batch_size=len(held))
assert expected.obs_names.equals(pred.obs_names) and expected.var_names.equals(pred.var_names)
np.testing.assert_allclose(pred.X,expected.X,atol=1e-6,rtol=1e-5)
assert pred.obs_names.equals(sp.obs_names) and pred.var_names.tolist()==held
pred.write_h5ad(O/'ov-heldout.h5ad'); reread=ad.read_h5ad(O/'ov-heldout.h5ad')
np.testing.assert_array_equal(pred.X,reread.X);assert pred.obs_names.equals(reread.obs_names) and pred.var_names.equals(reread.var_names)
np.testing.assert_array_equal(pred.obsm['spatial'],reread.obsm['spatial'])
try:moscot_coupling(p,next(iter(p.solutions)),max_entries=10_000_000)
except ValueError: dense_guard=True
else:raise AssertionError('dense guard failed')
correlations={g:float(stats.pearsonr(pred.X[:,i],sp[:,g].X.ravel()).statistic) for i,g in enumerate(held)}
receipt={'status':'pass','excluded_zero_training_total':exclusions,'source':'https://tangram-sc.readthedocs.io/en/latest/tutorial_link.html','scope':'All supplied nuclei and spatial spots with positive training-marker totals; alpha=0 is expression-only, alpha>0 combines spatial geometry and training-marker PCA; not Tangram parity. Heldout correlation descriptive, not biological acceptance threshold. No full dense coupling materialized.','reference_shape':list(ref.shape),'spatial_shape':list(sp.shape),'source_original_shapes':original_shapes,'training_genes':train,'heldout_genes':held,'heldout_rule':'sorted shared curated markers, every fifth held out before normalization and fitting','normalization':'log1p(10000 * marker counts / training-marker total); heldout genes excluded from normalizer and fitting','selected_marker_counts_integer_nonnegative_verified':True,'pca':({'n_components':30,'svd_solver':'full','input':'training markers only'} if args.alpha else None),'initializer':'native moscot/ott defaults; identical for native and OV','rank_scope':('rank64 approximation, not full-rank parity' if args.alpha else 'full-rank batched linear OT'),'solve':solve,'native_vs_ov_max_abs':float(np.max(abs(pred.X-expected.X))),'comparison_tolerance':{'atol':1e-6,'rtol':1e-5},'converged':True,'h5ad_roundtrip':True,'dense_coupling_guard':dense_guard,'native_marginals':native_marginals,'heldout_pearson':correlations,'median_heldout_pearson':float(np.nanmedian(list(correlations.values()))),'seconds':time.time()-t,'versions':{n:importlib.metadata.version(n) for n in ['moscot','anndata','numpy','jax','ott-jax']}}
receipt['sha256']={n:hashlib.file_digest(open(P/n,'rb'),'sha256').hexdigest() for n in ['mop_sn_tutorial.h5ad','slideseq_MOp_1217.h5ad','mop_sn_tutorial.h5ad.gz','slideseq_MOp_1217.h5ad.gz','MOp_markers.csv']}
receipt['source_urls']={n:'https://storage.googleapis.com/tommaso-brain-data/tangram_demo/'+n for n in ['mop_sn_tutorial.h5ad.gz','slideseq_MOp_1217.h5ad.gz','MOp_markers.csv']}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
