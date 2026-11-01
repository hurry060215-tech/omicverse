"""Cell-ID and biological-replicate aware perturbation workflow utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .._backend_validation import validate_adata, expression, obs_column
from .._registry import register_function


@register_function(aliases=["guide RNA alignment"], category="single",
                   description="Align guide and RNA modalities by cell ID with explicit assignment and exclusion policy")
def prepare_perturbation_data(rna, guides, *, guide_to_target, control_guides,
                              assignment_threshold, guide_layer=None,
                              multiple_guides="exclude", prefix="perturbation"):
    """Return an RNA copy with guide/target/status columns, preserving RNA order.

    Guide assignment uses native pertpy's highest-count rule. Cells absent from
    the guide modality, below threshold, or carrying multiple above-threshold
    guides are marked explicitly. ``multiple_guides='max'`` is opt-in and retains
    the multiplicity flag. Every guide must have a target mapping; control guides
    map to the literal 'control'. Excluded cells are retained with missing target,
    not mislabeled as controls. No backend classification is run on this helper.
    """
    from ._perturbation import assign_guides
    validate_adata(rna)
    counts = expression(guides, guide_layer, counts=True)
    if sparse.issparse(counts):
        counts = sparse.csr_matrix(counts, copy=True)
        counts.sum_duplicates()
        counts.sort_indices()
        from .._backend_validation import validate_matrix
        validate_matrix(counts, "Canonical guide counts", counts=True)
    if not isinstance(prefix,str) or not prefix or "/" in prefix:
        raise ValueError("prefix must be a nonempty storage-safe string without slashes.")
    if multiple_guides not in {"exclude","max"}:
        raise ValueError("multiple_guides must be 'exclude' or 'max'.")
    if isinstance(assignment_threshold,bool) or not np.isscalar(assignment_threshold) or not np.isfinite(assignment_threshold) or assignment_threshold <= 0:
        raise ValueError("assignment_threshold must be positive and finite.")
    mapping = dict(guide_to_target)
    if set(guides.var_names) - set(mapping):
        raise ValueError("Every guide must have an explicit guide_to_target entry.")
    if any(not isinstance(v,str) or not v for v in mapping.values()):
        raise ValueError("Guide targets must be nonempty strings.")
    controls = set(control_guides)
    if not controls or not controls.issubset(guides.var_names):
        raise ValueError("control_guides must name one or more guides in the guide modality.")
    if any(mapping[g] != "control" for g in controls) or any(mapping[g] == "control" for g in guides.var_names if g not in controls):
        raise ValueError("Exactly the declared control_guides must map to 'control'.")
    columns = [f"{prefix}_{s}" for s in ['guide','target','status','n_guides']]
    if any(c in rna.obs for c in columns) or prefix in rna.uns:
        raise ValueError("Output prefix already exists; choose another prefix.")
    work = guides.copy()
    # Native GuideAssignment dispatch supports dense and CSR, not every SciPy layout.
    if sparse.issparse(counts):
        if guide_layer is None:
            work.X = sparse.csr_matrix(counts)
        else:
            work.layers[guide_layer] = sparse.csr_matrix(counts)
    # A fresh temporary key avoids colliding with an existing guide annotation.
    key = '__ov_workflow_assigned_guide'
    if key in work.obs:
        del work.obs[key]
    labels = assign_guides(work, assignment_threshold=assignment_threshold,
                           layer=guide_layer, key_added=key)
    n_guides = np.asarray((counts >= assignment_threshold).sum(axis=1)).ravel().astype(int)
    assignments = pd.DataFrame({'guide':labels.astype(str), 'n_guides':n_guides}, index=guides.obs_names)
    aligned = assignments.reindex(rna.obs_names)
    missing = aligned['guide'].isna()
    status = pd.Series('assigned',index=rna.obs_names,dtype=object)
    status[missing] = 'missing_guide_modality'
    status[~missing & (aligned.n_guides == 0)] = 'unassigned'
    multiple = ~missing & (aligned.n_guides > 1)
    status[multiple] = 'multiple_guides' if multiple_guides == 'exclude' else 'assigned_multiple_max'
    target = aligned.guide.map(mapping)
    eligible = status.isin(['assigned','assigned_multiple_max'])
    target[~eligible] = np.nan
    if target[eligible].isna().any():
        raise RuntimeError("Native guide assignment returned an unknown guide identifier.")
    result=rna.copy()
    result.obs[columns[0]]=pd.Categorical(aligned.guide)
    result.obs[columns[1]]=pd.Categorical(target)
    result.obs[columns[2]]=pd.Categorical(status)
    result.obs[columns[3]]=aligned.n_guides.fillna(-1).astype(int) # -1 means unavailable, not zero
    result.uns[prefix]={'assignment_threshold':float(assignment_threshold),
                       'multiple_guides':multiple_guides,'control_guides':sorted(controls),
                       'guide_to_target':{'guide_ids':[str(k) for k in mapping], 'targets':[str(v) for v in mapping.values()]},
                       'rna_cells':rna.n_obs,'guide_only_cells':int((~guides.obs_names.isin(rna.obs_names)).sum()),
                       'status_counts':{str(k):int(v) for k,v in status.value_counts().items()},
                       'scope':'guide_assignment_not_perturbation_efficacy'}
    return result


@register_function(aliases=["replicate perturbation summary"], category="single",
                   description="Summarize perturbation classifications per biological sample")
def perturbation_sample_summary(adata, *, sample_key, perturbation_key, classification_key):
    """Return counts/fractions per observed sample×perturbation×class combination.

    Missing sample IDs are rejected. Missing perturbation/classes are reported
    as '<missing>', not silently removed. No p-values or efficacy claims added.
    """
    validate_adata(adata)
    obs_column(adata,sample_key)
    for key in [perturbation_key,classification_key]:
        if key not in adata.obs:
            raise KeyError(key)
    frame=adata.obs[[sample_key,perturbation_key,classification_key]].copy()
    for key in frame:
        if frame[key].isna().any() and (frame[key].astype(object) == '<missing>').any():
            raise ValueError("The '<missing>' label collides with a real value; rename it before summarizing.")
        frame[key]=frame[key].astype(object).where(frame[key].notna(),'<missing>')
    result=frame.groupby(list(frame.columns),sort=False,dropna=False,observed=True).size().rename('n_cells').reset_index()
    totals=result.groupby([sample_key,perturbation_key],sort=False,observed=True)['n_cells'].transform('sum')
    result['fraction']=result.n_cells/totals
    return result


@register_function(aliases=["donor heldout perturbation baseline"], category="single",
                   description="Evaluate simple perturbation means on disjoint biological samples")
def perturbation_baseline(adata, *, sample_key, perturbation_key, control,
                          train_samples, test_samples, layer=None, strategy="control_mean", donor_key=None):
    """Evaluate donor-balanced training means against each held-out sample.

    Input should be consistently normalized/log-transformed expression. Samples,
    never random cells, define the split. Use ``donor_key`` when several samples/libraries belong to one donor; a donor
    cannot occur on both sides. Training sample means are first averaged within
    donor, then equally across donors. Without donor_key each sample is assumed
    to be an independent biological donor. ``control_mean`` predicts the training control
    mean for every target; ``condition_mean`` uses its training condition mean
    and rejects unseen targets. Returned errors describe this supplied split;
    they are not a significance test or independent generalization guarantee.
    """
    values=expression(adata,layer)
    samples=obs_column(adata,sample_key)
    labels=obs_column(adata,perturbation_key)
    donors=samples if donor_key is None else obs_column(adata,donor_key)
    sample_donor={}
    for sample in samples.unique():
        unique=donors[samples==sample].unique()
        if len(unique)!=1:
            raise ValueError("Each sample must map to exactly one biological donor.")
        sample_donor[sample]=unique[0]
    train,test=list(train_samples),list(test_samples)
    if not train or not test or len(set(train))!=len(train) or len(set(test))!=len(test) or set(train)&set(test):
        raise ValueError("Training and test sample lists must be nonempty, unique and disjoint.")
    if not (set(train)|set(test)).issubset(set(samples)):
        raise ValueError("Every requested sample must occur in the data.")
    if {sample_donor[s] for s in train} & {sample_donor[s] for s in test}:
        raise ValueError("Training and test samples must have disjoint biological donors.")
    if strategy not in {'control_mean','condition_mean'}:
        raise ValueError("Unknown baseline strategy.")
    if control not in set(labels):
        raise ValueError("control is absent.")
    # Preserve explicit replicate design: no donor may lack the control.
    for sample in train+test:
        if not ((samples==sample)&(labels==control)).any():
            raise ValueError("Every training and test sample must contain control cells.")
    means={}
    for sample in train+test:
        for label in labels[samples==sample].unique():
            mask=np.asarray((samples==sample)&(labels==label))
            means[sample,label]=np.asarray(values[mask].mean(axis=0)).ravel()
    training={}
    for label in labels.unique():
        donor_means=[]
        for donor in dict.fromkeys(sample_donor[s] for s in train):
            candidates=[means[s,label] for s in train if sample_donor[s]==donor and (s,label) in means]
            if candidates:
                donor_means.append(np.mean(candidates,axis=0))
        if donor_means:
            training[label]=np.mean(donor_means,axis=0)
    rows=[]
    for (sample,label),actual in means.items():
        if sample not in test:
            continue
        key=control if strategy=='control_mean' else label
        if key not in training:
            raise ValueError(f"No training observations for test condition {label!r}.")
        error=training[key]-actual
        rows.append({'sample':sample,'perturbation':label,'n_cells':int(((samples==sample)&(labels==label)).sum()),
                     'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(np.abs(error))),
                     'strategy':strategy,'n_training_samples':sum((s,key) in means for s in train),
                     'n_training_donors':len({sample_donor[s] for s in train if (s,key) in means})})
    return pd.DataFrame(rows)
