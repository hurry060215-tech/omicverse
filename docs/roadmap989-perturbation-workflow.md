# Perturbation workflow extensions



```python
joined = ov.single.prepare_perturbation_data(
    rna, guide_counts, guide_to_target={'NT1':'control', 'gA':'GENE_A'},
    control_guides=['NT1'], assignment_threshold=5, multiple_guides='exclude')
eligible = joined.obs['perturbation_target'].notna()
classified = ov.single.mixscape(joined[eligible].copy(),
    pert_key='perturbation_target', control='control', split_by='donor',
    layer='lognorm')
summary = ov.single.perturbation_sample_summary(classified,
    sample_key='donor', perturbation_key='perturbation_target',
    classification_key='mixscape_class_global')
```

Guide modality and RNA are aligned by unique cell identifiers. Missing modality,
no guide above threshold, and multiple guides are retained as explicit statuses,
not converted into controls. Mapping is explicit, and ties/multiplicity have an
explicit policy. `multiple_guides='max'` opts into the native maximum-count rule.
RNA counts and row order are retained. Mixscape remains a cell-level classifier;
replicate summaries do not become a hypothesis test.

```python
scores = ov.single.perturbation_baseline(classified, sample_key='donor',
    perturbation_key='perturbation_target', control='control', layer='lognorm',
    train_samples=['D1','D2'], test_samples=['D3'], strategy='condition_mean')
```

Both control-mean and condition-mean baselines evaluate sample-condition means.
With `donor_key`, they first average technical samples within donor, then weight
training donors equally; shared donors across splits are rejected. Without it,
each sample is assumed to be an independent biological replicate. Overlapping
sample splits or missing controls always raise. This is not Systema and not a claim of validated
cross-dataset prediction. Preprocessing learned on the entire dataset can still
leak information; callers must fit transformations on training data only.


## Real perturbation acceptance

`examples/roadmap989/perturbation_papalexi.py` ran on the official pertpy
Papalexi dataset: all 20,729 RNA cells and 111 guide features were joined before
explicit CUL3/control selection (2,262 cells). No random cell thinning was used.
HVG selection used rep1/rep2 only; rep3 was held out for descriptive baselines.
These are THP-1 experimental replicates, not independent human donors.
Native pertpy 1.4.0 rerun reproduced the saved signature exactly and all global
class labels exactly. H5AD changes object labels to categorical storage; comparison
checks identifier order and label values rather than pandas dtype equality.
Raw counts and H5AD roundtrip were verified. Systema, calibrated differential
inference and clinical/generalization claims remain outside this core acceptance.

This separate draft includes the unchanged adapter prerequisite from PR #1016. No existing PR branch was modified. Numerical and real-data results were run 2026-10-31, not re-created by this publication step.
