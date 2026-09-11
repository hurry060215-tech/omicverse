import importlib.util

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from omicverse.external.flowsig.tools import (
    _network, apply_biological_flow, filter_low_confidence_edges,
)


class _Unpicklable:
    def __reduce__(self):
        raise TypeError("intentional non-picklable test object")


def _fake_gsp(samples, _alpha):
    return np.eye(samples.shape[1], dtype=float)


def _fake_utigsp(control, _perturbed, _alpha, _alpha_inv):
    size = control.shape[1]
    return np.eye(size, dtype=float), [{0, size - 1}]


def _make_adata(n_obs=24, n_vars=3):
    rng = np.random.default_rng(42)
    adata = ad.AnnData(np.ones((n_obs, 2)))
    adata.obsm["X_flow"] = rng.normal(size=(n_obs, n_vars))
    adata.obs["block"] = pd.Categorical(np.repeat(["a", "b"], n_obs // 2))
    adata.uns["flowsig_network"] = {
        "flow_var_info": pd.DataFrame(index=[f"flow_{i}" for i in range(n_vars)])
    }
    return adata


def test_indices_by_block_and_resampling_stay_within_blocks():
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    blocks = _network._indices_by_block(labels)
    samples = np.column_stack([np.arange(6), np.array([0, 0, 0, 1, 1, 1])])

    resampled = _network._resample_matrix(
        samples, np.random.default_rng(7), indices_by_blocks=blocks
    )

    assert [block.tolist() for block in blocks] == [[0, 1, 2], [3, 4, 5]]
    np.testing.assert_array_equal(resampled[:3, 1], 0)
    np.testing.assert_array_equal(resampled[3:, 1], 1)


def test_resampling_is_seeded_without_global_rng_state():
    samples = np.arange(80).reshape(40, 2)
    first = _network._resample_matrix(samples, np.random.default_rng(11))
    repeated = _network._resample_matrix(samples, np.random.default_rng(11))
    different = _network._resample_matrix(samples, np.random.default_rng(12))

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize("use_spatial", [False, True])
def test_run_gsp_supports_spatial_and_nonspatial_inputs(monkeypatch, use_spatial):
    monkeypatch.setattr(_network, "_learn_gsp", _fake_gsp)
    samples = np.column_stack(
        [np.arange(12), np.arange(12) ** 2, np.ones(12)]
    ).astype(float)
    blocks = _network._indices_by_block(np.repeat([0, 1], 6))

    result = _network.run_gsp(
        samples,
        use_spatial=use_spatial,
        indices_by_blocks=blocks if use_spatial else None,
        seed=3,
    )

    np.testing.assert_array_equal(result["nonzero_flow_vars_indices"], [0, 1])
    np.testing.assert_array_equal(result["adjacency_cpdag"], np.eye(2))


@pytest.mark.parametrize("use_spatial", [False, True])
def test_run_utigsp_supports_spatial_and_nonspatial_inputs(monkeypatch, use_spatial):
    monkeypatch.setattr(_network, "_learn_utigsp", _fake_utigsp)
    control = np.column_stack(
        [np.arange(12), np.arange(12) ** 2, np.ones(12)]
    ).astype(float)
    perturbed = [
        np.column_stack(
            [np.arange(12) + 1, (np.arange(12) + 1) ** 2, np.ones(12)]
        ).astype(float)
    ]
    blocks = _network._indices_by_block(np.repeat([0, 1], 6))

    result = _network.run_utigsp(
        control,
        perturbed,
        use_spatial=use_spatial,
        indices_by_blocks_control=blocks if use_spatial else None,
        indices_by_blocks_perturbed=[blocks] if use_spatial else None,
        seed=4,
    )

    np.testing.assert_array_equal(result["nonzero_flow_vars_indices"], [0, 1])
    np.testing.assert_array_equal(result["adjacency_cpdag"], np.eye(2))
    np.testing.assert_array_equal(result["perturbed_targets_indices"][0], [0, 1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_bootstraps": 0}, "n_bootstraps"),
        ({"flow_expr_key": "missing"}, "flow_expr_key"),
        ({"use_spatial": True}, "block_key"),
        ({"use_spatial": True, "block_key": "missing"}, "block_key"),
        ({"condition_key": "missing", "control_key": "control"}, "condition_key"),
        ({"condition_key": "condition"}, "control_key"),
        (
            {"condition_key": "condition", "control_key": "missing"},
            "control_key",
        ),
    ],
)
def test_learn_intercellular_flows_validates_inputs(kwargs, message):
    adata = _make_adata()
    adata.obs["condition"] = pd.Categorical(
        np.repeat(["control", "perturbed"], adata.n_obs // 2)
    )

    with pytest.raises(ValueError, match=message):
        _network.learn_intercellular_flows(adata, **kwargs)


def test_flow_matrix_must_be_numeric_and_match_metadata():
    adata = _make_adata()
    adata.obsm["X_flow"] = np.full((adata.n_obs, 3), "bad", dtype=object)
    with pytest.raises(ValueError, match="numeric"):
        _network.learn_intercellular_flows(adata, n_bootstraps=1)

    adata = _make_adata()
    adata.uns["flowsig_network"]["flow_var_info"] = pd.DataFrame(
        index=["only_one"]
    )
    with pytest.raises(ValueError, match="does not match"):
        _network.learn_intercellular_flows(adata, n_bootstraps=1)


@pytest.mark.skipif(
    importlib.util.find_spec("causaldag") is None,
    reason="requires the omicverse[flowsig] optional dependencies",
)
def test_parallel_public_api_ignores_unpicklable_anndata_state():
    serial = _make_adata()
    parallel = serial.copy()
    serial.uns["unpicklable"] = _Unpicklable()
    parallel.uns["unpicklable"] = _Unpicklable()

    _network.learn_intercellular_flows(
        serial,
        use_spatial=True,
        block_key="block",
        n_jobs=1,
        n_bootstraps=2,
    )
    _network.learn_intercellular_flows(
        parallel,
        use_spatial=True,
        block_key="block",
        n_jobs=2,
        n_bootstraps=2,
    )

    serial_result = serial.uns["flowsig_network"]["network"]
    parallel_result = parallel.uns["flowsig_network"]["network"]
    assert serial_result["flow_vars"] == parallel_result["flow_vars"]
    np.testing.assert_allclose(
        serial_result["adjacency"], parallel_result["adjacency"]
    )
    assert serial_result["adjacency"].shape == (3, 3)


@pytest.mark.skipif(
    importlib.util.find_spec("causaldag") is None,
    reason="requires the omicverse[flowsig] optional dependencies",
)
def test_parallel_utigsp_matches_serial_targets_with_zero_variance_column():
    serial = _make_adata(n_obs=32, n_vars=4)
    serial.obsm["X_flow"][:, -1] = 1.0
    serial.obs["condition"] = pd.Categorical(
        np.repeat(["control", "perturbed"], serial.n_obs // 2)
    )
    parallel = serial.copy()
    parallel.uns["unpicklable"] = _Unpicklable()

    common = {
        "condition_key": "condition",
        "control_key": "control",
        "n_bootstraps": 2,
    }
    _network.learn_intercellular_flows(serial, n_jobs=1, **common)
    _network.learn_intercellular_flows(parallel, n_jobs=2, **common)

    serial_result = serial.uns["flowsig_network"]["network"]
    parallel_result = parallel.uns["flowsig_network"]["network"]
    assert serial_result["flow_vars"] == parallel_result["flow_vars"]
    np.testing.assert_allclose(
        serial_result["adjacency"], parallel_result["adjacency"]
    )
    np.testing.assert_allclose(
        serial_result["perturbed_targets"][0],
        parallel_result["perturbed_targets"][0],
    )
    np.testing.assert_array_equal(serial_result["adjacency"][-1], 0)
    np.testing.assert_array_equal(serial_result["adjacency"][:, -1], 0)


@pytest.mark.parametrize('interventional', [False, True])
@pytest.mark.parametrize('reverse_second,expected_support', [(False, 0.5), (True, 1.0)])
def test_threshold_counts_each_bootstrap_edge_once(monkeypatch, interventional, reverse_second, expected_support):
    pytest.importorskip('graphical_models')
    a = ad.AnnData(np.ones((24, 1)))
    a.obsm['X_flow'] = np.random.default_rng(0).normal(size=(24, 3))
    a.obs['condition'] = ['control'] * 12 + ['treated'] * 12
    a.uns['flowsig_network'] = {'flow_var_info': pd.DataFrame(
        {'Type': ['module'] * 3}, index=['GEM-1', 'GEM-2', 'constant'])}
    # Variable 2 was dropped in these bootstraps; mapping must stay aligned.
    def bootstrap(*args):
        seed = args[-1]
        matrix = (np.array([[0, 1], [0, 0]]) if seed == 0 else np.array([[0, 0], [1, 0]])) if reverse_second else (
            np.array([[0, 1], [1, 0]]) if seed == 0 else np.zeros((2, 2)))
        return {'nonzero_flow_vars_indices': np.array([0, 1]), 'adjacency_cpdag': matrix,
                'perturbed_targets_indices': [np.array([], dtype=int)]}
    monkeypatch.setattr(_network, 'run_gsp', bootstrap)
    monkeypatch.setattr(_network, 'run_utigsp', bootstrap)
    options = {'condition_key': 'condition', 'control_key': 'control'} if interventional else {}
    _network.learn_intercellular_flows(a, n_bootstraps=2, **options)
    apply_biological_flow(a)
    filter_low_confidence_edges(a, edge_threshold=0.8, adjacency_key='adjacency_validated')
    net = a.uns['flowsig_network']['network']
    assert bool(np.count_nonzero(net['adjacency_validated_filtered'])) == reverse_second
    assert net['edge_support'][0, 1] == expected_support
    assert not net['edge_support'][2].any()


@pytest.mark.parametrize('undirected', [False, True])
def test_legacy_network_requires_support_only_for_undirected_edges(undirected):
    pytest.importorskip('graphical_models')
    a = ad.AnnData(np.ones((2, 1)))
    adjacency = np.array([[0, 0.9], [0.9 if undirected else 0, 0]])
    a.uns['flowsig_network'] = {
        'flow_var_info': pd.DataFrame(index=['A', 'B']),
        'network': {'adjacency': adjacency},
    }
    if undirected:
        with pytest.raises(ValueError, match='rerun learn_intercellular_flows'):
            filter_low_confidence_edges(a, edge_threshold=0.8)
        assert 'adjacency_filtered' not in a.uns['flowsig_network']['network']
    else:
        filter_low_confidence_edges(a, edge_threshold=0.8)
        np.testing.assert_array_equal(
            a.uns['flowsig_network']['network']['adjacency_filtered'], adjacency)
