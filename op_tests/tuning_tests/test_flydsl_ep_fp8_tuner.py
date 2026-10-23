# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP FP8 tuning policy and CSV integration; tests do not launch GPU kernels."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from csrc.ck_gemm_moe_2stages_codegen import flydsl_ep_fp8_tune as module
from aiter.ops.flydsl.kernels.moe_ep_decode_fp8_block128_gfx950.configuration import (
    MoeConfig,
    lookup_profile,
)


def shape(**updates):
    row = dict(
        gfx="gfx950",
        cu_num=256,
        token=64,
        model_dim=6144,
        inter_dim=2048,
        expert=48,
        global_expert=384,
        topk=8,
        source_tokens=8,
    )
    return dict(row, **updates)


@pytest.fixture
def tuner(monkeypatch):
    instance = module.EpFp8FlydslTuner()
    monkeypatch.setattr(instance, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(instance, "get_cu_num", lambda: 256)
    return instance


def arguments(tmp_path):
    return SimpleNamespace(
        tune_file=str(tmp_path / "tuned.csv"),
        profile_file=str(tmp_path / "profile.csv"),
        warmup=1,
        iters=2,
    )


def test_selects_fastest_valid_configuration_without_external_baselines(
    tuner, monkeypatch, tmp_path
):
    configs = [MoeConfig(), MoeConfig(stage2_n=128), MoeConfig(block_m=64)]
    monkeypatch.setattr(module, "tuning_candidates", lambda _: tuple(configs))
    monkeypatch.setattr(tuner, "_lookup", lambda *_: configs[0])
    monkeypatch.setattr(tuner, "_prepare_case", lambda _: object())

    def measure(row, config, args, case):
        if config == configs[1]:
            raise ValueError("unsupported candidate")
        return module.profile_row(row, config, 120 if config == configs[0] else 100)

    monkeypatch.setattr(tuner, "_run_candidate", measure)
    result = tuner._tune_one_shape(shape(), arguments(tmp_path))
    assert result["us"] == 100
    assert result["kernelName1"] == configs[2].kernel_names()[0]
    records = pd.read_csv(tmp_path / "profile.csv")
    assert len(records) == 3
    assert records.status.str.startswith("error:").sum() == 1
    assert not {"asm_us", "baseline_us", "worst_reference_ratio"} & set(records.columns)


def test_failed_tuning_keeps_previous_valid_profile(tuner, monkeypatch, tmp_path):
    args = arguments(tmp_path)
    previous = module.profile_row(shape(), MoeConfig(), 200)
    pd.DataFrame([previous]).to_csv(args.tune_file, index=False)
    monkeypatch.setattr(module, "tuning_candidates", lambda _: (MoeConfig(),))
    monkeypatch.setattr(tuner, "_prepare_case", lambda _: object())

    def fail(*_):
        raise ValueError("bad candidate")

    monkeypatch.setattr(tuner, "_run_candidate", fail)
    result = tuner._tune_one_shape(shape(), args)
    assert result["us"] == tuner.INVALID_TIME
    tuner.result_to_csv(pd.DataFrame([result]), args.tune_file)
    assert pd.read_csv(args.tune_file).us.tolist() == [200]
    assert len(tuner.failed) == 1


def test_profile_roundtrip_and_distinct_workload_keys(tuner, tmp_path):
    path = tmp_path / "tuned.csv"
    rows = [
        module.profile_row(shape(), MoeConfig(), 100),
        module.profile_row(shape(source_tokens=16), MoeConfig(stage2_n=128), 150),
        module.profile_row(shape(global_expert=768), MoeConfig(block_m=64), 110),
    ]
    tuner.result_to_csv(pd.DataFrame(rows), path)
    assert lookup_profile(path, 256, 64, 6144, 2048, 48, 384, 8, 16).stage2_n == 128
    assert len(pd.read_csv(path)) == 3
    # A slower retune is still this backend's result, not a comparison gate.
    tuner.result_to_csv(pd.DataFrame([dict(rows[0], us=180)]), path)
    assert (
        pd.read_csv(path).query("source_tokens == 8 and global_expert == 384").us.item()
        == 180
    )


def test_shape_validation_rejects_ambiguous_or_invalid_inputs(tuner):
    with pytest.raises(ValueError, match="one tuning fixture"):
        tuner._normalize_shapes(pd.DataFrame([shape(), shape()]))
    with pytest.raises(ValueError, match="Tuning uses R=C"):
        tuner._normalize_shapes(
            pd.DataFrame([shape(received_tokens=32, source_tokens=-1)])
        )
    with pytest.raises(ValueError, match="integer"):
        tuner._normalize_shapes(pd.DataFrame([shape(token=64.5)]))
    with pytest.raises(ValueError, match="Unsupported"):
        tuner._normalize_shapes(pd.DataFrame([shape(global_expert=48)]))


def test_standalone_config_merge_uses_declared_keys(tmp_path):
    from aiter.jit.core import AITER_CONFIG

    paths = [tmp_path / "one.csv", tmp_path / "two.csv"]
    for path, hint in zip(paths, [8, 16]):
        pd.DataFrame(
            [module.profile_row(shape(source_tokens=hint), MoeConfig(), 100)]
        ).to_csv(path, index=False)
    original = [p.read_bytes() for p in paths]
    config = AITER_CONFIG()
    config._cache_dir = tmp_path / "merged"
    merged = config.update_config_files(":".join(map(str, paths)), "tuned_fmoe")
    assert len(pd.read_csv(merged)) == 2
    assert [p.read_bytes() for p in paths] == original


def test_bounded_candidates_are_valid():
    for capacity in (8, 32, 672, 2048):
        candidates = module.tuning_candidates(capacity)
        assert len(candidates) == len(set(candidates)) <= 65
        for config in candidates:
            config.validate()


def test_explicit_hardware_rows_are_not_reinterpreted(tuner):
    frame = pd.DataFrame([shape(), shape(gfx="gfx942"), shape(cu_num=304)])
    selected = tuner._normalize_shapes(frame)
    assert len(selected) == 1
    assert selected.iloc[0].gfx == "gfx950"
    assert selected.iloc[0].cu_num == 256


@pytest.mark.parametrize(
    "value",
    [
        "[]",
        "null",
        '{"routing":"balanced"}',
        '{"received_tokens":32}',
        '{"routing":"hot","hot_experts":true}',
        '{"routing":"hot","hot_experts":0}',
        '{"routing":"hot","hot_experts":1.5}',
        '{"routing":"uniform","hot_experts":8}',
        '{"unknown":1}',
    ],
)
def test_invalid_tune_kwargs_are_rejected(value):
    with pytest.raises(ValueError):
        module.TuneOptions.parse(value)


def test_tune_kwargs_defaults_and_per_shape_hot_range():
    assert module.TuneOptions.parse("{}").routing == "uniform"
    hot = module.TuneOptions.parse('{"routing":"hot"}')
    assert hot.resolve(shape()).hot_experts == 8
    explicit = module.TuneOptions.parse('{"routing":"hot","hot_experts":32}')
    assert explicit.resolve(shape()).hot_experts == 32
    with pytest.raises(ValueError, match="exceeds local experts"):
        explicit.resolve(shape(expert=24))


def test_cli_kwargs_are_backend_local(tuner, monkeypatch):
    monkeypatch.setenv("TUNE_MOE_EXPERT_BALANCE", "True")
    monkeypatch.setattr("sys.argv", ["tune", "--ep-fp8-flydsl"])
    tuner.parse_args()
    assert tuner.tune_options is None  # Uniform default; no balanced-env override.
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune",
            "--ep-fp8-flydsl",
            "--ep-fp8-flydsl-tune-kwargs",
            '{"routing":"hot","hot_experts":4}',
        ],
    )
    tuner.parse_args()
    assert tuner.tune_options == module.TuneOptions("hot", 4)


def native_profile(**updates):
    row = module.profile_row(shape(), MoeConfig(), 80)
    row.update(
        _tag="",
        kernelName1="native_stage1",
        kernelName2="native_stage2",
        ksplit=0,
        run_1stage=0,
        native_note="preserve",
    )
    for key in ("source_tokens", "global_expert", "tune_kwargs"):
        row.pop(key)
    return dict(row, **updates)


def test_ep_write_and_sort_preserve_native_variants(tuner, tmp_path):
    path = tmp_path / "mixed.csv"
    native = [native_profile(), native_profile(dtype="torch.float16", us=90)]
    pd.DataFrame(native).to_csv(path, index=False)
    rows = [
        module.profile_row(shape(source_tokens=s), MoeConfig(), 100) for s in [8, 16]
    ]
    tuner.result_to_csv(pd.DataFrame(rows), path)
    tuner.sortResults(str(path), False, tuner.sort_keys)
    saved = pd.read_csv(path)
    assert len(saved) == 4
    assert saved.native_note.dropna().tolist() == ["preserve", "preserve"]
    assert len(tuner.get_tuned_gemm_list(str(path))) == 2
    options = '{"routing":"hot","hot_experts":8}'
    tuner.result_to_csv(
        pd.DataFrame([dict(rows[0], us=120, tune_kwargs=options)]), path
    )
    updated = pd.read_csv(path)
    assert len(updated) == 4
    row = updated.loc[updated.source_tokens.eq(8)].iloc[0]
    assert row.us == 120 and json.loads(row.tune_kwargs)["routing"] == "hot"


def test_shared_runtime_and_tuner_merges_preserve_extended_keys(tuner, tmp_path):
    from aiter.jit.core import AITER_CONFIG

    first, second = tmp_path / "native.csv", tmp_path / "ep.csv"
    pd.DataFrame([native_profile(), native_profile(dtype="torch.float16")]).to_csv(
        first, index=False
    )
    rows = [
        module.profile_row(shape(), MoeConfig(), 100),
        module.profile_row(shape(source_tokens=16), MoeConfig(), 100),
        module.profile_row(shape(global_expert=768), MoeConfig(), 100),
    ]
    pd.DataFrame(rows).to_csv(second, index=False)
    before = [p.read_bytes() for p in [first, second]]
    config = AITER_CONFIG()
    config._cache_dir = tmp_path / "runtime"
    files = f"{first}:{second}"
    assert len(pd.read_csv(config.update_config_files(files, "tuned_fmoe"))) == 5
    assert len(pd.read_csv(tuner.update_config_files(files, tuner.name))) == 5
    assert [p.read_bytes() for p in [first, second]] == before


def test_native_write_and_sort_preserve_ep_profiles(tmp_path):
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner
    from aiter.utility.moe_config import _NATIVE_KEYS

    result_columns = [
        "block_m",
        "ksplit",
        "us",
        "kernelName1",
        "kernelName2",
        "run_1stage",
    ]
    native = FmoeTuner("fmoeTuner", list(_NATIVE_KEYS), result_columns)
    path = tmp_path / "shared.csv"
    ep = [module.profile_row(shape(source_tokens=s), MoeConfig(), 100) for s in [8, 16]]
    pd.DataFrame(ep).to_csv(path, index=False)
    native.result_to_csv(pd.DataFrame([native_profile(us=75)]), str(path))
    native.sortResults(str(path), False, native.sort_keys)
    saved = pd.read_csv(path)
    assert len(saved) == 3
    assert sorted(saved.source_tokens.dropna().astype(int)) == [8, 16]
    assert len(native.get_tuned_gemm_list(str(path))) == 1


def test_replay_uses_saved_kwargs_and_allows_explicit_override(
    tuner, monkeypatch, tmp_path
):
    import aiter.jit.core as core

    saved = module.profile_row(shape(), MoeConfig(), 100)
    saved["tune_kwargs"] = '{"routing":"hot","hot_experts":4}'
    path = tmp_path / "profile.csv"
    pd.DataFrame([saved]).to_csv(path, index=False)
    tuner.untunedf = pd.DataFrame([saved])
    monkeypatch.setattr(
        core, "AITER_CONFIGS", SimpleNamespace(AITER_CONFIG_FMOE_FILE=str(path))
    )
    monkeypatch.setattr(tuner, "_prepare_case", lambda row: object())
    seen = []

    def measure(row, config, args, case, *, public):
        assert public
        seen.append(json.loads(row["tune_kwargs"]))
        return module.profile_row(row, config, 100)

    monkeypatch.setattr(tuner, "_run_candidate", measure)
    args = SimpleNamespace(run_config=str(path))
    tuner.run_config(args)
    tuner.run_config(SimpleNamespace(run_config=False))
    tuner.tune_options = module.TuneOptions("randn")
    tuner.run_config(args)
    assert seen == [
        {"routing": "hot", "hot_experts": 4},
        {"routing": "uniform"},
        {"routing": "randn"},
    ]


def test_replay_validates_requested_file_not_default_shape_list(tuner, tmp_path):
    path = tmp_path / "ep8.csv"
    pd.DataFrame([module.profile_row(shape(), MoeConfig(), 100)]).to_csv(
        path, index=False
    )
    tuner.tune_options = module.TuneOptions("hot", 32)
    args = SimpleNamespace(
        run_config=str(path),
        untune_file="unused.csv",
        tune_file=str(tmp_path / "not_created.csv"),
        last=False,
        all=False,
    )
    tuner.pre_process(args)
    assert len(tuner.untunedf) == 1
    assert tuner.untunedf.iloc[0].expert == 48


def test_compare_update_preserves_other_backend_rows(tuner, tmp_path):
    base, candidate = tmp_path / "base.csv", tmp_path / "candidate.csv"
    ep = module.profile_row(shape(), MoeConfig(), 100)
    pd.DataFrame([native_profile(), ep]).to_csv(base, index=False)
    pd.DataFrame([dict(ep, us=80)]).to_csv(candidate, index=False)
    comparison = pd.DataFrame([dict(shape(), update=True)])
    merged = tuner._merge_compare_filtered_results(
        str(base), str(candidate), comparison
    )
    assert len(merged) == 2
    assert sorted(merged.us) == [80, 80]
    assert merged.native_note.dropna().tolist() == ["preserve"]


def test_merging_different_routing_profiles_does_not_rewrite_sources(tuner, tmp_path):
    from aiter.jit.core import AITER_CONFIG

    paths = [tmp_path / "uniform.csv", tmp_path / "hot.csv"]
    for path, mode in zip(paths, ("uniform", "hot")):
        row = module.profile_row(shape(), MoeConfig(), 100)
        row["tune_kwargs"] = module.TuneOptions(mode).resolve(shape()).to_json()
        pd.DataFrame([row]).to_csv(path, index=False)
    before = [path.read_bytes() for path in paths]
    files = ":".join(map(str, paths))
    for owner, name in [(AITER_CONFIG(), "tuned_fmoe"), (tuner, tuner.name)]:
        with pytest.raises(ValueError, match="Conflicting EP FP8 tune_kwargs"):
            owner.update_config_files(files, name)
        assert [path.read_bytes() for path in paths] == before
