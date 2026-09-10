import importlib
from functools import partial

import pytest
import torch

fused_moe_mod = importlib.import_module("aiter.fused_moe")


@pytest.fixture(autouse=True)
def _clear_config_cache():
    fused_moe_mod.get_2stage_cfgs.cache_clear()
    yield
    fused_moe_mod.get_2stage_cfgs.cache_clear()


def test_fused_moe_forwards_ep_route_convention(monkeypatch):
    schema = str(torch.ops.aiter.fused_moe_.default._schema)
    assert "bool ep_has_fake_route=True" in schema

    captured = {}

    def fake_fused_moe_(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    monkeypatch.setattr(fused_moe_mod, "fused_moe_", fake_fused_moe_)
    hidden_states = torch.zeros((1, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)

    output = fused_moe_mod.fused_moe(
        hidden_states=hidden_states,
        w1=torch.empty((2, 8, 4), dtype=torch.bfloat16),
        w2=torch.empty((2, 4, 4), dtype=torch.bfloat16),
        topk_weight=torch.ones((1, 2), dtype=torch.float32),
        topk_ids=topk_ids,
        expert_mask=torch.ones(2, dtype=torch.int32),
        ep_has_fake_route=False,
    )

    assert output is hidden_states
    assert captured["ep_has_fake_route"] is False


def test_fused_moe_plumbs_route_convention_to_first_lookup_and_stage2(monkeypatch):
    lookup_flags = []
    stage2_flags = []

    def fake_get_2stage_cfgs(*args, **kwargs):
        lookup_flags.append(kwargs["ep_has_fake_route"])
        return fused_moe_mod.MOEMetadata(
            lambda *args, **kwargs: None, lambda: None, 16, 0
        )

    def fake_moe_sorting(*args, **kwargs):
        hidden_states = torch.zeros((1, 4), dtype=torch.bfloat16)
        empty_i32 = torch.empty(0, dtype=torch.int32)
        return empty_i32, torch.empty(0), empty_i32, empty_i32, hidden_states

    def fake_fused_moe_2stages(*args, **kwargs):
        stage2_flags.append(kwargs["ep_has_fake_route"])
        return args[0]

    monkeypatch.setattr(fused_moe_mod, "get_2stage_cfgs", fake_get_2stage_cfgs)
    monkeypatch.setattr(fused_moe_mod, "moe_sorting", fake_moe_sorting)
    monkeypatch.setattr(fused_moe_mod, "fused_moe_2stages", fake_fused_moe_2stages)

    output = fused_moe_mod.fused_moe(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        w1=torch.empty((2, 8, 4), dtype=torch.bfloat16),
        w2=torch.empty((2, 4, 4), dtype=torch.bfloat16),
        topk_weight=torch.ones((1, 2), dtype=torch.float32),
        topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
        expert_mask=torch.ones(2, dtype=torch.int32),
        ep_has_fake_route=False,
    )

    assert output.shape == (1, 4)
    assert lookup_flags == [False]
    assert stage2_flags == [False]


def test_fused_moe_stage2_forwards_route_convention_to_lookup(monkeypatch):
    lookup_flags = []

    def stage1(*args, **kwargs):
        return args[6]

    def stage2(*args, **kwargs):
        return None

    def fake_get_2stage_cfgs(*args, **kwargs):
        lookup_flags.append(kwargs["ep_has_fake_route"])
        return fused_moe_mod.MOEMetadata(
            partial(stage1), partial(stage2), block_m=16, ksplit=0
        )

    def fake_quant(hidden_states, **kwargs):
        return hidden_states, None

    monkeypatch.setattr(fused_moe_mod, "get_2stage_cfgs", fake_get_2stage_cfgs)
    monkeypatch.setattr(fused_moe_mod, "get_quant", lambda quant_type: fake_quant)

    hidden_states = torch.zeros((1, 4), dtype=torch.bfloat16)
    empty_i32 = torch.empty(0, dtype=torch.int32)
    output = fused_moe_mod.fused_moe_2stages(
        hidden_states=hidden_states,
        w1=torch.empty((2, 8, 4), dtype=torch.bfloat16),
        w2=torch.empty((2, 4, 4), dtype=torch.bfloat16),
        topk=2,
        sorted_ids=empty_i32,
        sorted_weights=torch.empty(0),
        sorted_expert_ids=empty_i32,
        num_valid_ids=empty_i32,
        moe_out=torch.zeros((1, 4), dtype=torch.bfloat16),
        isG1U1=True,
        block_size_M=16,
        quant_type=fused_moe_mod.QuantType.No,
        q_dtype_a=torch.bfloat16,
        q_dtype_w=torch.bfloat16,
        expert_mask=torch.ones(2, dtype=torch.int32),
        ep_has_fake_route=False,
    )

    assert output.shape == (1, 4)
    assert lookup_flags == [False]


def test_ep_config_lookup_supports_legacy_and_routed_only_topk(monkeypatch):
    cu_num = 304
    token = 64
    model_dim = 16
    inter_dim = 32
    experts = 4
    routed_topk = 8
    dtype = torch.bfloat16
    q_dtype_a = torch.bfloat16
    q_dtype_w = torch.bfloat16
    q_type = fused_moe_mod.QuantType.No
    activation = fused_moe_mod.ActivationType.Silu
    use_g1u1 = True
    doweight_stage1 = False

    key = (
        cu_num,
        token,
        model_dim,
        inter_dim,
        experts,
        routed_topk,
        activation,
        str(dtype),
        str(q_dtype_a),
        str(q_dtype_w),
        str(q_type),
        use_g1u1,
        doweight_stage1,
    )
    tuned = {
        "block_m": 37,
        "ksplit": 0,
        "kernelName1": "",
        "kernelName2": "",
        "run_1stage": False,
    }
    monkeypatch.setattr(fused_moe_mod, "cfg_2stages", ({key: tuned}, {}))
    monkeypatch.setattr(fused_moe_mod, "get_cu_num", lambda: cu_num)
    monkeypatch.setattr(fused_moe_mod, "is_flydsl_available", lambda: True)

    common = dict(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        expert=experts,
        dtype=dtype,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w,
        q_type=q_type,
        use_g1u1=use_g1u1,
        activation=activation,
        doweight_stage1=doweight_stage1,
        hidden_pad=0,
        intermediate_pad=0,
        is_ep=True,
    )

    legacy = fused_moe_mod.get_2stage_cfgs(
        topk=routed_topk + 1,
        ep_has_fake_route=True,
        **common,
    )
    routed_only = fused_moe_mod.get_2stage_cfgs(
        topk=routed_topk,
        ep_has_fake_route=False,
        **common,
    )

    assert legacy.block_m == tuned["block_m"]
    assert routed_only.block_m == tuned["block_m"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
