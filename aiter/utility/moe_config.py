# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Row ownership and shape identities for the shared FMoE config table.

This module has no GPU dependencies. Extra EP keys apply only to EP rows;
native FMoE rows keep their existing identity even in a mixed model CSV.
"""

EP_FP8_TAG = "flydsl_ep_decode_fp8_block128_gfx950"
EP_FP8_EXTRA_KEYS = ("global_expert", "source_tokens")
_NATIVE_KEYS = (
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)


def fmoe_config_tags(frame):
    import pandas as pd

    tags = frame.get("_tag", pd.Series("", index=frame.index))
    tags = tags.fillna("").replace({0: "", "0": ""}).astype(str)
    # Older EP-only CSVs have kernel names but no _tag column.
    names = frame.get("kernelName1", pd.Series("", index=frame.index))
    return tags.mask(
        names.fillna("").astype(str).str.startswith(EP_FP8_TAG + "_"), EP_FP8_TAG
    )


def fmoe_config_identity(frame, keys):
    """Return comparable keys without adding temporary columns to saved CSVs."""
    import pandas as pd

    columns = [name for name in dict.fromkeys((*_NATIVE_KEYS, *keys)) if name in frame]
    result = frame.loc[:, columns].copy()
    result["_tag"] = fmoe_config_tags(frame)
    ep = result["_tag"].eq(EP_FP8_TAG)
    for name in EP_FP8_EXTRA_KEYS:
        values = frame.get(name, pd.Series(-1, index=frame.index))
        # Source hint is optional. Global expert count must be specified for EP.
        values = pd.to_numeric(values.where(ep, -1), errors="raise").fillna(-1)
        if name == "global_expert" and (values[ep] <= 0).any():
            raise ValueError("EP FP8 config rows require global_expert")
        result[name] = values
    return result


def check_fmoe_fixture_conflicts(frame, identity):
    """Different routing measurements cannot be ranked by their saved latency."""
    import json

    if "tune_kwargs" not in frame:
        return
    duplicates = identity.duplicated(keep=False) & fmoe_config_tags(frame).eq(
        EP_FP8_TAG
    )
    candidates = identity.loc[duplicates].copy()
    if candidates.empty:
        return
    values = frame.loc[duplicates, "tune_kwargs"].fillna("{}")
    candidates["_fixture"] = values.map(
        lambda value: json.dumps(
            json.loads(value) if isinstance(value, str) else {}, sort_keys=True
        )
    )
    variants = candidates.groupby(list(identity.columns), dropna=False)[
        "_fixture"
    ].nunique()
    if (variants > 1).any():
        raise ValueError(
            "Conflicting EP FP8 tune_kwargs for the same shape; select one routing "
            "profile with AITER_CONFIG_FMOE or explicitly retune with --all."
        )
