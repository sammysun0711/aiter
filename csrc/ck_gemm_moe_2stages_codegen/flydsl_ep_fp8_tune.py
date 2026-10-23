# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Offline tuning of the gfx950 EP FP8 Block128 local-MoE implementation.

Only this backend is measured. Each candidate is a complete pipeline; the
lowest valid GPU-event latency wins. Test fixtures and timing are shared with
op_tests/test_moe_blockscale.py. This module is not imported during inference.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from aiter import logger
from aiter.jit.core import AITER_ROOT_DIR
from aiter.utility.base_tuner import TunerCommon, _read_csv
from aiter.utility.moe_config import EP_FP8_TAG, fmoe_config_tags
from aiter.ops.flydsl.kernels.moe_ep_decode_fp8_block128_gfx950.configuration import (
    KEY_COLUMNS,
    MoeConfig,
    Stage1,
    OutputMode,
    WeightCache,
    default_config,
    lookup_profile,
    _read_profiles,
)

KEYS = ("gfx", *KEY_COLUMNS)
RESULTS = (
    "block_m",
    "kernelName1",
    "kernelName2",
    "us",
    "global_nsd",
    "max_row_nsd",
    "tune_kwargs",
    "_tag",
    "metric",
    "status",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)
CONTRACT = dict(
    metric="cuda_graph_event_mean",
    act_type="ActivationType.Silu",
    dtype="torch.bfloat16",
    q_dtype_a="torch.float8_e4m3fn",
    q_dtype_w="torch.float8_e4m3fn",
    q_type="QuantType.per_1x128",
    use_g1u1=1,
    doweight_stage1=0,
)


@dataclass(frozen=True)
class TuneOptions:
    """Offline fixture options; none of these fields is a runtime lookup key."""

    routing: str = "uniform"
    hot_experts: int | None = None

    @classmethod
    def parse(cls, value):
        if value is None:
            return cls()
        fields = json.loads(value) if isinstance(value, str) else value
        if not isinstance(fields, dict):
            raise ValueError("tune kwargs must be a JSON object")
        unknown = fields.keys() - {"routing", "hot_experts"}
        if unknown:
            raise ValueError(f"Unknown tune kwargs: {sorted(unknown)}")
        result = cls(**fields)
        if result.routing not in ("randn", "uniform", "hot"):
            raise ValueError("routing must be randn, uniform or hot")
        if result.hot_experts is not None and (
            result.routing != "hot"
            or type(result.hot_experts) is not int
            or result.hot_experts < 1
        ):
            raise ValueError("hot_experts requires hot routing and a positive integer")
        return result

    def resolve(self, row):
        if self.routing != "hot":
            return self
        count = self.hot_experts
        if count is None:
            count = min(int(row["topk"]), int(row["expert"]))
        if count > int(row["expert"]):
            raise ValueError(
                f"hot_experts={count} exceeds local experts={row['expert']}"
            )
        return TuneOptions("hot", count)

    def to_json(self):
        return json.dumps(
            {key: value for key, value in asdict(self).items() if value is not None},
            sort_keys=True,
            separators=(",", ":"),
        )


def _separate_quant_candidates():
    """Measured narrow/wide families; no Cartesian expansion of inactive flags."""
    for n2 in (32, 64):
        for w1 in (6, 8):
            for w2 in (6, 8):
                for output in OutputMode:
                    yield MoeConfig(
                        stage1=Stage1.SEPARATE_FP8_QUANT,
                        stage1_workers=w1,
                        stage2_workers=w2,
                        stage2_n=n2,
                        stage1_cache=WeightCache.ADAPTIVE,
                        quant_tiled=True,
                        down_adaptive_m=True,
                        output=output,
                    )
    for block_m in (32, 64):
        for n2 in (64, 128):
            for workers in (6, 8):
                for cache in (WeightCache.STREAM, WeightCache.ADAPTIVE):
                    for output in OutputMode:
                        yield MoeConfig(
                            block_m=block_m,
                            stage1=Stage1.SEPARATE_FP8_QUANT_TRANSPOSED,
                            stage1_n=128,
                            stage2_n=n2,
                            stage1_workers=4,
                            stage2_workers=workers,
                            stage1_cache=cache,
                            quant_tiled=True,
                            down_adaptive_m=True,
                            output=output,
                        )


def tuning_candidates(capacity):
    """A small family-based search; the caller also measures its current profile.

    No routing label enters a configuration. Candidate selection uses only this backend
    on the specified fixture; cross-backend comparisons belong in benchmarks.
    """
    candidates = [
        MoeConfig(),
        MoeConfig(block_m=16, output=OutputMode.ATOMIC),
        *[
            MoeConfig(
                block_m=16,
                stage1=Stage1.SEPARATE_FP8_QUANT,
                stage1_n=n,
                output=OutputMode.ATOMIC,
            )
            for n in (32, 64)
        ],
    ]
    if capacity >= 32:
        candidates.extend(_separate_quant_candidates())
        for n2 in (64, 128):
            for workers in (4, 6, 8):
                for output in OutputMode:
                    candidates.append(
                        MoeConfig(
                            stage1_workers=4,
                            stage2_n=n2,
                            stage2_workers=workers,
                            stage1_cache=WeightCache.ADAPTIVE,
                            stage2_cache=WeightCache.ADAPTIVE,
                            output=output,
                        )
                    )
    if capacity >= 512:
        # Keep the large-M parallel-quant family for larger workloads.
        candidates.append(
            MoeConfig(
                block_m=64,
                stage1=Stage1.FUSED_FP8_QUANT_PARALLEL,
                stage2_n=128,
                stage1_cache=WeightCache.CACHE,
                stage2_cache=WeightCache.CACHE,
                down_adaptive_m=True,
            )
        )
    return tuple(dict.fromkeys(candidates))


def profile_row(row, config, us, errors=None, status="ok"):
    first, second = config.kernel_names()
    return dict(
        {key: row[key] for key in KEYS},
        **CONTRACT,
        block_m=config.block_m,
        kernelName1=first,
        kernelName2=second,
        us=us,
        global_nsd=(errors or {}).get("global_nsd", math.nan),
        max_row_nsd=(errors or {}).get("max_row_nsd", math.nan),
        tune_kwargs=row.get("tune_kwargs", TuneOptions().to_json()),
        _tag=EP_FP8_TAG,
        status=status,
    )


class EpFp8FlydslTuner(TunerCommon):
    ARG_DEFAULTS = {
        **TunerCommon.ARG_DEFAULTS,
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/{EP_FP8_TAG}_untuned_fmoe.csv",
        "tune_file": f"{AITER_ROOT_DIR}/aiter/configs/{EP_FP8_TAG}_tuned_fmoe.csv",
        "config_env_name": "AITER_CONFIG_FMOE",
        "timeout": None,
        "errRatio": 1e-3,
    }

    def __init__(self):
        super().__init__("flydsl_ep_fp8", list(KEYS), list(RESULTS))
        self.tune_options = None

    def _setup_specific_arguments(self):
        self.parser.add_argument("--ep-fp8-flydsl", action="store_true")
        self.parser.add_argument(
            "--ep-fp8-flydsl-tune-kwargs",
            default=None,
            metavar="JSON",
            help='Fixture options: {"routing":"uniform"} or {"routing":"hot","hot_experts":8}. '
            "Default: uniform; --run_config reuses saved options unless overridden.",
        )
        self.parser.add_argument(
            "--last", action="store_true", help="Tune the last input row"
        )
        self.parser.set_defaults(mp=1, errRatio=1e-3)
        help_text = {
            "--mp": "Number of tuning devices; this backend currently supports 1",
            "--errRatio": "Fixed EP FP8 global NSD limit 1e-3; row NSD limit is 2e-3",
            "--warmup": "Untimed rounds of the four address sets before graph capture",
            "--iters": "Rounds of four address sets in each captured graph; five replays are averaged",
            "--timeout": "Reserved for isolated workers; unsupported by this single-process tuner",
        }
        for option, text in help_text.items():
            self.parser._option_string_actions[option].help = text

    def parse_args(self):
        args = super().parse_args()
        if args.ep_fp8_flydsl_tune_kwargs is not None:
            try:
                self.tune_options = TuneOptions.parse(args.ep_fp8_flydsl_tune_kwargs)
            except (TypeError, ValueError) as exc:
                self.parser.error(str(exc))
        if args.mp != 1:
            self.parser.error("The EP FP8 tuner currently supports --mp 1")
        if args.splitK:
            self.parser.error("EP FP8 candidates do not use split-K")
        if args.warmup < 1 or args.iters < 1:
            self.parser.error("--warmup and --iters must be positive")
        if args.errRatio != 1e-3:
            self.parser.error(
                "EP FP8 validation keeps fixed global/row NSD limits 1e-3/2e-3"
            )
        if args.timeout is not None:
            self.parser.error(
                "The single-process EP FP8 tuner does not support --timeout"
            )
        if os.pathsep in args.tune_file or os.pathsep in args.untune_file:
            self.parser.error(
                "-i and -o each require one CSV file; runtime config lists are not output paths"
            )
        if (
            args.profile_file
            and Path(args.profile_file).resolve() == Path(args.tune_file).resolve()
        ):
            self.parser.error("-o and -o2 must be different files")
        return args

    def _normalize_shapes(self, frame):
        frame = frame.copy()
        if "kernelName1" in frame:
            frame = frame.loc[fmoe_config_tags(frame).eq(EP_FP8_TAG)].copy()
        gfx, cu = self.get_gfx(), self.get_cu_num()
        if gfx != "gfx950":
            raise ValueError("The EP FP8 Block128 backend requires gfx950")
        for key, value in (("gfx", gfx), ("cu_num", cu)):
            if key in frame:
                frame = frame.loc[frame[key].isna() | frame[key].eq(value)].copy()
            frame[key] = value
        if "source_tokens" not in frame:
            frame["source_tokens"] = -1
        frame["source_tokens"] = frame["source_tokens"].fillna(-1)
        # Tuning uses the input shape's full valid range. Dynamic counts remain
        # covered by operator tests; users need not observe or supply R.
        if "received_tokens" in frame:
            received = frame["received_tokens"].fillna(frame["token"])
            if not received.eq(frame["token"]).all():
                raise ValueError("Tuning uses R=C; test R<C in test_moe_blockscale.py")
        for key in KEY_COLUMNS:
            values = pd.to_numeric(frame[key], errors="raise")
            if values.isna().any() or (values % 1 != 0).any():
                raise ValueError(f"Expected integer shape column: {key}")
            frame[key] = values.astype("int64")
        for row in frame.to_dict("records"):
            c, e, g = (row[k] for k in ("token", "expert", "global_expert"))
            if not (
                0 < c <= 2048
                and 0 < e < g
                and g % e == 0
                and row["model_dim"] > 0
                and row["model_dim"] % 1024 == 0
                and row["inter_dim"] > 0
                and row["inter_dim"] % 128 == 0
                and 1 <= row["topk"] <= min(32, g)
                and row["source_tokens"] >= -1
            ):
                raise ValueError(f"Unsupported EP FP8 input shape: {row}")
            (self.tune_options or TuneOptions()).resolve(row)
            for key, value in CONTRACT.items():
                if key == "metric":
                    continue  # Measurement metadata is not an input-format constraint.
                if key in row and pd.notna(row[key]) and str(row[key]) != str(value):
                    raise ValueError(
                        f"Unsupported {key}={row[key]!r}; expected {value!r}"
                    )
        if frame.duplicated(subset=list(KEYS)).any():
            raise ValueError(
                "Each runtime profile key must have one tuning fixture; duplicate keys would overwrite results"
            )
        metadata = ["tune_kwargs"] if "tune_kwargs" in frame else []
        return frame[list(KEYS) + metadata]

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        frame = super().get_tuned_gemm_list(tuned_gemm_file, columns or [])
        return frame.loc[fmoe_config_tags(frame).eq(EP_FP8_TAG)].copy()

    def pre_process(self, args):
        input_file = (
            args.run_config if isinstance(args.run_config, str) else args.untune_file
        )
        self.untunedf = self._normalize_shapes(_read_csv(input_file))
        output = Path(args.tune_file)
        self.tunedf = self.get_tuned_gemm_list(str(output))
        if args.last:
            self.untunedf = self.untunedf.tail(1)
        elif not args.all and not args.run_config and not self.tunedf.empty:
            existing = set(map(tuple, self.tunedf[list(KEYS)].values))
            mask = self.untunedf[list(KEYS)].apply(tuple, axis=1).isin(existing)
            self.untunedf = self.untunedf.loc[~mask]

    @staticmethod
    def _source(row):
        return None if int(row["source_tokens"]) == -1 else int(row["source_tokens"])

    def _prepare_case(self, row):
        from aiter.utility.moe_blockscale_benchmark import make_blockscale_case

        options = TuneOptions.parse(row.get("tune_kwargs")).resolve(row)
        return make_blockscale_case(
            torch.bfloat16,
            int(row["token"]),
            int(row["model_dim"]),
            int(row["inter_dim"]),
            int(row["global_expert"]),
            int(row["topk"]),
            ep=int(row["global_expert"]) // int(row["expert"]),
            routing=options.routing,
            hot_experts=options.hot_experts,
        )

    def _lookup(self, row, path):
        return lookup_profile(
            path,
            int(row["cu_num"]),
            int(row["token"]),
            int(row["model_dim"]),
            int(row["inter_dim"]),
            int(row["expert"]),
            int(row["global_expert"]),
            int(row["topk"]),
            self._source(row),
        )

    def _run_candidate(self, row, config, args, case, *, public=False):
        from aiter.utility.moe_blockscale_benchmark import (
            benchmark_ep_moe,
            check_blockscale_output,
        )
        from aiter.ops.flydsl import moe_ep_decode_fp8_block128_gfx950 as backend
        import importlib

        source = self._source(row)
        impl = backend.fused_moe_ep_decode_fp8_block128_gfx950
        fn, inputs = impl, case.flydsl_args()
        options = dict(config=config, num_tokens_before_dispatch=source)
        if public:
            fm = importlib.import_module("aiter.fused_moe")
            fn, inputs, options = fm.fused_moe, case.inputs, case.public_kwargs(source)
            with patch.object(
                fm, "_USE_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950", True
            ), patch.object(
                backend, "fused_moe_ep_decode_fp8_block128_gfx950", wraps=impl
            ) as selected:
                eager = fn(*inputs, **options)
                if not selected.called:
                    raise RuntimeError(
                        "Public dispatch did not select the EP FP8 backend"
                    )
                actual = selected.call_args.kwargs["config"] or default_config(
                    source, row["topk"], row["expert"]
                )
                if actual.kernel_names() != config.kernel_names():
                    raise RuntimeError("Public dispatch selected a different profile")
                errors = check_blockscale_output(
                    case.reference, eager[: case.received], "tune eager", case.ep
                )
                output, us = benchmark_ep_moe(
                    fn, inputs, options, warmup=args.warmup, iters=args.iters
                )
        else:
            eager = fn(*inputs, **options)
            errors = check_blockscale_output(
                case.reference, eager[: case.received], "tune eager", case.ep
            )
            output, us = benchmark_ep_moe(
                fn, inputs, options, warmup=args.warmup, iters=args.iters
            )
        # References and reductions for validation are outside the event interval.
        replay_errors = check_blockscale_output(
            case.reference, output[: case.received], "tune graph", case.ep
        )
        errors = {key: max(value, replay_errors[key]) for key, value in errors.items()}
        return profile_row(row, config, us, errors)

    @staticmethod
    def _write_measurements(path, rows):
        if not path or not rows:
            return
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        if output.exists() and list(_read_csv(output, nrows=0)) != list(frame):
            raise ValueError("Candidate CSV schema differs; use a new -o2 file")
        frame.to_csv(output, mode="a", header=not output.exists(), index=False)

    def _tune_one_shape(self, row, args):
        row = dict(
            row, tune_kwargs=(self.tune_options or TuneOptions()).resolve(row).to_json()
        )
        case = self._prepare_case(row)
        config = self._lookup(row, args.tune_file) or default_config(
            self._source(row), row["topk"], row["expert"]
        )
        candidates = tuple(dict.fromkeys((config, *tuning_candidates(row["token"]))))
        best, records = None, []
        for candidate in candidates:
            try:
                result = self._run_candidate(row, candidate, args, case)
                if not math.isfinite(result["us"]) or result["us"] <= 0:
                    raise ValueError("Invalid GPU-event latency")
                if best is None or result["us"] < best["us"]:
                    best = result
            except Exception as exc:
                from aiter.utility.mp_tuner import _is_accelerator_error

                if _is_accelerator_error(exc):
                    raise
                result = profile_row(
                    row, candidate, self.INVALID_TIME, status=f"error: {exc}"
                )
                logger.warning("EP FP8 candidate failed: %s", exc)
            records.append(result)
            logger.info(
                "[tune] C=%s kwargs=%s %s us=%s status=%s",
                row["token"],
                row["tune_kwargs"],
                candidate.kernel_names(),
                result["us"],
                result["status"],
            )
        self._write_measurements(args.profile_file, records)
        return best or profile_row(
            row, config, self.INVALID_TIME, status="error: no valid candidate"
        )

    def tune(self, untunedf, tunedf, args):
        del tunedf
        return [self._tune_one_shape(row, args) for row in untunedf.to_dict("records")]

    def post_process(self, results, args, topk=-1, fast_mode=False):
        return pd.DataFrame(results, columns=self.columns)

    def result_to_csv(self, results, file, concat=False):
        output = Path(file)
        output.parent.mkdir(parents=True, exist_ok=True)
        old = (
            _read_csv(output)
            if output.is_file()
            else pd.DataFrame(columns=self.columns)
        )
        valid = results[results["status"].eq("ok") & results["us"].gt(0)]
        self.success = pd.concat([self.success, valid], ignore_index=True)
        self.failed = pd.concat(
            [self.failed, results.drop(valid.index)], ignore_index=True
        )
        owned = fmoe_config_tags(old).eq(EP_FP8_TAG)
        merged = pd.concat([old.loc[owned], valid], ignore_index=True).drop_duplicates(
            subset=list(KEYS), keep="last"
        )
        merged["_tag"] = EP_FP8_TAG
        merged = pd.concat([old.loc[~owned], merged], ignore_index=True)
        columns = list(dict.fromkeys([*old.columns, *self.columns]))
        temporary = output.with_suffix(output.suffix + ".tmp")
        merged.reindex(columns=columns).to_csv(temporary, index=False)
        temporary.replace(output)
        self._clear_op_caches()

    def _clear_op_caches(self):
        _read_profiles.cache_clear()
        lookup_profile.cache_clear()
        from aiter.jit.core import AITER_CONFIGS

        AITER_CONFIGS.get_config_file.cache_clear()
        import importlib

        fm = importlib.import_module("aiter.fused_moe")
        fm.cfg_2stages = None
        fm.get_2stage_cfgs.cache_clear()

    def _set_config_env_for_run_config(self, args, config_file=None):
        # FlyDSL profile changes do not require rebuilding native AITER modules.
        name = self.ARG_DEFAULTS["config_env_name"]
        previous = os.environ.get(name)
        os.environ[name] = str(config_file or args.tune_file)
        self._clear_op_caches()
        return previous, 0

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        if old_val is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_val
        self._clear_op_caches()

    def run_config(self, args):
        from aiter.jit.core import AITER_CONFIGS

        path = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
        results = []
        for row in self._normalize_shapes(self.untunedf).to_dict("records"):
            # A direct replay reproduces its saved fixture. The inherited
            # --compare workflow must use one selected fixture before/after.
            saved = row.get("tune_kwargs") if args.run_config else None
            if not isinstance(saved, str) or not saved:
                saved = None
            options = self.tune_options or TuneOptions.parse(saved)
            row["tune_kwargs"] = options.resolve(row).to_json()
            config = self._lookup(row, path)
            if isinstance(args.run_config, str) and config is None:
                raise ValueError("No matching EP FP8 profile in the requested CSV")
            config = config or default_config(
                self._source(row), row["topk"], row["expert"]
            )
            result = self._run_candidate(
                row, config, args, self._prepare_case(row), public=True
            )
            results.append(
                dict(
                    shape=str(tuple(row[k] for k in KEYS)),
                    e2e_us=result["us"],
                    status="ok",
                )
            )
        return results


def main():
    tuner = EpFp8FlydslTuner()
    tuner.run(tuner.parse_args(), False)


if __name__ == "__main__":
    main()
