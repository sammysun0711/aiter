# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Canonical EP decode configuration and standard AITER CSV compatibility.

Only the CSV decoder understands legacy flags/defaults. GPU launchers receive
explicit cache/output choices and concrete separate-GEMM worker counts.
"""
import csv
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import lru_cache
from pathlib import Path
import re
from aiter.utility.moe_config import EP_FP8_TAG

TAG = EP_FP8_TAG
PREFIX = TAG + "_"
KEY_COLUMNS = (
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "global_expert",
    "topk",
    "source_tokens",
)


class WeightCache(IntEnum):
    STREAM = 0
    CACHE = 1
    ADAPTIVE = 2


class OutputMode(str, Enum):
    ATOMIC = "atomic"
    PARTIAL = "partial"


class Stage1(str, Enum):
    """Whether Gate/Up GEMM fuses SiLU and intermediate FP8 quantization.

    SEPARATE_FP8_QUANT writes FP32 Gate/Up, followed by a separate SiLU/quant kernel.
    """

    FUSED_FP8_QUANT = "fused_fp8_quant"
    FUSED_FP8_QUANT_PARALLEL = "fused_fp8_quant_parallel"
    SEPARATE_FP8_QUANT = "separate_fp8_quant"
    SEPARATE_FP8_QUANT_TRANSPOSED = "separate_fp8_quant_transposed"

    @property
    def uses_separate_quant(self):
        return self in (Stage1.SEPARATE_FP8_QUANT, Stage1.SEPARATE_FP8_QUANT_TRANSPOSED)

    @property
    def transposed(self):
        return self == Stage1.SEPARATE_FP8_QUANT_TRANSPOSED


@dataclass(frozen=True)
class MoeConfig:
    """One effective pipeline configuration.

    Worker counts are CU multiples; zero is allowed only for fused stage 1 and
    means one block per task. Cache policies never inherit another stage.
    """

    block_m: int = 32
    stage1: Stage1 = Stage1.FUSED_FP8_QUANT
    stage1_n: int = 64
    stage2_n: int = 64
    stage1_workers: int = 0
    stage2_workers: int = 8
    stage1_cache: WeightCache = WeightCache.STREAM
    stage2_cache: WeightCache = WeightCache.STREAM
    quant_tiled: bool = False
    output: OutputMode = OutputMode.PARTIAL
    down_adaptive_m: bool = False

    def __post_init__(self):
        for name, kind in (
            ("stage1", Stage1),
            ("output", OutputMode),
            ("stage1_cache", WeightCache),
            ("stage2_cache", WeightCache),
        ):
            object.__setattr__(self, name, kind(getattr(self, name)))
        if self.uses_separate_quant and self.stage1_workers == 0:
            object.__setattr__(self, "stage1_workers", 8)
        if not self.uses_separate_quant:
            object.__setattr__(self, "quant_tiled", False)

    @property
    def uses_separate_quant(self):
        return self.stage1.uses_separate_quant

    @property
    def uses_atomic_output(self):
        return self.output == OutputMode.ATOMIC

    def validate(self):
        if (
            self.block_m not in (16, 32, 64)
            or self.stage1_n
            not in ((16, 32, 64, 128) if self.uses_separate_quant else (16, 32, 64))
            or self.stage2_n not in (32, 64, 128)
            or self.stage1_workers not in (0, 1, 2, 3, 4, 6, 8, 16)
            or self.stage2_workers not in (2, 3, 4, 6, 8, 16)
            or (self.stage1 == Stage1.FUSED_FP8_QUANT_PARALLEL and self.stage1_n != 64)
        ):
            raise ValueError("Unsupported static MoE tuning configuration")

    def kernel_names(self):
        """Keep the existing CSV codec: ``proj`` denotes the SEPARATE_FP8_QUANT pipeline."""
        first = (
            f"{PREFIX}s1_m{self.block_m}_n{self.stage1_n}_{'proj' if self.uses_separate_quant else 'fused'}"
            f"_t{int(self.stage1.transposed)}_c{int(self.stage1_cache)}_w{self.stage1_workers}"
            f"_k1_a0_np{int(self.stage1 == Stage1.FUSED_FP8_QUANT_PARALLEL)}"
        )
        if self.stage1.transposed:
            first += "_sa1"
        if self.quant_tiled:
            first += "_qt1"
        second = (
            f"{PREFIX}s2_m{self.block_m}_n{self.stage2_n}_c{int(self.stage2_cache)}"
            f"_w{self.stage2_workers}_a1_am{int(self.down_adaptive_m)}"
        )
        second += "_rv8"
        second += f"_ao{int(self.uses_atomic_output)}"
        return first, second

    @classmethod
    def from_kernel_names(cls, first, second):
        """Decode old CSV flags once; all callers get the maintained pipeline.

        Old input-scale transpose, no-activation-prefetch and reduction-layout flags are retired
        tuning choices. Their CSV rows use row-major scales, full prefetch and vector8 reduction.
        The logical operation and input contract are unchanged.
        """
        one = re.fullmatch(
            re.escape(PREFIX)
            + r"s1_m(\d+)_n(\d+)_(proj|fused)_t([01])_c([012])_w(\d+)_k1_a([01])_np([01])(_sa1)?(_qt1)?",
            first,
        )
        two = re.fullmatch(
            re.escape(PREFIX)
            + r"s2_m(\d+)_n(\d+)_c([012])_w(\d+)_a([01])(?:_am([01]))?(_rc1)?(_fr1|_rv8)?(?:_ao([01]))?",
            second,
        )
        if not one or not two or one[1] != two[1]:
            raise ValueError("Invalid EP decode FP8 profile kernel names")
        if one[3] == "proj":
            # The maintained transposed projection also uses scalar base
            # addressing. Legacy independent flags name the same logical GEMM.
            stage = (
                Stage1.SEPARATE_FP8_QUANT_TRANSPOSED
                if int(one[4]) or one[9]
                else Stage1.SEPARATE_FP8_QUANT
            )
        else:
            if one[9] or one[10] or int(one[4]):
                raise ValueError("Projection-only flags in a fused profile")
            stage = (
                Stage1.FUSED_FP8_QUANT_PARALLEL
                if int(one[8])
                else Stage1.FUSED_FP8_QUANT
            )
        if one[3] == "proj" and int(one[8]):
            raise ValueError("Parallel-N applies only to fused stage 1")
        cache2 = WeightCache(int(two[3]))
        if two[7] and cache2 != WeightCache.CACHE:
            cache2 = WeightCache.ADAPTIVE
        atomic = int(one[1]) == 16 if two[9] is None else bool(int(two[9]))
        result = cls(
            block_m=int(one[1]),
            stage1=stage,
            stage1_n=int(one[2]),
            stage2_n=int(two[2]),
            stage1_cache=WeightCache(int(one[5])),
            stage2_cache=cache2,
            stage1_workers=int(one[6]),
            stage2_workers=int(two[4]),
            down_adaptive_m=bool(int(two[6] or 0)),
            quant_tiled=bool(one[10]),
            output=OutputMode.ATOMIC if atomic else OutputMode.PARTIAL,
        )
        result.validate()
        return result


def _integer(row, name, default=None):
    value = row.get(name, default)
    if value is None or value == "":
        value = default
    return int(float(value))


@lru_cache(None)
def _read_profiles(path):
    # Profiles are immutable during inference; restart after offline retuning.
    if not Path(path).is_file():
        return {}
    result = {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            # The model table may also hold native FMoE rows. Kernel prefixes
            # identify this backend, including old EP rows without a tag.
            if not all(
                row.get(c, "").startswith(PREFIX)
                for c in ("kernelName1", "kernelName2")
            ):
                continue
            if row.get("gfx", "gfx950") != "gfx950":
                continue
            if (
                row.get("dtype") != "torch.bfloat16"
                or row.get("q_dtype_a") != "torch.float8_e4m3fn"
                or row.get("q_dtype_w") != "torch.float8_e4m3fn"
                or row.get("q_type") != "QuantType.per_1x128"
                or row.get("act_type") != "ActivationType.Silu"
                or _integer(row, "use_g1u1") != 1
                or _integer(row, "doweight_stage1") != 0
            ):
                continue
            key = tuple(
                _integer(row, c, -1 if c == "source_tokens" else None)
                for c in KEY_COLUMNS
            )
            result[key] = MoeConfig.from_kernel_names(
                row["kernelName1"], row["kernelName2"]
            )
    return result


@lru_cache(None)
def lookup_profile(
    path,
    cu_num,
    capacity,
    hidden,
    intermediate,
    local_experts,
    global_experts,
    topk,
    source_tokens=None,
):
    if local_experts <= 0:
        return None
    profiles = _read_profiles(str(path))

    def key(tokens, source):
        return (
            cu_num,
            tokens,
            hidden,
            intermediate,
            local_experts,
            global_experts,
            topk,
            source,
        )

    if source_tokens is not None:
        exact = profiles.get(key(capacity, source_tokens))
        if exact is not None:
            return exact
        # Each source has its own token shard. This avoids selecting a large
        # workload merely because MORI allocated a larger receive buffer.
        workload = max(1, source_tokens * global_experts // local_experts)
        exact = profiles.get(key(workload, source_tokens))
        if exact is not None:
            return exact
    else:
        workload = capacity
    exact = profiles.get(key(workload, -1))
    if exact is not None:
        return exact
    # Do not apply a neighboring shape's profile without measuring it.
    return None


def default_config(source_tokens, topk, local_experts):
    """Untuned mean over all local experts, including inactive ones.

    P sources with m rows each and L local experts have mean
    (P*m)*topk/(P*L) = m*topk/L. Repeated routing changes variance, not this mean.
    This CPU hint never bounds the GPU receive count.
    """
    mean = None if source_tokens is None else source_tokens * topk / local_experts
    small = mean is not None and mean <= 8
    separate_quant = mean is not None and mean <= 4
    return MoeConfig(
        block_m=16 if small else 32,
        stage1=Stage1.SEPARATE_FP8_QUANT if separate_quant else Stage1.FUSED_FP8_QUANT,
        output=OutputMode.ATOMIC if small else OutputMode.PARTIAL,
    )
