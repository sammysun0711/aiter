# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune spaces for the gfx950 8-wave and gfx942 4-wave FP8 blockscale GEMMs.

On gfx950 IDs 0 and 1 select plain and preshuffled B with the fixed 8-wave
pipeline. On gfx942 each tile has a plain and a preshuffled variant. The
output tensor specializes the epilogue for BF16 or FP16.
"""

from dataclasses import dataclass

TILE_M = 256
TILE_N = 256
TILE_K = 128
KERNEL_PREFIX = "flydsl_blockscale_8w_"


@dataclass(frozen=True)
class kernelInstance:
    preshuffle_b: bool
    gfx: str = "gfx950"
    tile_m: int = TILE_M
    tile_n: int = TILE_N
    variant: str = ""

    @property
    def name(self) -> str:
        if self.gfx == "gfx942":
            suffix = f"_{self.variant}" if self.variant else ""
            return f"flydsl_blockscale_gfx942_{self.tile_m}x{self.tile_n}x128_ps{int(self.preshuffle_b)}{suffix}"
        return (
            f"{KERNEL_PREFIX}{TILE_M}x{TILE_N}x{TILE_K}_F8_F8_B16_"
            f"ps{int(self.preshuffle_b)}"
        )


# Candidate IDs are consecutive; --preshuffle selects the B layout.
kernels_list = {
    0: kernelInstance(False),
    1: kernelInstance(True),
}
for tm, tn in ((32, 64), (64, 64), (64, 128), (128, 64), (128, 128)):
    for packed in (False, True):
        kernels_list[len(kernels_list)] = kernelInstance(packed, "gfx942", tm, tn)
# Append packed candidates so existing tuned IDs keep their meaning.
for variant in ("lds_m", "lds_n", "reg_a"):
    kernels_list[len(kernels_list)] = kernelInstance(True, "gfx942", 128, 128, variant)
kernels_by_name = {ki.name: ki for ki in kernels_list.values()}


def lds_bytes(K: int) -> int:
    # Eight padded 128x128 FP8 buffers and two 512-f32 A-scale buffers.
    # ScaleB uses scalar global loads; LDS no longer grows with K.
    return 8 * 16896 + 2 * 512 * 4


def kernel_fits_shape(ki: kernelInstance, M: int, N: int, K: int, gfx: str) -> bool:
    if gfx != ki.gfx:
        return False
    if gfx == "gfx942":
        # Exact register tiles, i32 dimensions, i64 matrix addresses. Flattened
        # scale tensors also have to fit the FlyDSL launcher extent ABI.
        return (
            min(M, N, K) > 0
            and M % ki.tile_m == 0
            and N % ki.tile_n == 0
            and K % 128 == 0
            and (ki.variant != "reg_a" or 128 * K < 2**31)
            and max(M, N, K, M * (K // 128), ((N + 127) // 128) * (K // 128)) < 2**31
        )
    if gfx != "gfx950" or M <= 0 or N <= 0 or K < 256 or K % 256:
        return False
    if N % (16 if ki.preshuffle_b else 8):
        return False
    if lds_bytes(K) > 160 * 1024:
        return False
    # A/B/C bases are rebased in i64 per WG. Only local descriptor-relative
    # spans must fit signed i32, including the two speculative K prefetches.
    padded_m = (M + TILE_M - 1) // TILE_M * TILE_M
    padded_n = (N + TILE_N - 1) // TILE_N * TILE_N
    # Scale bases were not rebased. Keep their full extents/prefetch offsets
    # and the launch-grid arithmetic in range; do not restore a matrix-size cap.
    kb = K // 128
    return (
        max(
            TILE_M * K,
            TILE_N * K,
            TILE_M * N * 2,
            kb * padded_m * 4,
            (padded_n // 128) * kb * 4,
        )
        < 2**31
    )
