import triton
from functools import lru_cache


@lru_cache(maxsize=1)
def get_arch():
    try:
        arch = (
            triton.runtime.driver.active.get_current_target().arch
        )  # If running with torch
    except RuntimeError:  # else running with JAX
        from jax._src.lib import gpu_triton as triton_kernel_call_lib

        arch = triton_kernel_call_lib.get_arch_details("0")
        arch = arch.split(":")[0]

    return arch


def is_gluon_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_fp4_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_moe_a16w4_avail():
    """Return whether the BF16-activation/MXFP4-weight MoE path is supported."""
    return get_arch() in ("gfx942", "gfx950", "gfx1250")


def is_fp8_avail():
    return get_arch() in ("gfx942", "gfx950", "gfx1250", "gfx1200", "gfx1201")


def is_tdm_avail():
    """Return whether Triton tensor-descriptor memory operations are supported."""
    return get_arch() in ("gfx1250",)
